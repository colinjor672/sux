extends Node
## 共享内存视频读取器（Python → Godot）
##
## 双模式自动切换：
##   GDExtension 可用: ShmSync (mmap + semaphore) — 零文件I/O、零CPU轮询
##   GDExtension 不可用: FileAccess 回退 — 兼容 Windows 编辑器/导出

signal video_frame_received(frame_id: int, width: int, height: int, img: Image)

const INPUT_W: int = 640
const INPUT_H: int = 360
const INPUT_SIZE: int = INPUT_W * INPUT_H * 4

# FileAccess 回退路径（Jetson 用 /dev/shm/，Windows 用 user://）
var SHM_CTRL: String = "/dev/shm/godot_input_ctrl"
var SHM_SLOT_0: String = "/dev/shm/godot_input_0.raw"
var SHM_SLOT_1: String = "/dev/shm/godot_input_1.raw"

var _frame_count: int = 0
var _err_count: int = 0
var _started: bool = false
var _use_gdext: bool = false

# GDExtension 对象（无类型标注，兼容编辑器）
var _shm = null
# FileAccess 回退状态
var _poll_timer: float = 0.0
var _last_write_index: int = -1
var _retry_count: int = 0
var _shm_ready: bool = false

# ImageTexture 复用
var _tex: ImageTexture = null


func _ready() -> void:
	# Windows 上 /dev/shm/ 不可用，使用 user:// 目录（不影响 Jetson）
	if OS.has_feature("windows"):
		SHM_CTRL = "user://godot_input_ctrl"
		SHM_SLOT_0 = "user://godot_input_0.raw"
		SHM_SLOT_1 = "user://godot_input_1.raw"
		print("[ShmVideoReader] Windows 平台：使用 user:// 目录作为文件回退路径")

	# 检测 GDExtension 是否可用
	_use_gdext = ClassDB.class_exists("ShmSync")

	if _use_gdext:
		_init_gdext()
	else:
		_init_file_fallback()


func _init_gdext() -> void:
	_shm = ClassDB.instantiate("ShmSync")
	add_child(_shm)

	# Reader 模式：等待 Python Writer 创建 shm
	# _process 中轮询重试，不阻塞启动
	_retry_count = 0
	_started = true  # 先标记 started，让 _process 开始运行
	print("[ShmVideoReader] GDExt 模式：等待 Python 创建 shm_input...")


func _init_file_fallback() -> void:
	_started = true
	print("[ShmVideoReader] ⚠ GDExtension 不可用，回退 FileAccess 轮询模式 %dx%d" % [INPUT_W, INPUT_H])


func _exit_tree() -> void:
	_started = false
	if _shm:
		if _shm.has_method("close_shm"):
			_shm.close_shm()
		_shm.queue_free()
		_shm = null


func _process(delta: float) -> void:
	if not _started:
		return

	if not _shm_ready and _use_gdext:
		# 轮询重试打开 shm_input（每 200ms 尝试一次）
		_retry_count += 1
		if _retry_count % 40 == 0:  # 约每 200ms (40帧 @ 5ms/帧)
			if _shm.open_shm("shm_input", INPUT_SIZE, false):
				_shm_ready = true
				print("[ShmVideoReader] mmap+semaphore Reader 已启动 %dx%d" % [INPUT_W, INPUT_H])
			elif _retry_count >= 1200:  # ~6秒超时
				push_warning("[ShmVideoReader] ShmSync 超时，回退 FileAccess")
				_use_gdext = false
		if not _shm_ready:
			return

	if _use_gdext:
		_process_gdext(delta)
	else:
		_process_file(delta)


func _process_gdext(_delta: float) -> void:
	# semaphore 阻塞等待新帧（替代 5ms 轮询，CPU 零开销）
	if not _shm.wait_for_new_frame(100):
		return  # 超时，无新帧

	_read_and_emit(_shm.get_frame_id(), _shm.get_data())


func _process_file(delta: float) -> void:
	_poll_timer += delta
	if _poll_timer < 0.005:
		return
	_poll_timer = 0.0

	# 读控制结构
	var ctrl_file := FileAccess.open(SHM_CTRL, FileAccess.READ)
	if ctrl_file == null:
		return
	var ctrl_data := ctrl_file.get_buffer(8)
	ctrl_file.close()
	if ctrl_data.size() < 8:
		return

	var write_index := ctrl_data.decode_u32(0)
	var frame_id := ctrl_data.decode_u32(4)
	if write_index == _last_write_index:
		return
	_last_write_index = write_index

	# 读最新槽
	var slot_path := SHM_SLOT_0 if (write_index % 2) == 0 else SHM_SLOT_1
	var slot_file := FileAccess.open(slot_path, FileAccess.READ)
	if slot_file == null:
		return
	var rgba_data := slot_file.get_buffer(INPUT_SIZE)
	slot_file.close()

	_read_and_emit(frame_id, rgba_data)


func _read_and_emit(frame_id: int, rgba_data: PackedByteArray) -> void:
	_frame_count += 1

	if _frame_count % 200 == 0:
		print("[ShmVideoReader] 心跳 frames=%d err=%d mode=%s" %
			[_frame_count, _err_count, "gdext" if _use_gdext else "file"])

	if rgba_data.size() != INPUT_SIZE:
		_err_count += 1
		if _err_count <= 3:
			push_warning("[ShmVideoReader] 数据大小异常: %d != %d" % [rgba_data.size(), INPUT_SIZE])
		return

	var img := Image.create_from_data(INPUT_W, INPUT_H, false, Image.FORMAT_RGBA8, rgba_data)
	if img == null or img.is_empty():
		_err_count += 1
		if _err_count <= 3:
			push_warning("[ShmVideoReader] Image 创建失败")
		return

	# 首帧数据检查
	if _frame_count <= 1:
		var all_zero := true
		for j in range(mini(rgba_data.size(), 4096)):
			if rgba_data[j] != 0:
				all_zero = false
				break
		print("[ShmVideoReader] 首帧 frame_id=%d all_zero=%s" % [frame_id, all_zero])

	video_frame_received.emit(frame_id, INPUT_W, INPUT_H, img)


## 供 VideoDisplay 调用：复用 ImageTexture
func update_texture(texture_rect: TextureRect, img: Image) -> void:
	if img == null or img.is_empty():
		return

	if _tex == null or _tex.get_width() != img.get_width() or _tex.get_height() != img.get_height():
		_tex = ImageTexture.create_from_image(img)
	else:
		_tex.update(img)

	texture_rect.texture = _tex
