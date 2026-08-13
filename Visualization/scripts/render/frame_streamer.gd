extends Node

const STREAM_WIDTH  : int = 640
const STREAM_HEIGHT : int = 360
const STREAM_FPS    : int = 20

# FileAccess 回退路径
const FRAME_CTRL : String = "/dev/shm/godot_frame_ctrl"
const FRAME_0    : String = "/dev/shm/godot_frame_0.raw"
const FRAME_1    : String = "/dev/shm/godot_frame_1.raw"

var _is_streaming   : bool = false
var _frame_acc      : float = 0.0
var _interval       : float = 1.0 / float(STREAM_FPS)
var _frame_count    : int = 0
var _capture_vp     : SubViewport = null
var _use_gdext      : bool = false

# GDExtension 对象（无类型标注，兼容编辑器）
var _shm     = null
var _capture = null
# FileAccess 回退状态
var _write_index: int = 0


func _ready() -> void:
	await get_tree().process_frame
	await get_tree().process_frame

	# 创建 SubViewport 640×360，用 GPU TextureRect 采样主 Viewport 做降采样
	_capture_vp = SubViewport.new()
	_capture_vp.size = Vector2i(STREAM_WIDTH, STREAM_HEIGHT)
	_capture_vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(_capture_vp)

	var tr := TextureRect.new()
	tr.set_anchors_preset(Control.PRESET_FULL_RECT)
	tr.expand_mode = TextureRect.EXPAND_FIT_WIDTH_PROPORTIONAL
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	tr.texture = get_viewport().get_texture()
	_capture_vp.add_child(tr)

	# 检测 GDExtension 是否可用
	_use_gdext = ClassDB.class_exists("ShmSync") and ClassDB.class_exists("FrameCapture")

	if _use_gdext:
		_init_gdext()
	else:
		_init_file_fallback()


func _init_gdext() -> void:
	_shm = ClassDB.instantiate("ShmSync")
	add_child(_shm)

	var ok: bool = _shm.open_shm("shm_frame", STREAM_WIDTH * STREAM_HEIGHT * 4, true)
	if not ok:
		push_error("[FrameStreamer] ShmSync Writer 打开失败！")
		return

	_capture = ClassDB.instantiate("FrameCapture")
	add_child(_capture)

	_is_streaming = true
	print("[FrameStreamer] GPU降采样 + mmap/semaphore 推流已启动 (零文件I/O)")


func _init_file_fallback() -> void:
	_is_streaming = true
	print("[FrameStreamer] ⚠ GDExtension 不可用，回退 FileAccess 推流模式")


func _exit_tree() -> void:
	_is_streaming = false
	if _shm:
		if _shm.has_method("close_shm"):
			_shm.close_shm()
		_shm.queue_free()
		_shm = null
	if _capture:
		_capture.queue_free()
		_capture = null
	if _capture_vp:
		_capture_vp.queue_free()
		_capture_vp = null


func _process(delta: float) -> void:
	if not _is_streaming:
		return
	_frame_acc += delta
	if _frame_acc < _interval:
		return
	_frame_acc = 0.0

	if _use_gdext:
		_capture_frame_gdext()
	else:
		_capture_frame_file()


func _capture_frame_gdext() -> void:
	if _capture_vp == null or _shm == null:
		return

	_frame_count += 1

	var vp_tex := _capture_vp.get_texture()
	if vp_tex == null:
		return

	var image := vp_tex.get_image()
	if image == null or image.is_empty():
		return

	_capture.image_to_shm(_shm, image)
	_shm.set_frame_id(_frame_count)
	_shm.signal_new_frame()

	if _frame_count % 100 == 0:
		print("[FrameStreamer] 已捕获 %d 帧 (mmap+semaphore)" % _frame_count)


func _capture_frame_file() -> void:
	if _capture_vp == null:
		return

	_frame_count += 1

	var vp_tex := _capture_vp.get_texture()
	if vp_tex == null:
		return

	var image := vp_tex.get_image()
	if image == null or image.is_empty():
		return

	image.convert(Image.FORMAT_RGBA8)
	var data := image.get_data()
	if data.size() != STREAM_WIDTH * STREAM_HEIGHT * 4:
		return

	# 双缓冲写入
	var slot_idx := _write_index % 2
	var slot_path := FRAME_0 if slot_idx == 0 else FRAME_1
	_write_index += 1

	var f := FileAccess.open(slot_path, FileAccess.WRITE)
	if f == null:
		return
	f.store_buffer(data)
	f.close()

	var ctrl := FileAccess.open(FRAME_CTRL, FileAccess.WRITE)
	if ctrl != null:
		ctrl.store_32(_write_index)
		ctrl.store_32(_frame_count)
		ctrl.close()

	if _frame_count % 100 == 0:
		print("[FrameStreamer] 已捕获 %d 帧 (FileAccess 回退)" % _frame_count)
