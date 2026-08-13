extends TextureRect

## 视频帧显示（接收已解码的 Image）

var _tex: ImageTexture = null
var _last_w := 0
var _last_h := 0
var _frame_count := 0

func _ready():
	expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	stretch_mode = TextureRect.STRETCH_SCALE
	set_anchors_preset(Control.PRESET_FULL_RECT)

func update_frame(w: int, h: int, img: Image):
	if img == null or img.is_empty():
		return

	_frame_count += 1

	if _frame_count == 1:
		# 首帧：检查像素数据是否全黑
		var data := img.get_data()
		var all_zero := true
		for i in range(mini(data.size(), 4096)):
			if data[i] != 0:
				all_zero = false
				break
		print("[VideoDisplay] 首帧 %dx%d all_zero=%s data_size=%d" % [w, h, all_zero, data.size()])

	if _last_w != w or _last_h != h or _tex == null:
		_tex = ImageTexture.create_from_image(img)
		texture = _tex
		_last_w = w
		_last_h = h
		if _frame_count <= 1:
			print("[VideoDisplay] ImageTexture 已创建 %dx%d" % [w, h])
	else:
		_tex.update(img)

	if _frame_count % 100 == 0:
		print("[VideoDisplay] 已显示 %d 帧" % _frame_count)
