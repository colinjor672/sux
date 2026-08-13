extends Control

## 船只BBox + 标签绘制（替代Unity CPU Texture2D画框）
## Godot用_draw()直接GPU绘制

@export var normal_color := Color(0, 1, 0, 0.8)
@export var threat_color := Color(1, 0, 0, 0.8)
@export var box_thickness := 2.0
@export var show_labels := true
@export var font_size := 14

var _ships: Array = []
var _coord_w := 1920
var _coord_h := 1080
var _dirty := false

func _ready():
	set_anchors_preset(Control.PRESET_FULL_RECT)

func update_ships(packet: Dictionary):
	var ships = packet.get("ships", [])
	var fw: int = packet.get("width", 1920)
	var fh: int = packet.get("height", 1080)
	_coord_w = packet.get("coord_w", fw)
	_coord_h = packet.get("coord_h", fh)

	_ships = ships
	_dirty = true
	queue_redraw()

func _draw():
	if _ships.is_empty():
		return

	var my_size = size
	var scale_x = my_size.x / float(_coord_w) if _coord_w > 0 else 1.0
	var scale_y = my_size.y / float(_coord_h) if _coord_h > 0 else 1.0

	for ship in _ships:
		if ship == null:
			continue

		var bbox = ship.get("bbox", [])
		if bbox.size() < 4:
			continue

		var x1 = bbox[0] * scale_x
		var y1 = bbox[1] * scale_y
		var x2 = bbox[2] * scale_x
		var y2 = bbox[3] * scale_y

		var rect = Rect2(x1, y1, x2 - x1, y2 - y1)

		var threat = ship.get("threat_level", 0)
		var color = threat_color if threat > 0 else normal_color

		# 画框
		draw_rect(rect, color, false, box_thickness)

		# 标签
		if show_labels:
			var label_text = ship.get("label", "ship")
			var conf = ship.get("conf", 0.0)
			var display_text = "%s %.0f%%" % [label_text, conf * 100]

			var font = ThemeDB.fallback_font
			var text_pos = Vector2(x1, y1 - 4)
			draw_string(font, text_pos, display_text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, color)

func clear_ships():
	_ships.clear()
	queue_redraw()
