extends Control

## 船只BBox + 标签绘制（替代Unity CPU Texture2D画框）
## Godot用_draw()直接GPU绘制

@export var normal_color := Color(0, 1, 0, 0.8)
@export var threat_color := Color(1, 0, 0, 0.8)
@export var box_thickness := 2.0
@export var show_labels := true
@export var font_size := 14
@export var fusion_panel_color := Color(0.0, 0.45, 0.12, 0.72)
@export var fusion_panel_border_color := Color(0.2, 1.0, 0.35, 0.95)
@export var fusion_text_color := Color(0.92, 1.0, 0.94, 1.0)
@export var fusion_panel_padding := 6.0
@export var fusion_panel_gap := 6.0
@export var fusion_panel_min_width := 160.0

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

		# 标签 / 融合信息
		if show_labels:
			var font = ThemeDB.fallback_font
			_draw_fusion_panel(font, ship, rect)

func _draw_fusion_panel(font: Font, ship: Dictionary, ship_rect: Rect2):
	var lines := PackedStringArray([
		"north_vel: ",
		"east_vel: ",
		"distance: ",
		"yaw: ",
	])
	if bool(ship.get("has_fusion_data", false)):
		lines = PackedStringArray([
			"north_vel: %.2f m/s" % float(ship.get("north_vel", 0.0)),
			"east_vel: %.2f m/s" % float(ship.get("east_vel", 0.0)),
			"distance: %.2f m" % float(ship.get("distance", 0.0)),
			"yaw: %.2f" % float(ship.get("yaw", 0.0)),
		])

	var line_height := font.get_height(font_size)
	var text_width := 0.0
	for line in lines:
		text_width = maxf(
			text_width,
			font.get_string_size(line, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).x
		)

	var panel_size := Vector2(
		maxf(fusion_panel_min_width, text_width + fusion_panel_padding * 2.0),
		line_height * lines.size() + fusion_panel_padding * 2.0
	)
	var max_x := maxf(0.0, size.x - panel_size.x)
	var panel_x := clampf(ship_rect.position.x, 0.0, max_x)
	var panel_y := maxf(
		0.0,
		ship_rect.position.y - fusion_panel_gap - panel_size.y
	)
	var panel_rect := Rect2(Vector2(panel_x, panel_y), panel_size)

	draw_rect(panel_rect, fusion_panel_color, true)
	draw_rect(panel_rect, fusion_panel_border_color, false, 1.0)

	var baseline_y := panel_y + fusion_panel_padding + font.get_ascent(font_size)
	for index in range(lines.size()):
		var text_pos := Vector2(
			panel_x + fusion_panel_padding,
			baseline_y + line_height * index
		)
		draw_string(
			font,
			text_pos,
			lines[index],
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			font_size,
			fusion_text_color
		)

func clear_ships():
	_ships.clear()
	queue_redraw()
