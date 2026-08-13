extends Control
## GPU 多边形填充 —— 替代 CPU Image.set_pixel 扫描线，大幅降低 CPU 占用


@export var water_color := Color(0.0, 0.86, 1.0, 0.14)
@export var bridge_color := Color(0.0, 0.0, 1.0, 0.25)
@export var border_alpha_multiplier := 0.3
@export var subdivide_edge_length := 2.0   # 边缘细分阈值（像素），越小越平滑
@export var antialias_border_width := 1.5  # 抗锯齿轮廓线宽度（像素）
@export var warning_line_color := Color(1.0, 0.3, 0.1, 0.9)  # 警戒线颜色（橙红色）
@export var warning_line_width := 2.5  
var _water_polys: Array = []
var _bridge_polys: Array = []
var _warning_line: Array = []
var _fw := 0
var _fh := 0


func _ready() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func update_masks(packet: Dictionary) -> void:
	_fw = packet.get("width", 1280)
	_fh = packet.get("height", 720)
	_water_polys = packet.get("water_polygons", [])
	_bridge_polys = packet.get("bridge_polygons", [])
	_warning_line = packet.get("bridge_warning_line", [])
	queue_redraw()


func _draw() -> void:
	var rect := get_rect()
	if rect.size.x < 1.0 or rect.size.y < 1.0:
		return

	var sx := rect.size.x / maxf(1.0, float(_fw))
	var sy := rect.size.y / maxf(1.0, float(_fh))

	_draw_poly_list(_water_polys, water_color, sx, sy)
	_draw_poly_list(_bridge_polys, bridge_color, sx, sy)
	_draw_warning_line(sx, sy)


func _draw_poly_list(polygons: Array, color: Color, sx: float, sy: float) -> void:
	for poly in polygons:
		var points_data
		if poly is Dictionary and poly.has("points"):
			points_data = poly["points"]
		elif poly is Array:
			points_data = poly
		else:
			continue

		if points_data == null or points_data.size() < 3:
			continue

		var pts := PackedVector2Array()
		for p in points_data:
			var px: float
			var py: float
			if p is Dictionary:
				px = float(p.get("x", 0)) * sx
				py = float(p.get("y", 0)) * sy
			elif p is Array and p.size() >= 2:
				px = float(p[0]) * sx
				py = float(p[1]) * sy
			else:
				continue
			pts.append(Vector2(px, py))

		if pts.size() >= 3:
			# 边缘细分：过长的边插入中间顶点，使 GPU 光栅化更平滑
			if subdivide_edge_length > 0.0:
				pts = _subdivide_polygon(pts, subdivide_edge_length)
			# 填充多边形
			draw_colored_polygon(pts, color)
			# 叠加抗锯齿轮廓线，柔化边缘
			var border_color := Color(
				color.r,
				color.g,
				color.b,
				color.a * border_alpha_multiplier 
				)
			draw_polyline(pts, border_color, antialias_border_width, true)
			
func _draw_warning_line(sx: float, sy: float) -> void:
	if _warning_line == null or _warning_line.size() < 2:
		return
 
	var pts := PackedVector2Array()
	for p in _warning_line:
		var px: float
		var py: float
		if p is Dictionary:
			px = float(p.get("x", 0)) * sx
			py = float(p.get("y", 0)) * sy
		elif p is Array and p.size() >= 2:
			px = float(p[0]) * sx
			py = float(p[1]) * sy
		else:
			continue
		pts.append(Vector2(px, py))
 
	if pts.size() < 2:
		return
 
	# 用抗锯齿折线绘制警戒线
	draw_polyline(pts, warning_line_color, warning_line_width, true)

# ── 多边形边缘细分 ──
func _subdivide_polygon(pts: PackedVector2Array, max_len: float) -> PackedVector2Array:
	var out := PackedVector2Array()
	var m := pts.size()
	for i in range(m):
		var p0 := pts[i]
		var p1 := pts[(i + 1) % m]
		out.append(p0)
		var edge := p1 - p0
		var edge_len := edge.length()
		if edge_len > max_len:
			var subdivs := ceili(edge_len / max_len) - 1
			for j in range(1, subdivs + 1):
				out.append(p0 + edge * (float(j) / float(subdivs + 1)))
	return out


func clear() -> void:
	_water_polys.clear()
	_bridge_polys.clear()
	_warning_line.clear()
	queue_redraw()
