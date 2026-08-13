# NavigationBandSDF.gd
# SDF 屏幕空间距离场渲染 —— 彻底消除三角形拼接导致的几何重叠与颜色叠加
# 每像素精确计算到中心线的距离，无 UV 插值失真
# 对齐新版 navigation_band.gd：4 层合成（glow+body+center+flow）+ 水平偏移边界线
class_name NavigationBandSDF
extends Control


@export_group("视频原始尺寸")
@export var frame_width := 1280
@export var frame_height := 720


@export_group("坐标映射")
@export var use_aspect_fill := true
@export var curve_starts_near := true

@export_group("AR 导航带主体")
@export var near_width := 180.0
@export var far_width := 60.2
@export var near_color := Color(0.05, 0.85, 1.0, 0.60)
@export var far_color := Color(0.05, 0.55, 1.0, 0.00)

@export var edge_near_color := Color(0.0, 0.95, 1.0, 0.35)
@export var edge_far_color := Color(0.0, 0.45, 1.0, 0.00)

@export_group("外层发光")
@export var draw_glow := true
@export var glow_extra_width := 60.0
@export var glow_near_color := Color(0.0, 0.75, 1.0, 0.45)
@export var glow_far_color := Color(0.0, 0.35, 1.0, 0.00)

@export_group("导航带边界细线")
@export var draw_edge_lines := true
@export var edge_line_width_near := 5.0
@export var edge_line_width_far := 2.5
@export var edge_line_near_color := Color(0.0, 0.85, 1.0, 0.95)
@export var edge_line_far_color := Color(0.0, 0.55, 1.0, 0.25)
@export var edge_line_offset_near := 120.0
@export var edge_line_offset_far := 60.0

@export_group("中心柔光线")
@export var draw_center_light := true
@export var center_light_near_width := 60.2
@export var center_light_far_width := 22.4
@export var center_light_near_color := Color(0.75, 1.0, 1.0, 0.45)
@export var center_light_far_color := Color(0.75, 1.0, 1.0, 0.00)


@export_group("中心流动光效")
@export var draw_flow_light := true
@export var flow_speed := 150.0
@export var flow_segment_length := 120.0
@export var flow_segment_gap := 170.0
@export var flow_near_color := Color(0.9, 1.0, 1.0, 0.85)
@export var flow_far_color := Color(0.6, 0.9, 1.0, 0.02)
@export var reverse_flow_direction := false

@export_group("曲线质量")
@export_range(0, 10)
var smooth_iterations := 10
@export_range(16, 180)
var render_segments := 150

@export_range(1, 8)
var tangent_window := 4

@export_group("更新频率")
@export_range(5, 60)
var flow_fps := 14

@export_range(5, 60)
var max_curve_update_fps := 18

# ── SDF 纹理容量（编译期固定上限，GLES3 要求循环上限为常量）──
const MAX_SEGMENTS := 180

# ── 内嵌 SDF shader（Jetson 安全，无外部文件依赖）──
# 4 层合成：glow + body + center + flow，完全对齐 navigation_ribbon.gdshader
const SDF_SHADER_CODE := \
"shader_type canvas_item;\n" + \
"render_mode unshaded, blend_premul_alpha;\n" + \
"\n" + \
"const int MAX_SEGMENTS = 180;\n" + \
"\n" + \
"uniform sampler2D path_tex;\n" + \
"uniform int    segment_count;\n" + \
"uniform vec2   quad_min;\n" + \
"uniform vec2   quad_max;\n" + \
"\n" + \
"uniform bool  near_at_start      = true;\n" + \
"uniform bool  draw_glow          = true;\n" + \
"uniform bool  draw_center_light  = true;\n" + \
"uniform bool  draw_flow_light    = true;\n" + \
"\n" + \
"uniform float near_width         = 130.0;\n" + \
"uniform float far_width          = 60.2;\n" + \
"uniform float glow_extra_width   = 60.0;\n" + \
"\n" + \
"uniform float center_light_near_width = 60.2;\n" + \
"uniform float center_light_far_width  = 22.4;\n" + \
"\n" + \
"uniform vec4  near_color : source_color = vec4(0.05, 0.85, 1.0, 0.60);\n" + \
"uniform vec4  far_color  : source_color = vec4(0.05, 0.55, 1.0, 0.10);\n" + \
"uniform vec4  edge_near_color : source_color = vec4(0.0, 0.95, 1.0, 0.35);\n" + \
"uniform vec4  edge_far_color  : source_color = vec4(0.0, 0.45, 1.0, 0.00);\n" + \
"uniform vec4  glow_near_color : source_color = vec4(0.0, 0.75, 1.0, 0.45);\n" + \
"uniform vec4  glow_far_color  : source_color = vec4(0.0, 0.35, 1.0, 0.08);\n" + \
"uniform vec4  center_light_near_color : source_color = vec4(0.75, 1.0, 1.0, 0.45);\n" + \
"uniform vec4  center_light_far_color  : source_color = vec4(0.75, 1.0, 1.0, 0.08);\n" + \
"\n" + \
"uniform float flow_speed         = 150.0;\n" + \
"uniform float flow_segment_length= 120.0;\n" + \
"uniform float flow_segment_gap   = 170.0;\n" + \
"uniform bool  reverse_flow_direction = false;\n" + \
"uniform vec4  flow_near_color : source_color = vec4(0.9, 1.0, 1.0, 0.85);\n" + \
"uniform vec4  flow_far_color  : source_color = vec4(0.6, 0.9, 1.0, 0.08);\n" + \
"uniform float path_length = 1.0;\n" + \
"\n" + \
"\n" + \
"float smooth01(float x) {\n" + \
"	x = clamp(x, 0.0, 1.0);\n" + \
"	return x * x * (3.0 - 2.0 * x);\n" + \
"}\n" + \
"\n" + \
"float get_perspective_t(float path_t) {\n" + \
"	path_t = clamp(path_t, 0.0, 1.0);\n" + \
"	return near_at_start ? path_t : 1.0 - path_t;\n" + \
"}\n" + \
"\n" + \
"float ribbon_alpha(float normalized_side) {\n" + \
"	float x = clamp(normalized_side, 0.0, 1.0);\n" + \
"	if (x <= 0.48) {\n" + \
"		float t = smoothstep(0.0, 0.48, x);\n" + \
"		return mix(1.0, 0.86, t);\n" + \
"	}\n" + \
"	if (x <= 0.82) {\n" + \
"		float t = smoothstep(0.48, 0.82, x);\n" + \
"		return mix(0.86, 0.28, t);\n" + \
"	}\n" + \
"	float t = smoothstep(0.82, 1.0, x);\n" + \
"	return mix(0.28, 0.0, t);\n" + \
"}\n" + \
"\n" + \
"// 单层 ribbon：给定 cross_abs（到中心的归一化距离）和 width_ratio，输出 RGBA\n" + \
"vec4 make_ribbon_layer(\n" + \
"	float cross_abs,\n" + \
"	float width_ratio,\n" + \
"	float perspective_t,\n" + \
"	vec4 core_near,\n" + \
"	vec4 core_far,\n" + \
"	vec4 edge_near,\n" + \
"	vec4 edge_far\n" + \
") {\n" + \
"	width_ratio = max(width_ratio, 0.0001);\n" + \
"	float normalized_side = cross_abs / width_ratio;\n" + \
"	if (normalized_side >= 1.0) return vec4(0.0);\n" + \
"	float alpha_profile = ribbon_alpha(normalized_side);\n" + \
"	vec4 core_color = mix(core_near, core_far, perspective_t);\n" + \
"	vec4 edge_color = mix(edge_near, edge_far, perspective_t);\n" + \
"	float edge_mix = smoothstep(0.15, 1.0, normalized_side);\n" + \
"	vec4 color = mix(core_color, edge_color, edge_mix);\n" + \
"	color.a *= alpha_profile;\n" + \
"	return color;\n" + \
"}\n" + \
"\n" + \
"float get_flow_mask(float distance_on_path) {\n" + \
"	float cycle = max(1.0, flow_segment_length + flow_segment_gap);\n" + \
"	float offset = mod(TIME * flow_speed, cycle);\n" + \
"	if (reverse_flow_direction) offset = cycle - offset;\n" + \
"	float local_distance = mod(distance_on_path - offset + cycle, cycle);\n" + \
"	if (local_distance >= flow_segment_length) return 0.0;\n" + \
"	float segment_t = clamp(local_distance / max(flow_segment_length, 0.001), 0.0, 1.0);\n" + \
"	return sin(segment_t * 3.14159265359);\n" + \
"}\n" + \
"\n" + \
"// 点到线段距离，同时返回沿段投影 t\n" + \
"float sd_segment_t(vec2 p, vec2 a, vec2 b, out float t) {\n" + \
"	vec2 pa = p - a;\n" + \
"	vec2 ba = b - a;\n" + \
"	float denom = max(dot(ba, ba), 0.0001);\n" + \
"	float h = clamp(dot(pa, ba) / denom, 0.0, 1.0);\n" + \
"	t = h;\n" + \
"	return length(pa - ba * h);\n" + \
"}\n" + \
"\n" + \
"void fragment() {\n" + \
"	// 当前像素在包围盒局部坐标空间的位置\n" + \
"	vec2 screen_pos = mix(quad_min, quad_max, UV);\n" + \
"\n" + \
"	// 遍历所有线段，找最近距离 + 路径累计长度\n" + \
"	float min_dist = 1e10;\n" + \
"	float best_path_dist = 0.0;\n" + \
"\n" + \
"	for (int i = 0; i < MAX_SEGMENTS; i++) {\n" + \
"		if (i >= segment_count) break;\n" + \
"\n" + \
"		// 采样 p0, p1（纹理 R=x, G=y, B=累计长度）\n" + \
"		float u0 = (float(i) + 0.5) / float(MAX_SEGMENTS);\n" + \
"		float u1 = (float(i + 1) + 0.5) / float(MAX_SEGMENTS);\n" + \
"		vec4 d0 = texture(path_tex, vec2(u0, 0.5));\n" + \
"		vec4 d1 = texture(path_tex, vec2(u1, 0.5));\n" + \
"		vec2 p0 = d0.xy;\n" + \
"		vec2 p1 = d1.xy;\n" + \
"		float len0 = d0.b;\n" + \
"		float len1 = d1.b;\n" + \
"\n" + \
"		float t;\n" + \
"		float d = sd_segment_t(screen_pos, p0, p1, t);\n" + \
"\n" + \
"		if (d < min_dist) {\n" + \
"			min_dist = d;\n" + \
"			best_path_dist = mix(len0, len1, t);\n" + \
"		}\n" + \
"	}\n" + \
"\n" + \
"	// 路径参数与宽度\n" + \
"	float path_t = clamp(best_path_dist / max(path_length, 0.001), 0.0, 1.0);\n" + \
"	float perspective_t = get_perspective_t(path_t);\n" + \
"	float width_t = smooth01(perspective_t);\n" + \
"\n" + \
"	float geometry_width = mix(near_width + glow_extra_width, far_width + glow_extra_width, width_t);\n" + \
"	float body_width    = mix(near_width, far_width, width_t);\n" + \
"	float center_width  = mix(center_light_near_width, center_light_far_width, width_t);\n" + \
"\n" + \
"	float body_ratio   = clamp(body_width / max(geometry_width, 0.001), 0.0, 1.0);\n" + \
"	float center_ratio = clamp(center_width / max(geometry_width, 0.001), 0.0, 1.0);\n" + \
"\n" + \
"	// SDF 归一化侧边位置：cross_abs = min_dist / geometry_half_width\n" + \
"	// 等价于原 shader 的 cross_abs = abs(UV.x * 2.0 - 1.0)\n" + \
"	float geometry_half_width = geometry_width * 0.5;\n" + \
"	float cross_abs = min_dist / max(geometry_half_width, 0.001);\n" + \
"\n" + \
"	vec4 glow = vec4(0.0);\n" + \
"	vec4 body = vec4(0.0);\n" + \
"	vec4 center = vec4(0.0);\n" + \
"	vec4 flow = vec4(0.0);\n" + \
"\n" + \
"	// glow 层（width_ratio = 1.0）\n" + \
"	if (draw_glow) {\n" + \
"		vec4 glow_edge_near = glow_near_color;\n" + \
"		glow_edge_near.a *= 0.55;\n" + \
"		glow = make_ribbon_layer(cross_abs, 1.0, perspective_t,\n" + \
"			glow_near_color, glow_far_color, glow_edge_near, glow_far_color);\n" + \
"	}\n" + \
"\n" + \
"	// body 层\n" + \
"	body = make_ribbon_layer(cross_abs, body_ratio, perspective_t,\n" + \
"		near_color, far_color, edge_near_color, edge_far_color);\n" + \
"\n" + \
"	// center 层\n" + \
"	if (draw_center_light) {\n" + \
"		center = make_ribbon_layer(cross_abs, center_ratio, perspective_t,\n" + \
"			center_light_near_color, center_light_far_color,\n" + \
"			center_light_near_color, center_light_far_color);\n" + \
"	}\n" + \
"\n" + \
"	// flow 层\n" + \
"	if (draw_flow_light) {\n" + \
"		float flow_ratio = clamp(center_ratio * 1.35, 0.0, 1.0);\n" + \
"		float flow_mask = get_flow_mask(best_path_dist);\n" + \
"		flow = make_ribbon_layer(cross_abs, flow_ratio, perspective_t,\n" + \
"			flow_near_color, flow_far_color, flow_near_color, flow_far_color);\n" + \
"		flow.a *= flow_mask;\n" + \
"	}\n" + \
"\n" + \
"	// 4 层合成（与原 shader 完全一致）\n" + \
"	vec3 rgb = vec3(0.0);\n" + \
"	rgb += glow.rgb * glow.a;\n" + \
"	rgb += body.rgb * body.a;\n" + \
"	rgb += center.rgb * center.a;\n" + \
"	rgb += flow.rgb * flow.a * 1.25;\n" + \
"\n" + \
"	float alpha = clamp(glow.a * 0.55 + body.a + center.a * 0.85 + flow.a, 0.0, 1.0);\n" + \
"\n" + \
"	if (alpha <= 0.0001) discard;\n" + \
"\n" + \
"	COLOR = vec4(rgb, alpha);\n" + \
"}\n"

# ── 内部状态 ──
var _curve_points: Array[Vector2] = []
var _render_points: Array[Vector2] = []
var _render_ts: Array[float] = []
var _smooth_buffer: Array[Vector2] = []
var _strip_normals: Array[Vector2] = []
var _mesh_half_widths: Array[float] = []

var _last_curve_update_time := -1.0

var _pending_curves: Array = []
var _pending_src_w := 0
var _pending_src_h := 0
var _has_pending_curve := false

var _curve_total_length := 0.0
var _render_total_length := 0.0

# SDF 渲染节点
var _sdf_mesh_node: MeshInstance2D = null
var _sdf_material: ShaderMaterial = null
var _path_image: Image = null
var _path_texture: ImageTexture = null

# 左右边界细线
var _line_left: Line2D = null
var _line_right: Line2D = null


func _ready() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	_init_sdf_mesh()
	_init_edge_lines()
	print("[NavBandSDF] 就绪: SDF 屏幕空间距离场渲染, MAX_SEGMENTS=%d" % MAX_SEGMENTS)


# ════════════════════════════════════════════
# SDF Mesh 初始化
# ════════════════════════════════════════════

func _init_sdf_mesh() -> void:
	var shader := Shader.new()
	shader.code = SDF_SHADER_CODE

	var material := ShaderMaterial.new()
	material.shader = shader
	_sdf_material = material

	var mesh_node := MeshInstance2D.new()
	mesh_node.name = "NavigationSDFMesh"
	mesh_node.z_index = 10
	mesh_node.visible = true
	mesh_node.material = material
	_sdf_mesh_node = mesh_node
	add_child(_sdf_mesh_node)

	# 预创建路径纹理（MAX_SEGMENTS+1 个点，1 像素高）
	_path_image = Image.create(MAX_SEGMENTS + 1, 1, false, Image.FORMAT_RGBAF)
	_path_texture = ImageTexture.create_from_image(_path_image)
	_sdf_material.set_shader_parameter("path_tex", _path_texture)

	_sync_shader_params()


# ════════════════════════════════════════════
# 左右边界细线初始化（完全对齐新版 navigation_band.gd）
# ════════════════════════════════════════════

func _init_edge_lines() -> void:
	_line_left = _create_edge_line("NavigationEdgeLeft")
	_line_right = _create_edge_line("NavigationEdgeRight")


func _create_edge_line(node_name: String) -> Line2D:
	var line := Line2D.new()
	line.name = node_name
	line.z_index = 12
	line.visible = draw_edge_lines
	line.width = edge_line_width_near
	line.default_color = edge_line_near_color
	line.joint_mode = Line2D.LINE_JOINT_ROUND
	line.round_precision = 12
	line.begin_cap_mode = Line2D.LINE_CAP_ROUND
	line.end_cap_mode = Line2D.LINE_CAP_ROUND
	line.antialiased = true
	add_child(line)
	return line


func _clear_edge_lines() -> void:
	if _line_left != null:
		_line_left.clear_points()
	if _line_right != null:
		_line_right.clear_points()


func _update_edge_lines() -> void:
	if _line_left == null or _line_right == null:
		return

	_line_left.visible = draw_edge_lines
	_line_right.visible = draw_edge_lines

	if not draw_edge_lines:
		_clear_edge_lines()
		return

	var point_count := _render_points.size()
	if point_count < 2:
		_clear_edge_lines()
		return

	# ── 颜色渐变 ──
	var gradient := Gradient.new()
	var offsets := PackedFloat32Array()
	var colors := PackedColorArray()
	for i in range(point_count):
		var path_t := float(i) / maxf(1.0, float(point_count - 1))
		var perspective_t := _get_perspective_t(path_t)
		var width_t := _smooth01(perspective_t)
		offsets.append(path_t)
		colors.append(edge_line_near_color.lerp(edge_line_far_color, width_t))
	gradient.offsets = offsets
	gradient.colors = colors

	# ── 细线自身宽度渐变 ──
	var maximum_line_width := maxf(edge_line_width_near, edge_line_width_far)
	var width_curve := Curve.new()
	width_curve.min_value = 0.0
	width_curve.max_value = 1.0
	width_curve.bake_resolution = 128
	for i in range(point_count):
		var path_t := float(i) / maxf(1.0, float(point_count - 1))
		var perspective_t := _get_perspective_t(path_t)
		var width_t := _smooth01(perspective_t)
		var current_line_width := lerpf(edge_line_width_near, edge_line_width_far, width_t)
		var normalized_width := current_line_width / maxf(maximum_line_width, 0.001)
		width_curve.add_point(
			Vector2(path_t, clampf(normalized_width, 0.001, 1.0)),
			0.0, 0.0,
			Curve.TANGENT_LINEAR, Curve.TANGENT_LINEAR
		)
	width_curve.bake()

	_line_left.gradient = gradient
	_line_right.gradient = gradient.duplicate() as Gradient
	_line_left.width = maximum_line_width
	_line_right.width = maximum_line_width
	_line_left.width_curve = width_curve
	_line_right.width_curve = width_curve.duplicate() as Curve

	_line_left.clear_points()
	_line_right.clear_points()

	# ── 水平偏移构造（与新版 navigation_band.gd 完全一致）──
	# 右线 = 中心路径水平偏移
	# 左线 = 右线关于中心路径镜像
	for i in range(point_count):
		var path_t := float(i) / maxf(1.0, float(point_count - 1))
		var perspective_t := _get_perspective_t(path_t)
		var width_t := _smooth01(perspective_t)
		var line_offset := lerpf(edge_line_offset_near, edge_line_offset_far, width_t)
		var center: Vector2 = _render_points[i]
		var right_point: Vector2 = center + Vector2(line_offset, 0.0)
		var left_point: Vector2 = center * 2.0 - right_point
		_line_right.add_point(right_point)
		_line_left.add_point(left_point)


# ════════════════════════════════════════════
# 主循环
# ════════════════════════════════════════════

func _process(_delta: float) -> void:
	var now := Time.get_ticks_msec() / 1000.0
	if not _has_pending_curve:
		return
	var update_interval := 1.0 / maxf(1.0, float(max_curve_update_fps))
	if _last_curve_update_time < 0.0 or now - _last_curve_update_time >= update_interval:
		_last_curve_update_time = now
		_has_pending_curve = false
		_apply_curve(_pending_curves, _pending_src_w, _pending_src_h)


# ════════════════════════════════════════════
# 公开接口（与 navigation_band.gd 一致）
# ════════════════════════════════════════════

func set_curve_from_packet(packet: Dictionary) -> void:
	var curves = packet.get("curves", packet.get("curve", []))
	var src_w: int = packet.get("coord_w", packet.get("width", 1280))
	var src_h: int = packet.get("coord_h", packet.get("height", 720))
	set_curve(curves, src_w, src_h)


func set_curve(curves: Array, src_w: int, src_h: int) -> void:
	_pending_curves = curves
	_pending_src_w = src_w
	_pending_src_h = src_h
	_has_pending_curve = true


func clear() -> void:
	_has_pending_curve = false
	_pending_curves.clear()
	_curve_points.clear()
	_render_points.clear()
	_render_ts.clear()
	_strip_normals.clear()
	_mesh_half_widths.clear()
	_curve_total_length = 0.0
	_render_total_length = 0.0
	_clear_sdf_mesh()
	_clear_edge_lines()


# ════════════════════════════════════════════
# 路径处理（对齐新版 navigation_band.gd）
# ════════════════════════════════════════════

func _apply_curve(curves: Array, src_w: int, src_h: int) -> void:
	frame_width = src_w
	frame_height = src_h
	_curve_points.clear()
	_render_points.clear()
	_render_ts.clear()

	if curves.size() < 2:
		_clear_all_geometry()
		return

	for item in curves:
		var video_point := Vector2.ZERO
		var valid_point := false
		if item is Array and item.size() >= 2:
			video_point = Vector2(float(item[0]), float(item[1]))
			valid_point = true
		elif item is Dictionary:
			video_point = Vector2(float(item.get("x", 0)), float(item.get("y", 0)))
			valid_point = true
		if not valid_point:
			continue

		var local_point := _video_pixel_to_local(video_point)

		# 清理连续重复点
		if not _curve_points.is_empty():
			var previous: Vector2 = _curve_points[_curve_points.size() - 1]
			if local_point.distance_squared_to(previous) < 0.0001:
				continue
		_curve_points.append(local_point)

	if _curve_points.size() < 2:
		_clear_all_geometry()
		return

	_smooth_curve(_curve_points, smooth_iterations)
	_curve_total_length = _get_polyline_length(_curve_points)

	if _curve_total_length <= 0.001:
		_clear_all_geometry()
		return

	_build_sampled_path(_curve_points, _render_points, _render_ts, render_segments, _curve_total_length)
	_render_total_length = _get_polyline_length(_render_points)

	# 法线计算（边线用，SDF 不需要法线但边线需要）
	_compute_stable_strip_geometry()

	# SDF 核心：路径纹理 + 包围盒 mesh + shader 参数
	_build_path_texture()
	_build_quad_mesh()
	_sync_shader_params()

	_update_edge_lines()


func _clear_all_geometry() -> void:
	_curve_total_length = 0.0
	_render_total_length = 0.0
	_strip_normals.clear()
	_mesh_half_widths.clear()
	_clear_sdf_mesh()
	_clear_edge_lines()


# ════════════════════════════════════════════
# 稳定条带几何（对齐新版 navigation_band.gd，边线专用）
# ════════════════════════════════════════════

func _compute_stable_strip_geometry() -> void:
	var point_count := _render_points.size()
	_strip_normals.resize(point_count)
	_mesh_half_widths.resize(point_count)
	if point_count < 2:
		return

	var safe_window := clampi(tangent_window, 1, maxi(1, point_count - 1))
	var previous_normal := Vector2.ZERO
	var previous_tangent := Vector2.ZERO

	for i in range(point_count):
		var previous_index := maxi(0, i - safe_window)
		var next_index := mini(point_count - 1, i + safe_window)
		var tangent: Vector2 = _render_points[next_index] - _render_points[previous_index]

		if tangent.length_squared() < 0.000001:
			if i < point_count - 1:
				tangent = _render_points[i + 1] - _render_points[i]
			elif i > 0:
				tangent = _render_points[i] - _render_points[i - 1]

		if tangent.length_squared() < 0.000001:
			if previous_tangent.length_squared() > 0.0:
				tangent = previous_tangent
			else:
				tangent = Vector2.UP
		else:
			tangent = tangent.normalized()

		if i > 0 and previous_tangent.length_squared() > 0.0 and tangent.dot(previous_tangent) < 0.0:
			tangent = -tangent

		var normal: Vector2 = Vector2(-tangent.y, tangent.x).normalized()

		if i > 0 and previous_normal.length_squared() > 0.0 and normal.dot(previous_normal) < 0.0:
			normal = -normal

		_strip_normals[i] = normal
		previous_tangent = tangent
		previous_normal = normal

		var path_t := float(i) / maxf(1.0, float(point_count - 1))
		var perspective_t := _get_perspective_t(path_t)
		var width_t := _smooth01(perspective_t)
		_mesh_half_widths[i] = lerpf(near_width + glow_extra_width, far_width + glow_extra_width, width_t) * 0.5


# ════════════════════════════════════════════
# SDF 核心：路径纹理打包
# ════════════════════════════════════════════

func _build_path_texture() -> void:
	var n := _render_points.size()
	if n < 2:
		return

	# 填充 Image：R=x, G=y, B=累计路径长度
	var acc_len := 0.0
	for i in range(n):
		var p: Vector2 = _render_points[i]
		_path_image.set_pixel(i, 0, Color(p.x, p.y, acc_len, 0.0))
		if i < n - 1:
			var next_p: Vector2 = _render_points[i + 1]
			acc_len += p.distance_to(next_p)

	# 剩余纹素填最后一个点（避免脏数据）
	var last_p: Vector2 = _render_points[n - 1]
	for i in range(n, MAX_SEGMENTS + 1):
		_path_image.set_pixel(i, 0, Color(last_p.x, last_p.y, acc_len, 0.0))

	_path_texture.update(_path_image)


# ════════════════════════════════════════════
# SDF 核心：包围盒 Quad Mesh
# ════════════════════════════════════════════

func _build_quad_mesh() -> void:
	if _render_points.size() < 2:
		_clear_sdf_mesh()
		return

	# 计算中心线 AABB，加 padding 覆盖整个导航带宽度（含 glow 层）
	var aabb_min := _render_points[0]
	var aabb_max := _render_points[0]
	for i in range(1, _render_points.size()):
		var p: Vector2 = _render_points[i]
		aabb_min.x = minf(aabb_min.x, p.x)
		aabb_min.y = minf(aabb_min.y, p.y)
		aabb_max.x = maxf(aabb_max.x, p.x)
		aabb_max.y = maxf(aabb_max.y, p.y)

	var padding := (near_width + glow_extra_width) * 0.5 + 20.0
	aabb_min -= Vector2(padding, padding)
	aabb_max += Vector2(padding, padding)

	# 构建 quad：4 顶点，UV (0,0)~(1,1) 映射到 aabb_min~aabb_max
	var vertices := PackedVector2Array()
	var uvs := PackedVector2Array()
	vertices.append(aabb_min);                              uvs.append(Vector2(0.0, 0.0))
	vertices.append(Vector2(aabb_max.x, aabb_min.y));       uvs.append(Vector2(1.0, 0.0))
	vertices.append(Vector2(aabb_min.x, aabb_max.y));       uvs.append(Vector2(0.0, 1.0))
	vertices.append(aabb_max);                              uvs.append(Vector2(1.0, 1.0))

	var indices := PackedInt32Array([0, 1, 2, 2, 1, 3])

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = indices

	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	_sdf_mesh_node.mesh = mesh

	# 传包围盒给 shader
	_sdf_material.set_shader_parameter("quad_min", aabb_min)
	_sdf_material.set_shader_parameter("quad_max", aabb_max)


func _clear_sdf_mesh() -> void:
	if _sdf_mesh_node:
		_sdf_mesh_node.mesh = null


# ════════════════════════════════════════════
# Shader 参数同步（完全对齐新版 navigation_band.gd）
# ════════════════════════════════════════════

func _sync_shader_params() -> void:
	if _sdf_material == null or _sdf_material.shader == null:
		return

	_sdf_material.set_shader_parameter("near_at_start",         curve_starts_near)
	_sdf_material.set_shader_parameter("draw_glow",              draw_glow)
	_sdf_material.set_shader_parameter("draw_center_light",      draw_center_light)
	_sdf_material.set_shader_parameter("draw_flow_light",        draw_flow_light)
	_sdf_material.set_shader_parameter("near_width",             near_width)
	_sdf_material.set_shader_parameter("far_width",              far_width)
	_sdf_material.set_shader_parameter("glow_extra_width",       glow_extra_width)
	_sdf_material.set_shader_parameter("center_light_near_width", center_light_near_width)
	_sdf_material.set_shader_parameter("center_light_far_width",  center_light_far_width)
	_sdf_material.set_shader_parameter("near_color",             near_color)
	_sdf_material.set_shader_parameter("far_color",              far_color)
	_sdf_material.set_shader_parameter("edge_near_color",        edge_near_color)
	_sdf_material.set_shader_parameter("edge_far_color",         edge_far_color)
	_sdf_material.set_shader_parameter("glow_near_color",        glow_near_color)
	_sdf_material.set_shader_parameter("glow_far_color",         glow_far_color)
	_sdf_material.set_shader_parameter("center_light_near_color", center_light_near_color)
	_sdf_material.set_shader_parameter("center_light_far_color",  center_light_far_color)
	_sdf_material.set_shader_parameter("flow_speed",             flow_speed)
	_sdf_material.set_shader_parameter("flow_segment_length",    flow_segment_length)
	_sdf_material.set_shader_parameter("flow_segment_gap",       flow_segment_gap)
	_sdf_material.set_shader_parameter("reverse_flow_direction", reverse_flow_direction)
	_sdf_material.set_shader_parameter("flow_near_color",        flow_near_color)
	_sdf_material.set_shader_parameter("flow_far_color",         flow_far_color)
	_sdf_material.set_shader_parameter("path_length",            maxf(_render_total_length, 1.0))
	_sdf_material.set_shader_parameter("segment_count",          maxi(_render_points.size() - 1, 0))


# ════════════════════════════════════════════
# 坐标映射 / 路径采样 / 几何工具（完全对齐新版）
# ════════════════════════════════════════════

func _video_pixel_to_local(pixel: Vector2) -> Vector2:
	var rect := get_rect()
	var video_aspect := float(frame_width) / maxf(1.0, float(frame_height))
	var rect_aspect := rect.size.x / maxf(1.0, rect.size.y)
	var draw_width := 0.0
	var draw_height := 0.0
	var offset_x := 0.0
	var offset_y := 0.0
	if use_aspect_fill:
		if rect_aspect > video_aspect:
			draw_width = rect.size.x
			draw_height = draw_width / video_aspect
			offset_y = (rect.size.y - draw_height) * 0.5
		else:
			draw_height = rect.size.y
			draw_width = draw_height * video_aspect
			offset_x = (rect.size.x - draw_width) * 0.5
	else:
		if rect_aspect > video_aspect:
			draw_height = rect.size.y
			draw_width = draw_height * video_aspect
			offset_x = (rect.size.x - draw_width) * 0.5
		else:
			draw_width = rect.size.x
			draw_height = draw_width / video_aspect
			offset_y = (rect.size.y - draw_height) * 0.5
	var normalized_x := pixel.x / maxf(1.0, float(frame_width))
	var normalized_y := pixel.y / maxf(1.0, float(frame_height))
	return Vector2(offset_x + normalized_x * draw_width, offset_y + normalized_y * draw_height)


func _build_sampled_path(
	source: Array[Vector2], out_points: Array[Vector2],
	out_ts: Array[float],   segments: int, total_length: float
) -> void:
	out_points.clear()
	out_ts.clear()
	if total_length <= 1.0:
		out_points.append_array(source)
		for i in range(source.size()):
			var normalized := float(i) / maxf(1.0, float(source.size() - 1))
			out_ts.append(_get_perspective_t(normalized))
		return
	var count := maxi(2, segments + 1)
	for i in range(count):
		var normalized := float(i) / float(count - 1)
		var target_distance := normalized * total_length
		var result := _get_point_and_tangent(source, target_distance)
		out_points.append(result[0])
		out_ts.append(_get_perspective_t(normalized))


func _get_polyline_length(points: Array[Vector2]) -> float:
	var total := 0.0
	for i in range(points.size() - 1):
		total += points[i].distance_to(points[i + 1])
	return total


func _get_point_and_tangent(
	points: Array[Vector2], target_distance: float
) -> Array:
	var travelled := 0.0
	for i in range(points.size() - 1):
		var segment_length := points[i].distance_to(points[i + 1])
		if travelled + segment_length >= target_distance:
			var local_t := (target_distance - travelled) / maxf(segment_length, 0.0001)
			var point := points[i].lerp(points[i + 1], local_t)
			var tangent := points[i + 1] - points[i]
			if tangent.length_squared() > 0.0001:
				tangent = tangent.normalized()
			else:
				tangent = Vector2.UP
			return [point, tangent]
		travelled += segment_length
	return [points[points.size() - 1], Vector2.UP]


func _get_perspective_t(normalized_distance: float) -> float:
	normalized_distance = clampf(normalized_distance, 0.0, 1.0)
	return normalized_distance if curve_starts_near else 1.0 - normalized_distance


func _smooth01(t: float) -> float:
	t = clampf(t, 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)


func _smooth_curve(points: Array[Vector2], iterations: int) -> void:
	if points.size() < 4 or iterations <= 0:
		return
	_smooth_buffer.resize(points.size())
	for _iteration in range(iterations):
		for i in range(points.size()):
			_smooth_buffer[i] = points[i]
		for i in range(1, points.size() - 1):
			points[i] = (_smooth_buffer[i - 1] + _smooth_buffer[i] * 2.0 + _smooth_buffer[i + 1]) * 0.25
