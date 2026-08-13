class_name NavigationBandCpuV2
extends Control

## MeshInstance2D 版导航带 —— 无自定义 Shader，用 CPU 预计算顶点颜色匹配原 shader 效果
## 跨截面多顶点 + miter join 裁剪，消除曲线拐弯处的重叠散射

@export_group("视频原始尺寸")
@export var frame_width := 1280
@export var frame_height := 720

@export_group("坐标映射")
@export var use_aspect_fill := true
@export var curve_starts_near := true

@export_group("AR 导航带主体")
@export var near_width := 341.0
@export var far_width := 112.2
@export var near_color := Color(0.05, 0.85, 1.0, 0.70)
@export var far_color := Color(0.05, 0.55, 1.0, 0.12)
@export var edge_near_color := Color(0.0, 0.95, 1.0, 0.50)
@export var edge_far_color := Color(0.0, 0.45, 1.0, 0.00)

@export_group("外层发光")
@export var draw_glow := true
@export var glow_extra_width := 137.5
@export var glow_near_color := Color(0.0, 0.75, 1.0, 0.45)
@export var glow_far_color := Color(0.0, 0.35, 1.0, 0.15)

@export_group("中心柔光线")
@export var draw_center_light := true
@export var center_light_near_width := 68.2
@export var center_light_far_width := 26.4
@export var center_light_near_color := Color(0.75, 1.0, 1.0, 0.25)
@export var center_light_far_color := Color(0.75, 1.0, 1.0, 0.10)

@export_group("中心流动光效")
@export var draw_flow_light := true
@export var flow_speed := 150.0
@export var flow_segment_length := 120.0
@export var flow_segment_gap := 170.0
@export var flow_near_color := Color(0.9, 1.0, 1.0, 0.55)
@export var flow_far_color := Color(0.6, 0.9, 1.0, 0.00)
@export var reverse_flow_direction := false

@export_group("曲线质量")
@export_range(0, 10) var smooth_iterations := 10
@export_range(16, 180) var render_segments := 120

@export_group("渲染质量")
@export_range(2.0, 10.0) var miter_limit := 10.0
@export_range(0.0, 1.0) var cross_blur := 1.0

@export_group("更新频率")
@export_range(5, 60) var flow_fps := 14
@export_range(5, 60) var max_curve_update_fps := 18

# ── 跨截面顶点：每侧 9 顶点 + 中心 = 19，近中心密集 ──
const VERTS_PER_SIDE := 9
const VERTS_PER_SEGMENT := 19     # 9左 + 1中心 + 9右

# 非均匀分布：中心附近密集，边缘稀疏（对应 center_ratio ≈ 0.14 也能覆盖）
# 左半: [1.0, 0.82, 0.66, 0.50, 0.36, 0.24, 0.14, 0.06, 0.015]
# 右半: 镜像
const CROSS_SIDES := [
	1.0, 0.82, 0.66, 0.50, 0.36, 0.24, 0.14, 0.06, 0.015,
	0.0,
	0.015, 0.06, 0.14, 0.24, 0.36, 0.50, 0.66, 0.82, 1.0
]

# 内嵌最小 pass-through shader（无 preload，避免 Jetson 路径问题）
const PASSTHROUGH_SHADER_CODE := \
"shader_type canvas_item;\n" + \
"render_mode blend_mix;\n" + \
"\n" + \
"void fragment() {\n" + \
"\tCOLOR = COLOR;\n" + \
"}"

# ── 内部状态 ──
var _curve_points: Array[Vector2] = []
var _render_points: Array[Vector2] = []
var _render_ts: Array[float] = []
var _smooth_buffer: Array[Vector2] = []
var _miter_normals: Array[Vector2] = []   # 每个点的左向 miter 法线

var _last_curve_update_time := -1.0
var _last_flow_update_time := -1.0
var _pending_curves: Array = []
var _pending_src_w := 0
var _pending_src_h := 0
var _has_pending_curve := false

var _curve_total_length := 0.0
var _render_total_length := 0.0
var _has_ribbon := false

var _flow_offset := 0.0

# Mesh 节点
var _static_mesh_node: MeshInstance2D = null
var _flow_mesh_node: MeshInstance2D = null


# ═══════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════

func _ready() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	_init_mesh_nodes()
	print("[NavBandCPU] 就绪: MeshInstance2D + %d顶点/截面 + miter join + 跨截面模糊" % VERTS_PER_SEGMENT)


func _init_mesh_nodes() -> void:
	# 内嵌 shader 创建（无 preload，Jetson 安全）
	var shader := Shader.new()
	shader.code = PASSTHROUGH_SHADER_CODE

	var static_mat := ShaderMaterial.new()
	static_mat.shader = shader
	_static_mesh_node = MeshInstance2D.new()
	_static_mesh_node.name = "StaticRibbonMesh"
	_static_mesh_node.z_index = 10
	_static_mesh_node.material = static_mat
	add_child(_static_mesh_node)

	var flow_mat := ShaderMaterial.new()
	flow_mat.shader = shader
	_flow_mesh_node = MeshInstance2D.new()
	_flow_mesh_node.name = "FlowRibbonMesh"
	_flow_mesh_node.z_index = 11
	_flow_mesh_node.material = flow_mat
	add_child(_flow_mesh_node)


# ═══════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════

func _process(delta: float) -> void:
	var now := Time.get_ticks_msec() / 1000.0

	# 限帧更新曲线
	if _has_pending_curve:
		var curve_interval := 1.0 / maxf(1.0, float(max_curve_update_fps))
		if _last_curve_update_time < 0.0 or now - _last_curve_update_time >= curve_interval:
			_last_curve_update_time = now
			_has_pending_curve = false
			_apply_curve(_pending_curves, _pending_src_w, _pending_src_h)

	# 限帧更新流光
	if _has_ribbon and draw_flow_light:
		var flow_interval := 1.0 / maxf(1.0, float(flow_fps))
		if _last_flow_update_time < 0.0 or now - _last_flow_update_time >= flow_interval:
			_last_flow_update_time = now
			var cycle := maxf(1.0, flow_segment_length + flow_segment_gap)
			if reverse_flow_direction:
				_flow_offset -= flow_speed * flow_interval
			else:
				_flow_offset += flow_speed * flow_interval
			_flow_offset = fmod(_flow_offset, cycle * 100.0)
			_build_flow_mesh()


# ═══════════════════════════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════════════════════════

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
	_miter_normals.clear()
	_curve_total_length = 0.0
	_render_total_length = 0.0
	_has_ribbon = false
	if _static_mesh_node:
		_static_mesh_node.mesh = null
	if _flow_mesh_node:
		_flow_mesh_node.mesh = null


# ═══════════════════════════════════════════════════════════════
# 曲线处理
# ═══════════════════════════════════════════════════════════════

func _apply_curve(curves: Array, src_w: int, src_h: int) -> void:
	frame_width = src_w
	frame_height = src_h

	_curve_points.clear()
	_render_points.clear()
	_render_ts.clear()
	_miter_normals.clear()

	if curves.size() < 2:
		_has_ribbon = false
		_clear_meshes()
		return

	for p in curves:
		if p is Array and p.size() >= 2:
			_curve_points.append(_video_pixel_to_local(Vector2(float(p[0]), float(p[1]))))
		elif p is Dictionary:
			_curve_points.append(_video_pixel_to_local(
				Vector2(float(p.get("x", 0)), float(p.get("y", 0)))
			))

	if _curve_points.size() < 2:
		_has_ribbon = false
		_clear_meshes()
		return

	_smooth_curve(_curve_points, smooth_iterations)
	_curve_total_length = _get_polyline_length(_curve_points)
	_build_sampled_path(
		_curve_points, _render_points, _render_ts,
		render_segments, _curve_total_length
	)
	_render_total_length = _get_polyline_length(_render_points)
	_compute_miter_normals()
	_has_ribbon = true

	_build_static_mesh()
	_build_flow_mesh()


func _clear_meshes() -> void:
	if _static_mesh_node:
		_static_mesh_node.mesh = null
	if _flow_mesh_node:
		_flow_mesh_node.mesh = null


# ═══════════════════════════════════════════════════════════════
# Miter Join 法线计算
# ═══════════════════════════════════════════════════════════════

func _compute_miter_normals() -> void:
	var n := _render_points.size()
	_miter_normals.resize(n)

	if n < 2:
		return

	for i in range(n):
		if i == 0:
			var tangent := (_render_points[1] - _render_points[0]).normalized()
			_miter_normals[i] = Vector2(-tangent.y, tangent.x)
		elif i == n - 1:
			var tangent := (_render_points[n - 1] - _render_points[n - 2]).normalized()
			_miter_normals[i] = Vector2(-tangent.y, tangent.x)
		else:
			var d1 := (_render_points[i] - _render_points[i - 1]).normalized()
			var d2 := (_render_points[i + 1] - _render_points[i]).normalized()
			var n1 := Vector2(-d1.y, d1.x)
			var n2 := Vector2(-d2.y, d2.x)

			var miter := n1 + n2
			if miter.length_squared() < 0.0001:
				_miter_normals[i] = n1
			else:
				miter = miter.normalized()
				var scale := 1.0 / maxf(abs(n1.dot(miter)), 0.0001)
				if scale > miter_limit:
					scale = miter_limit   # bevel 截断
				_miter_normals[i] = miter * scale


# ═══════════════════════════════════════════════════════════════
# 静态 Mesh 构建（glow + body + center 合并为一个 mesh）
# ═══════════════════════════════════════════════════════════════

func _build_static_mesh() -> void:
	if not _has_ribbon or _render_points.size() < 2:
		return

	var n := _render_points.size()
	var vcount := n * VERTS_PER_SEGMENT
	var vertices := PackedVector2Array()
	var colors := PackedColorArray()
	var indices := PackedInt32Array()
	vertices.resize(vcount)
	colors.resize(vcount)

	# 计算每段的半宽（用于顶点位置）
	var half_widths: Array[float] = []
	half_widths.resize(n)
	for i in range(n):
		var perspective_t := _render_ts[i]
		var width_t := _smooth01(perspective_t)
		half_widths[i] = lerpf(near_width + glow_extra_width, far_width + glow_extra_width, width_t) * 0.5

	# 填充顶点
	for i in range(n):
		var p := _render_points[i]
		var miter_n := _miter_normals[i]   # 左向 miter 法线
		var hw := half_widths[i]
		var perspective_t := _render_ts[i]

		for j in range(VERTS_PER_SEGMENT):
			var s : float= CROSS_SIDES[j]
			var offset :float= s * hw

			if j < VERTS_PER_SIDE:
				# 左侧
				vertices[i * VERTS_PER_SEGMENT + j] = p + miter_n * offset
			elif j == VERTS_PER_SIDE:
				# 中心
				vertices[i * VERTS_PER_SEGMENT + j] = p
			else:
				# 右侧
				vertices[i * VERTS_PER_SEGMENT + j] = p - miter_n * offset

			colors[i * VERTS_PER_SEGMENT + j] = _compute_vertex_color(s, perspective_t)

	# ── 跨截面颜色模糊（消除生硬过渡）──
	if cross_blur > 0.001:
		_apply_cross_blur(colors, n)

	# 填充三角形索引
	var half := VERTS_PER_SIDE   # = 9, 中心索引
	for i in range(n - 1):
		var base0 := i * VERTS_PER_SEGMENT
		var base1 := (i + 1) * VERTS_PER_SEGMENT

		# 左侧条带 (j=0..half-1) → 连接中心
		for j in range(half):
			indices.append(base0 + j)
			indices.append(base0 + j + 1)
			indices.append(base1 + j)

			indices.append(base1 + j)
			indices.append(base0 + j + 1)
			indices.append(base1 + j + 1)

		# 中心 → 右侧 条带（修复缝隙）
		indices.append(base0 + half)
		indices.append(base0 + half + 1)
		indices.append(base1 + half)

		indices.append(base1 + half)
		indices.append(base0 + half + 1)
		indices.append(base1 + half + 1)

		# 右侧条带 (j=half+1..VERTS_PER_SEGMENT-2)
		for j in range(half + 1, VERTS_PER_SEGMENT - 1):
			indices.append(base0 + j)
			indices.append(base0 + j + 1)
			indices.append(base1 + j)

			indices.append(base1 + j)
			indices.append(base0 + j + 1)
			indices.append(base1 + j + 1)

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = indices

	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	_static_mesh_node.mesh = mesh


# ── 跨截面高斯模糊（3-tap，沿截面方向平滑颜色过渡）──
func _apply_cross_blur(cols: PackedColorArray, seg_count: int) -> void:
	var blurred := cols.duplicate()
	var strength := cross_blur

	for i in range(seg_count):
		var base := i * VERTS_PER_SEGMENT

		# 左侧（跳过端点 j=0，保持边缘不模糊）
		for j in range(1, VERTS_PER_SIDE):
			var c0 := cols[base + j - 1]
			var c1 := cols[base + j]
			var c2 := cols[base + j + 1]
			var blended := c0 * (strength * 0.5) + c1 * (1.0 - strength) + c2 * (strength * 0.5)
			blurred[base + j] = blended

		# 中心（左右各取一个邻居）
		var center_j := VERTS_PER_SIDE
		var c_left := cols[base + center_j - 1]
		var c_center := cols[base + center_j]
		var c_right := cols[base + center_j + 1]
		blurred[base + center_j] = c_left * (strength * 0.5) + c_center * (1.0 - strength) + c_right * (strength * 0.5)

		# 右侧（跳过端点）
		for j in range(VERTS_PER_SIDE + 1, VERTS_PER_SEGMENT - 1):
			var c0 := cols[base + j - 1]
			var c1 := cols[base + j]
			var c2 := cols[base + j + 1]
			var blended := c0 * (strength * 0.5) + c1 * (1.0 - strength) + c2 * (strength * 0.5)
			blurred[base + j] = blended

	# 写回
	for i in range(seg_count):
		var base := i * VERTS_PER_SEGMENT
		for j in range(1, VERTS_PER_SEGMENT - 1):
			cols[base + j] = blurred[base + j]


# ═══════════════════════════════════════════════════════════════
# 顶点颜色计算 —— 完全对照 shader fragment()
# ═══════════════════════════════════════════════════════════════

func _compute_vertex_color(normalized_side: float, perspective_t: float) -> Color:
	var width_t := _smooth01(perspective_t)
	var max_w := lerpf(near_width + glow_extra_width, far_width + glow_extra_width, width_t)
	var body_w := lerpf(near_width, far_width, width_t)
	var center_w := lerpf(center_light_near_width, center_light_far_width, width_t)

	var body_ratio := clampf(body_w / maxf(max_w, 0.001), 0.0, 1.0)
	var center_ratio := clampf(center_w / maxf(max_w, 0.001), 0.0, 1.0)

	var combined_r := 0.0
	var combined_g := 0.0
	var combined_b := 0.0
	var combined_a := 0.0

	# ── glow 层 ──
	if draw_glow:
		var glow_edge_near := Color(glow_near_color.r, glow_near_color.g, glow_near_color.b, glow_near_color.a * 0.55)
		var glow_edge_far := Color(glow_far_color.r, glow_far_color.g, glow_far_color.b, glow_far_color.a * 0.55)
		var g := _make_ribbon_layer(normalized_side, 1.0, perspective_t,
			glow_near_color, glow_far_color, glow_edge_near, glow_edge_far)
		combined_r += g.r * g.a
		combined_g += g.g * g.a
		combined_b += g.b * g.a
		combined_a += g.a * 0.55

	# ── body 层 ──
	var b := _make_ribbon_layer(normalized_side, body_ratio, perspective_t,
		near_color, far_color, edge_near_color, edge_far_color)
	combined_r += b.r * b.a
	combined_g += b.g * b.a
	combined_b += b.b * b.a
	combined_a += b.a

	# ── center 层 ──
	if draw_center_light:
		var c := _make_ribbon_layer(normalized_side, center_ratio, perspective_t,
			center_light_near_color, center_light_far_color,
			center_light_near_color, center_light_far_color)
		combined_r += c.r * c.a
		combined_g += c.g * c.a
		combined_b += c.b * c.a
		combined_a += c.a * 0.65

	combined_a = clampf(combined_a, 0.0, 1.0)
	return Color(combined_r, combined_g, combined_b, combined_a)


# 对应 shader make_ribbon_layer()
func _make_ribbon_layer(
	normalized_side: float,
	width_ratio: float,
	perspective_t: float,
	core_near: Color,
	core_far: Color,
	edge_near: Color,
	edge_far: Color
) -> Color:
	if normalized_side >= width_ratio or width_ratio < 0.0001:
		return Color(0, 0, 0, 0)

	var s := normalized_side / width_ratio
	var alpha_profile := _ribbon_alpha(s)
	var core_color := core_near.lerp(core_far, perspective_t)
	var edge_color := edge_near.lerp(edge_far, perspective_t)
	var edge_mix := _smoothstep_cpu(0.15, 1.0, s)
	var color := core_color.lerp(edge_color, edge_mix)
	color.a *= alpha_profile
	return color


# 对应 shader ribbon_alpha()
func _ribbon_alpha(x: float) -> float:
	x = clampf(x, 0.0, 1.0)
	if x <= 0.48:
		return lerpf(1.0, 0.86, x / 0.48)
	if x <= 0.82:
		return lerpf(0.86, 0.28, (x - 0.48) / 0.34)
	return lerpf(0.28, 0.0, (x - 0.82) / 0.18)


# ═══════════════════════════════════════════════════════════════
# 流光 Mesh 构建
# ═══════════════════════════════════════════════════════════════

func _build_flow_mesh() -> void:
	if not _has_ribbon or not draw_flow_light or _render_points.size() < 2:
		_flow_mesh_node.mesh = null
		return

	var n := _render_points.size()
	var cycle := maxf(1.0, flow_segment_length + flow_segment_gap)

	var vertices := PackedVector2Array()
	var colors := PackedColorArray()
	var indices := PackedInt32Array()

	var dist_acc := 0.0

	for i in range(n - 1):
		var seg_len := _render_points[i].distance_to(_render_points[i + 1])
		var seg_mid := dist_acc + seg_len * 0.5

		var local_d := fmod(seg_mid - _flow_offset + cycle * 100.0, cycle)
		if local_d > flow_segment_length:
			dist_acc += seg_len
			continue

		var seg_t := clampf(local_d / maxf(flow_segment_length, 0.001), 0.0, 1.0)
		var flow_mask := sin(seg_t * PI)
		if flow_mask < 0.01:
			dist_acc += seg_len
			continue

		var t0 := _render_ts[i]
		var t1 := _render_ts[i + 1]
		var w0 := _smooth01(t0)
		var w1 := _smooth01(t1)

		var center_w0 := lerpf(center_light_near_width, center_light_far_width, w0) * 0.5 * 1.35
		var center_w1 := lerpf(center_light_near_width, center_light_far_width, w1) * 0.5 * 1.35

		var fc0 := flow_near_color.lerp(flow_far_color, t0)
		var fc1 := flow_near_color.lerp(flow_far_color, t1)
		fc0.a *= flow_mask * 1.25
		fc1.a *= flow_mask * 1.25

		# 对流光段也使用 miter 法线
		var n0 := _miter_normals[i] if i < _miter_normals.size() else Vector2.UP
		var n1 := _miter_normals[i + 1] if i + 1 < _miter_normals.size() else Vector2.UP

		var p0 := _render_points[i]
		var p1 := _render_points[i + 1]

		var vbase := vertices.size()

		# 四边形顶点
		vertices.append(p0 - n0 * center_w0)   # 0: left-top
		vertices.append(p0 + n0 * center_w0)   # 1: right-top
		vertices.append(p1 - n1 * center_w1)   # 2: left-bottom
		vertices.append(p1 + n1 * center_w1)   # 3: right-bottom

		colors.append(fc0)
		colors.append(fc0)
		colors.append(fc1)
		colors.append(fc1)

		# 两个三角形
		indices.append(vbase)
		indices.append(vbase + 1)
		indices.append(vbase + 2)

		indices.append(vbase + 1)
		indices.append(vbase + 3)
		indices.append(vbase + 2)

		dist_acc += seg_len

	if vertices.size() == 0:
		_flow_mesh_node.mesh = null
		return

	# ── 流光沿路径方向模糊（消除生硬块状感）──
	if cross_blur > 0.001:
		var quad_count := vertices.size() / 4
		if quad_count >= 3:
			var blurred := colors.duplicate()
			var bs := cross_blur * 0.8  # 流光模糊强度稍弱
			for q in range(1, quad_count - 1):
				var b0 := q * 4
				var b_prev := (q - 1) * 4
				var b_next := (q + 1) * 4
				for k in range(4):
					blurred[b0 + k] = colors[b_prev + k] * (bs * 0.5) + colors[b0 + k] * (1.0 - bs) + colors[b_next + k] * (bs * 0.5)
			for q in range(1, quad_count - 1):
				var b0 := q * 4
				for k in range(4):
					colors[b0 + k] = blurred[b0 + k]

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = indices

	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	_flow_mesh_node.mesh = mesh


# ═══════════════════════════════════════════════════════════════
# 坐标映射
# ═══════════════════════════════════════════════════════════════

func _video_pixel_to_local(pixel: Vector2) -> Vector2:
	var rect := get_rect()
	if rect.size.x < 1.0 or rect.size.y < 1.0:
		rect = get_viewport_rect()

	var video_aspect := float(frame_width) / maxf(1.0, float(frame_height))
	var rect_aspect := rect.size.x / maxf(1.0, rect.size.y)

	var draw_w: float
	var draw_h: float
	var offset_x := 0.0
	var offset_y := 0.0

	if use_aspect_fill:
		if rect_aspect > video_aspect:
			draw_w = rect.size.x
			draw_h = draw_w / video_aspect
			offset_y = (rect.size.y - draw_h) * 0.5
		else:
			draw_h = rect.size.y
			draw_w = draw_h * video_aspect
			offset_x = (rect.size.x - draw_w) * 0.5
	else:
		if rect_aspect > video_aspect:
			draw_h = rect.size.y
			draw_w = draw_h * video_aspect
			offset_x = (rect.size.x - draw_w) * 0.5
		else:
			draw_w = rect.size.x
			draw_h = draw_w / video_aspect
			offset_y = (rect.size.y - draw_h) * 0.5

	var x01 := pixel.x / maxf(1.0, float(frame_width))
	var y01 := pixel.y / maxf(1.0, float(frame_height))
	return Vector2(offset_x + x01 * draw_w, offset_y + y01 * draw_h)


# ═══════════════════════════════════════════════════════════════
# 路径采样
# ═══════════════════════════════════════════════════════════════

func _build_sampled_path(
	source: Array[Vector2],
	out_points: Array[Vector2],
	out_ts: Array[float],
	segments: int,
	total_length: float
) -> void:
	out_points.clear()
	out_ts.clear()

	if total_length <= 1.0:
		out_points.append_array(source)
		for i in range(source.size()):
			out_ts.append(_get_perspective_t(
				float(i) / maxf(1.0, float(source.size() - 1))
			))
		return

	var count := maxi(2, segments + 1)
	for i in range(count):
		var normalized := float(i) / float(count - 1)
		var dist := normalized * total_length
		var result := _get_point_and_tangent(source, dist)
		out_points.append(result[0])
		out_ts.append(_get_perspective_t(normalized))


# ═══════════════════════════════════════════════════════════════
# 几何工具
# ═══════════════════════════════════════════════════════════════

func _get_polyline_length(points: Array[Vector2]) -> float:
	var length := 0.0
	for i in range(points.size() - 1):
		length += points[i].distance_to(points[i + 1])
	return length


func _get_point_and_tangent(points: Array[Vector2], target_dist: float) -> Array:
	var travelled := 0.0
	for i in range(points.size() - 1):
		var seg_len := points[i].distance_to(points[i + 1])
		if travelled + seg_len >= target_dist:
			var local_t := (target_dist - travelled) / maxf(seg_len, 0.0001)
			var point := points[i].lerp(points[i + 1], local_t)
			var tangent := points[i + 1] - points[i]
			if tangent.length_squared() < 0.0001:
				tangent = Vector2.UP
			else:
				tangent = tangent.normalized()
			return [point, tangent]
		travelled += seg_len
	return [points[points.size() - 1], Vector2.UP]


func _get_perspective_t(normalized_dist: float) -> float:
	normalized_dist = clampf(normalized_dist, 0.0, 1.0)
	return normalized_dist if curve_starts_near else 1.0 - normalized_dist


func _smooth01(t: float) -> float:
	t = clampf(t, 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)


func _smoothstep_cpu(edge0: float, edge1: float, x: float) -> float:
	var t := clampf((x - edge0) / (edge1 - edge0), 0.0, 1.0)
	return t * t * (3.0 - 2.0 * t)


func _smooth_curve(points: Array[Vector2], iterations: int) -> void:
	if points.size() < 4 or iterations <= 0:
		return
	_smooth_buffer.resize(points.size())
	for _it in range(iterations):
		for i in range(points.size()):
			_smooth_buffer[i] = points[i]
		for i in range(1, points.size() - 1):
			points[i] = (
				_smooth_buffer[i - 1]
				+ _smooth_buffer[i] * 2.0
				+ _smooth_buffer[i + 1]
			) * 0.25
