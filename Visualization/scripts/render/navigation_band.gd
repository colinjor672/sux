# NavigationBandRenderer.gd
class_name NavigationBandRenderer
extends Control


@export_group("视频原始尺寸")
@export var frame_width := 1280
@export var frame_height := 720


@export_group("坐标映射")
@export var use_aspect_fill := true
@export var curve_starts_near := true
@export var seal_near_end_to_video_bottom := true
@export_group("AR 导航带主体")
@export var near_width := 130.0
@export var far_width := 60.2
@export var near_color := Color(0.05, 0.85, 1.0, 0.60)
@export var far_color := Color(0.05, 0.55, 1.0, 0.10)

@export var edge_near_color := Color(0.0, 0.95, 1.0, 0.35)
@export var edge_far_color := Color(0.0, 0.45, 1.0, 0.00)

@export_group("外层发光")
@export var draw_glow := true
@export var glow_extra_width := 60.0
@export var glow_near_color := Color(0.0, 0.75, 1.0, 0.45)
@export var glow_far_color := Color(0.0, 0.35, 1.0, 0.12)

@export_group("导航带边界细线")
@export var draw_edge_lines := true
@export var edge_line_width := 5.0
@export var edge_line_near_color := Color(0.0, 0.85, 1.0, 0.95)
@export var edge_line_far_color := Color(0.0, 0.55, 1.0, 0.25)
@export var edge_line_offset_near := 90.0
@export var edge_line_offset_far := 60.0

@export_group("中心柔光线")
@export var draw_center_light := true
@export var center_light_near_width := 60.2
@export var center_light_far_width := 22.4
@export var center_light_near_color := Color(0.75, 1.0, 1.0, 0.45)
@export var center_light_far_color := Color(0.75, 1.0, 1.0, 0.08)


@export_group("中心流动光效")
@export var draw_flow_light := true
@export var flow_speed := 150.0
@export var flow_segment_length := 120.0
@export var flow_segment_gap := 170.0
@export var flow_near_color := Color(0.9, 1.0, 1.0, 0.85)
@export var flow_far_color := Color(0.6, 0.9, 1.0, 0.08)
@export var reverse_flow_direction := false

@export_group("曲线质量")
@export_range(0, 10)
var smooth_iterations := 10
@export_range(16, 180)
var render_segments := 150

@export_group("更新频率")
@export_range(5, 60)
var flow_fps := 14

@export_range(5, 60)
var max_curve_update_fps := 18

var _curve_points: Array[Vector2] = []
var _render_points: Array[Vector2] = []
var _render_ts: Array[float] = []
var _smooth_buffer: Array[Vector2] = []
var _mesh_half_widths: Array[float] = []


var _last_curve_update_time := -1.0

var _pending_curves: Array = []
var _pending_src_w := 0
var _pending_src_h := 0
var _has_pending_curve := false

var _curve_total_length := 0.0
var _render_total_length := 0.0


const RIBBON_SHADER_PATH := \
	"res://scripts/render/navigation_ribbon.gdshader"

const RIBBON_SHADER: Shader = preload(
	"res://scripts/render/navigation_ribbon.gdshader"
)


# 主体、外发光、中心柔光和流光
var _ribbon_mesh_node: MeshInstance2D = null
var _ribbon_material: ShaderMaterial = null
var _ribbon_ready := false


# 左右边界细线
var _line_left: Line2D = null
var _line_right: Line2D = null


func _ready() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)
	mouse_filter = Control.MOUSE_FILTER_IGNORE

	_init_ribbon_mesh()
	_init_edge_lines()


# ════════════════════════════════════════════
# 主 Mesh 初始化
# ════════════════════════════════════════════

func _init_ribbon_mesh() -> void:
	_destroy_ribbon_mesh("init_reset")

	var ribbon_shader := _resolve_ribbon_shader()

	if ribbon_shader == null:
		push_error(
			"[NavigationBand] Shader 加载失败："
			+ RIBBON_SHADER_PATH
		)
		return

	var mesh_node := MeshInstance2D.new()

	mesh_node.name = "NavigationRibbonMesh"
	mesh_node.z_index = 10
	mesh_node.visible = true

	var material := ShaderMaterial.new()
	material.shader = ribbon_shader

	if material.shader == null:
		push_error("[NavigationBand] ShaderMaterial 绑定失败")
		mesh_node.free()
		return

	mesh_node.material = material

	_ribbon_mesh_node = mesh_node
	_ribbon_material = material

	add_child(_ribbon_mesh_node)

	_ribbon_ready = true

	_sync_shader_params()

	print(
		"[NavigationBand] ShaderMesh 就绪：",
		RIBBON_SHADER_PATH
	)


func _resolve_ribbon_shader() -> Shader:
	if (
		RIBBON_SHADER != null
		and not String(RIBBON_SHADER.resource_path).is_empty()
	):
		return RIBBON_SHADER

	if ResourceLoader.exists(RIBBON_SHADER_PATH):
		var loaded := load(RIBBON_SHADER_PATH) as Shader

		if loaded != null:
			return loaded

	return null


func _destroy_ribbon_mesh(reason: String = "") -> void:
	var had_node := (
		_ribbon_mesh_node != null
		or _ribbon_material != null
	)

	_ribbon_ready = false

	if _ribbon_mesh_node != null:
		_ribbon_mesh_node.mesh = null
		_ribbon_mesh_node.material = null

		if is_instance_valid(_ribbon_mesh_node):
			if _ribbon_mesh_node.get_parent() != null:
				_ribbon_mesh_node.get_parent().remove_child(
					_ribbon_mesh_node
				)

			_ribbon_mesh_node.queue_free()

		_ribbon_mesh_node = null

	_ribbon_material = null

	if had_node and not reason.is_empty():
		print(
			"[NavigationBand] 已销毁 Ribbon Mesh：",
			reason
		)


# ════════════════════════════════════════════
# 左右边界细线初始化
# ════════════════════════════════════════════

func _init_edge_lines() -> void:
	_line_left = _create_edge_line(
		"NavigationEdgeLeft"
	)

	_line_right = _create_edge_line(
		"NavigationEdgeRight"
	)


func _create_edge_line(node_name: String) -> Line2D:
	var line := Line2D.new()

	line.name = node_name
	line.z_index = 12
	line.visible = draw_edge_lines

	line.width = edge_line_width
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


# ════════════════════════════════════════════
# 主循环
# ════════════════════════════════════════════

func _process(_delta: float) -> void:
	var now := Time.get_ticks_msec() / 1000.0

	if not _has_pending_curve:
		return

	var update_interval := (
		1.0
		/ maxf(
			1.0,
			float(max_curve_update_fps)
		)
	)

	if (
		_last_curve_update_time < 0.0
		or now - _last_curve_update_time >= update_interval
	):
		_last_curve_update_time = now
		_has_pending_curve = false

		_apply_curve(
			_pending_curves,
			_pending_src_w,
			_pending_src_h
		)


# ════════════════════════════════════════════
# 公开接口
# ════════════════════════════════════════════

func set_curve_from_packet(packet: Dictionary) -> void:
	var curves = packet.get(
		"curves",
		packet.get("curve", [])
	)

	var src_w: int = packet.get(
		"coord_w",
		packet.get("width", 1280)
	)

	var src_h: int = packet.get(
		"coord_h",
		packet.get("height", 720)
	)

	set_curve(curves, src_w, src_h)


func set_curve(
	curves: Array,
	src_w: int,
	src_h: int
) -> void:
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

	_mesh_half_widths.clear()

	_curve_total_length = 0.0
	_render_total_length = 0.0

	_clear_ribbon_mesh()
	_clear_edge_lines()


# ════════════════════════════════════════════
# 路径处理
# ════════════════════════════════════════════

func _apply_curve(
	curves: Array,
	src_w: int,
	src_h: int
) -> void:
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
			video_point = Vector2(
				float(item[0]),
				float(item[1])
			)

			valid_point = true

		elif item is Dictionary:
			video_point = Vector2(
				float(item.get("x", 0)),
				float(item.get("y", 0))
			)

			valid_point = true

		if not valid_point:
			continue

		var local_point := _video_pixel_to_local(
			video_point
		)

		# 清理连续重复点，防止零长度线段。
		if not _curve_points.is_empty():
			var previous := _curve_points[
				_curve_points.size() - 1
			]

			if local_point.distance_squared_to(previous) < 0.0001:
				continue

		_curve_points.append(local_point)

	if _curve_points.size() < 2:
		_clear_all_geometry()
		return

	_smooth_curve(
		_curve_points,
		smooth_iterations
	)

	_curve_total_length = _get_polyline_length(
		_curve_points
	)

	if _curve_total_length <= 0.001:
		_clear_all_geometry()
		return

	_build_sampled_path(
		_curve_points,
		_render_points,
		_render_ts,
		render_segments,
		_curve_total_length
	)

	_render_total_length = _get_polyline_length(
		_render_points
	)

	_compute_strip_widths()

	_rebuild_ribbon_mesh()
	_update_edge_lines()
	_sync_shader_params()


func _clear_all_geometry() -> void:
	_curve_total_length = 0.0
	_render_total_length = 0.0

	_mesh_half_widths.clear()

	_clear_ribbon_mesh()
	_clear_edge_lines()


func _compute_strip_widths() -> void:
	var point_count := _render_points.size()

	_mesh_half_widths.resize(point_count)

	if point_count < 2:
		return

	for i in range(point_count):
		var path_t := (
			float(i)
			/ maxf(
				1.0,
				float(point_count - 1)
			)
		)

		var perspective_t := _get_perspective_t(
			path_t
		)

		var width_t := _smooth01(
			perspective_t
		)

		_mesh_half_widths[i] = (
			lerpf(
				near_width + glow_extra_width,
				far_width + glow_extra_width,
				width_t
			)
			* 0.5
		)

func _clear_ribbon_mesh() -> void:
	if (
		not _ribbon_ready
		or _ribbon_mesh_node == null
	):
		return

	_ribbon_mesh_node.mesh = null


func _rebuild_ribbon_mesh() -> void:
	if not _ribbon_ready:
		return

	if (
		_ribbon_mesh_node == null
		or _ribbon_material == null
		or _ribbon_material.shader == null
	):
		return

	var point_count := _render_points.size()

	if point_count < 2:
		_clear_ribbon_mesh()
		return

	if (
		_mesh_half_widths.size() != point_count
	):
		_clear_ribbon_mesh()
		return

	var vertices := PackedVector2Array()
	var uvs := PackedVector2Array()
	var near_endpoint_index := (
		0
		if curve_starts_near
		else point_count - 1
	)
	var video_bottom_y := _get_video_bottom_local_y()

	vertices.resize(point_count * 2)
	uvs.resize(point_count * 2)

	for i in range(point_count):
		var center := _render_points[i]
		var half_width := _mesh_half_widths[i]

		var left := (
			center
			- Vector2(half_width, 0.0)
		)

		var right := (
			center
			+ Vector2(half_width, 0.0)
		)

		# 屏幕空间平行条带：所有截面均保持水平，边界和中心路径
		# 具有相同形状，急转时不会因法线偏移产生内凹或外凸。
		if (
			seal_near_end_to_video_bottom
			and i == near_endpoint_index
		):
			left = Vector2(
				center.x - half_width,
				video_bottom_y
			)
			right = Vector2(
				center.x + half_width,
				video_bottom_y
			)

		var path_t := (
			float(i)
			/ maxf(
				1.0,
				float(point_count - 1)
			)
		)

		vertices[i * 2] = left
		vertices[i * 2 + 1] = right

		# 手工 Mesh 原始 UV 方向：
		# UV.x = 带宽横向
		# UV.y = 路径纵向
		uvs[i * 2] = Vector2(
			0.0,
			path_t
		)

		uvs[i * 2 + 1] = Vector2(
			1.0,
			path_t
		)

	var indices := PackedInt32Array()

	for i in range(point_count - 1):
		var left_current := i * 2
		var right_current := i * 2 + 1

		var left_next := (i + 1) * 2
		var right_next := (i + 1) * 2 + 1

		indices.append(left_current)
		indices.append(left_next)
		indices.append(right_current)

		indices.append(right_current)
		indices.append(left_next)
		indices.append(right_next)

	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)

	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = indices

	var mesh := ArrayMesh.new()

	mesh.add_surface_from_arrays(
		Mesh.PRIMITIVE_TRIANGLES,
		arrays
	)

	_ribbon_mesh_node.mesh = mesh


# ════════════════════════════════════════════
# 左右边界细线更新
# ════════════════════════════════════════════

func _update_edge_lines() -> void:
	if (
		_line_left == null
		or _line_right == null
	):
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

	# ════════════════════════════════════════
	# 颜色渐变
	# ════════════════════════════════════════

	var gradient := Gradient.new()
	var offsets := PackedFloat32Array()
	var colors := PackedColorArray()

	for i in range(point_count):
		var path_t := (
			float(i)
			/ maxf(
				1.0,
				float(point_count - 1)
			)
		)

		var perspective_t := _get_perspective_t(
			path_t
		)

		var width_t := _smooth01(
			perspective_t
		)

		offsets.append(path_t)

		colors.append(
			edge_line_near_color.lerp(
				edge_line_far_color,
				width_t
			)
		)

	gradient.offsets = offsets
	gradient.colors = colors

	# ════════════════════════════════════════
	# 应用颜色与宽度
	# ════════════════════════════════════════

	_line_left.gradient = gradient

	_line_right.gradient = (
		gradient.duplicate()
		as Gradient
	)

	_line_left.width = edge_line_width
	_line_right.width = edge_line_width
	_line_left.width_curve = null
	_line_right.width_curve = null

	_line_left.clear_points()
	_line_right.clear_points()

	var near_endpoint_index := (
		0
		if curve_starts_near
		else point_count - 1
	)
	var video_bottom_y := _get_video_bottom_local_y()


	for i in range(point_count):
		var path_t := (
			float(i)
			/ maxf(
				1.0,
				float(point_count - 1)
			)
		)

		var perspective_t := _get_perspective_t(
			path_t
		)

		var center := _render_points[i]
		var line_offset := lerpf(
			edge_line_offset_near,
			edge_line_offset_far,
			perspective_t
		)

		var left_point := center - Vector2(line_offset, 0.0)
		var right_point := center + Vector2(line_offset, 0.0)

		if (
			seal_near_end_to_video_bottom
			and i == near_endpoint_index
		):
			left_point = Vector2(
				center.x - line_offset,
				video_bottom_y
			)
			right_point = Vector2(
				center.x + line_offset,
				video_bottom_y
			)

		_line_right.add_point(
			right_point
		)

		_line_left.add_point(
			left_point
		)

func _sync_shader_params() -> void:
	if not _ribbon_ready:
		return

	if (
		_ribbon_material == null
		or _ribbon_material.shader == null
	):
		return

	_ribbon_material.set_shader_parameter(
		"near_at_start",
		curve_starts_near
	)

	_ribbon_material.set_shader_parameter(
		"draw_glow",
		draw_glow
	)

	_ribbon_material.set_shader_parameter(
		"draw_center_light",
		draw_center_light
	)

	_ribbon_material.set_shader_parameter(
		"draw_flow_light",
		draw_flow_light
	)

	_ribbon_material.set_shader_parameter(
		"draw_edge_lines",
		false
	)

	_ribbon_material.set_shader_parameter(
		"edge_line_near_color",
		edge_line_near_color
	)

	_ribbon_material.set_shader_parameter(
		"edge_line_far_color",
		edge_line_far_color
	)

	_ribbon_material.set_shader_parameter(
		"near_width",
		near_width
	)

	_ribbon_material.set_shader_parameter(
		"far_width",
		far_width
	)

	_ribbon_material.set_shader_parameter(
		"glow_extra_width",
		glow_extra_width
	)

	_ribbon_material.set_shader_parameter(
		"center_light_near_width",
		center_light_near_width
	)

	_ribbon_material.set_shader_parameter(
		"center_light_far_width",
		center_light_far_width
	)

	_ribbon_material.set_shader_parameter(
		"near_color",
		near_color
	)

	_ribbon_material.set_shader_parameter(
		"far_color",
		far_color
	)

	_ribbon_material.set_shader_parameter(
		"edge_near_color",
		edge_near_color
	)

	_ribbon_material.set_shader_parameter(
		"edge_far_color",
		edge_far_color
	)

	_ribbon_material.set_shader_parameter(
		"glow_near_color",
		glow_near_color
	)

	_ribbon_material.set_shader_parameter(
		"glow_far_color",
		glow_far_color
	)

	_ribbon_material.set_shader_parameter(
		"center_light_near_color",
		center_light_near_color
	)

	_ribbon_material.set_shader_parameter(
		"center_light_far_color",
		center_light_far_color
	)

	_ribbon_material.set_shader_parameter(
		"flow_speed",
		flow_speed)
	_ribbon_material.set_shader_parameter("flow_segment_length",flow_segment_length)
	_ribbon_material.set_shader_parameter("flow_segment_gap",flow_segment_gap)
	_ribbon_material.set_shader_parameter("reverse_flow_direction",reverse_flow_direction)
	_ribbon_material.set_shader_parameter("flow_near_color",flow_near_color)
	_ribbon_material.set_shader_parameter("flow_far_color",flow_far_color)
	_ribbon_material.set_shader_parameter("path_length",maxf(_render_total_length,1.0))


# ════════════════════════════════════════════
# 坐标映射
# ════════════════════════════════════════════

func _video_pixel_to_local(pixel: Vector2) -> Vector2:
	var rect := get_rect()

	var video_aspect := (
		float(frame_width)
		/ maxf(
			1.0,
			float(frame_height)
		)
	)

	var rect_aspect := (
		rect.size.x
		/ maxf(
			1.0,
			rect.size.y
		)
	)

	var draw_width := 0.0
	var draw_height := 0.0

	var offset_x := 0.0
	var offset_y := 0.0

	if use_aspect_fill:
		if rect_aspect > video_aspect:
			draw_width = rect.size.x
			draw_height = (
				draw_width
				/ video_aspect
			)

			offset_y = (
				rect.size.y
				- draw_height
			) * 0.5
		else:
			draw_height = rect.size.y
			draw_width = (
				draw_height
				* video_aspect
			)

			offset_x = (
				rect.size.x
				- draw_width
			) * 0.5
	else:
		if rect_aspect > video_aspect:
			draw_height = rect.size.y
			draw_width = (
				draw_height
				* video_aspect
			)

			offset_x = (
				rect.size.x
				- draw_width
			) * 0.5
		else:
			draw_width = rect.size.x
			draw_height = (
				draw_width
				/ video_aspect
			)

			offset_y = (
				rect.size.y
				- draw_height
			) * 0.5

	var normalized_x := (
		pixel.x
		/ maxf(
			1.0,
			float(frame_width)
		)
	)

	var normalized_y := (
		pixel.y
		/ maxf(
			1.0,
			float(frame_height)
		)
	)

	return Vector2(
		offset_x
		+ normalized_x * draw_width,

		offset_y
		+ normalized_y * draw_height
	)


func _get_video_bottom_local_y() -> float:
	return _video_pixel_to_local(
		Vector2(0.0, float(frame_height))
	).y


# ════════════════════════════════════════════
# 路径采样
# ════════════════════════════════════════════

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
			var normalized := (
				float(i)
				/ maxf(
					1.0,
					float(source.size() - 1)
				)
			)

			out_ts.append(
				_get_perspective_t(
					normalized
				)
			)

		return

	var count := maxi(
		2,
		segments + 1
	)

	for i in range(count):
		var normalized := (
			float(i)
			/ float(count - 1)
		)

		var target_distance := (
			normalized
			* total_length
		)

		var result := _get_point_and_tangent(
			source,
			target_distance
		)

		out_points.append(result[0])

		out_ts.append(
			_get_perspective_t(
				normalized
			)
		)


# ════════════════════════════════════════════
# 几何工具
# ════════════════════════════════════════════

func _get_polyline_length(
	points: Array[Vector2]
) -> float:
	var total := 0.0

	for i in range(points.size() - 1):
		total += points[i].distance_to(
			points[i + 1]
		)

	return total


func _get_point_and_tangent(
	points: Array[Vector2],
	target_distance: float
) -> Array:
	var travelled := 0.0

	for i in range(points.size() - 1):
		var segment_length := points[i].distance_to(
			points[i + 1]
		)

		if (
			travelled + segment_length
			>= target_distance
		):
			var local_t := (
				target_distance
				- travelled
			) / maxf(
				segment_length,
				0.0001
			)

			var point := points[i].lerp(
				points[i + 1],
				local_t
			)

			var tangent := (
				points[i + 1]
				- points[i]
			)

			if tangent.length_squared() > 0.0001:
				tangent = tangent.normalized()
			else:
				tangent = Vector2.UP

			return [
				point,
				tangent
			]

		travelled += segment_length

	return [
		points[points.size() - 1],
		Vector2.UP
	]


func _get_perspective_t(
	normalized_distance: float
) -> float:
	normalized_distance = clampf(
		normalized_distance,
		0.0,
		1.0
	)

	return (
		normalized_distance
		if curve_starts_near
		else 1.0 - normalized_distance
	)


func _smooth01(t: float) -> float:
	t = clampf(t,0.0,1.0)
	return t * t * (3.0 - 2.0 * t)


func _smooth_curve(
	points: Array[Vector2],
	iterations: int
) -> void:
	if (
		points.size() < 4
		or iterations <= 0
	):
		return

	_smooth_buffer.resize(
		points.size()
	)

	for _iteration in range(iterations):
		for i in range(points.size()):
			_smooth_buffer[i] = points[i]

		for i in range(
			1,
			points.size() - 1
		):
			points[i] = (
				_smooth_buffer[i - 1]
				+ _smooth_buffer[i] * 2.0
				+ _smooth_buffer[i + 1]
			) * 0.25
