extends SceneTree


const TEST_CASES := {
	"straight": [
		[640.0, 719.0],
		[640.0, 620.0],
		[640.0, 500.0],
		[640.0, 360.0],
	],
	"hard_left": [
		[640.0, 719.0],
		[638.0, 700.0],
		[630.0, 680.0],
		[616.0, 660.0],
		[596.0, 640.0],
		[570.0, 620.0],
		[538.0, 600.0],
		[500.0, 580.0],
		[456.0, 560.0],
		[406.0, 540.0],
		[350.0, 520.0],
	],
	"hard_right": [
		[640.0, 719.0],
		[642.0, 700.0],
		[650.0, 680.0],
		[664.0, 660.0],
		[684.0, 640.0],
		[710.0, 620.0],
		[742.0, 600.0],
		[780.0, 580.0],
		[824.0, 560.0],
		[874.0, 540.0],
		[930.0, 520.0],
	],
}

var _failures: Array[String] = []


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	var band := NavigationBandRenderer.new()
	root.add_child(band)
	band.set_anchors_preset(Control.PRESET_TOP_LEFT)
	band.position = Vector2.ZERO
	band.size = Vector2(1280.0, 720.0)

	await process_frame
	_assert(
		band.get_rect().size == Vector2(1280.0, 720.0),
		"test fixture did not create a 1280x720 render area"
	)

	for case_name in TEST_CASES:
		_validate_case(band, case_name, TEST_CASES[case_name])

	_validate_fixed_edge_width_at_720(band)

	if not _failures.is_empty():
		for failure in _failures:
			push_error(failure)
		quit(1)
		return

	print("NavigationBand geometry tests passed: ", TEST_CASES.keys())
	quit(0)


func _validate_case(
	band: NavigationBandRenderer,
	case_name: String,
	curve: Array
) -> void:
	band.call("_apply_curve", curve, 1280, 720)

	var mesh_node := band.get_node("NavigationRibbonMesh") as MeshInstance2D
	_assert(mesh_node.mesh != null, "%s: mesh was not built" % case_name)

	var arrays := mesh_node.mesh.surface_get_arrays(0)
	var vertices: PackedVector2Array = arrays[Mesh.ARRAY_VERTEX]
	var centers: Array = band.get("_render_points")
	var half_widths: Array = band.get("_mesh_half_widths")

	_assert(
		vertices.size() == centers.size() * 2,
		"%s: vertex count does not match sampled path" % case_name
	)
	_assert(
		half_widths.size() == centers.size(),
		"%s: width count does not match sampled path" % case_name
	)

	for i in range(centers.size()):
		var center: Vector2 = centers[i]
		var half_width: float = half_widths[i]
		var path_t := float(i) / maxf(1.0, float(centers.size() - 1))
		var perspective_t: float = band.call("_get_perspective_t", path_t)
		var smooth_t: float = band.call("_smooth01", perspective_t)
		var expected_half_width := lerpf(
			band.near_width + band.glow_extra_width,
			band.far_width + band.glow_extra_width,
			smooth_t
		) * 0.5
		var left := vertices[i * 2]
		var right := vertices[i * 2 + 1]
		var expected_y := 720.0 if i == 0 else center.y

		_assert_close(
			half_width,
			expected_half_width,
			"%s: ribbon smooth width[%d]" % [case_name, i]
		)
		_assert_close(left.y, expected_y, "%s: left y[%d]" % [case_name, i])
		_assert_close(right.y, expected_y, "%s: right y[%d]" % [case_name, i])
		_assert_close(left.x, center.x - half_width, "%s: left x[%d]" % [case_name, i])
		_assert_close(right.x, center.x + half_width, "%s: right x[%d]" % [case_name, i])

	var left_line := band.get_node("NavigationEdgeLeft") as Line2D
	var right_line := band.get_node("NavigationEdgeRight") as Line2D
	_assert(left_line.visible, "%s: left edge Line2D is hidden" % case_name)
	_assert(right_line.visible, "%s: right edge Line2D is hidden" % case_name)
	_assert(left_line.points.size() == centers.size(), "%s: left edge point count" % case_name)
	_assert(right_line.points.size() == centers.size(), "%s: right edge point count" % case_name)
	_assert_close(left_line.width, band.edge_line_width, "%s: left edge fixed width" % case_name)
	_assert_close(right_line.width, band.edge_line_width, "%s: right edge fixed width" % case_name)
	_assert(left_line.width_curve == null, "%s: left edge has a width curve" % case_name)
	_assert(right_line.width_curve == null, "%s: right edge has a width curve" % case_name)

	for i in range(centers.size()):
		var path_t := float(i) / maxf(1.0, float(centers.size() - 1))
		var perspective_t: float = band.call("_get_perspective_t", path_t)
		var line_offset := lerpf(
			band.edge_line_offset_near,
			band.edge_line_offset_far,
			perspective_t
		)
		var expected_y: float = 720.0 if i == 0 else centers[i].y

		_assert_close(
			left_line.points[i].x,
			centers[i].x - line_offset,
			"%s: left edge linear offset[%d]" % [case_name, i]
		)
		_assert_close(
			right_line.points[i].x,
			centers[i].x + line_offset,
			"%s: right edge linear offset[%d]" % [case_name, i]
		)
		_assert_close(left_line.points[i].y, expected_y, "%s: left edge y[%d]" % [case_name, i])
		_assert_close(right_line.points[i].y, expected_y, "%s: right edge y[%d]" % [case_name, i])

	var material := mesh_node.material as ShaderMaterial
	_assert(
		not bool(material.get_shader_parameter("draw_edge_lines")),
		"%s: shader edge lines are still enabled" % case_name
	)


func _validate_fixed_edge_width_at_720(
	band: NavigationBandRenderer
) -> void:
	band.size = Vector2(720.0, 720.0)
	band.call("_apply_curve", TEST_CASES["hard_left"], 1280, 720)

	var left_line := band.get_node("NavigationEdgeLeft") as Line2D
	var right_line := band.get_node("NavigationEdgeRight") as Line2D

	_assert_close(
		left_line.width,
		band.edge_line_width,
		"720-wide viewport: left edge fixed width"
	)
	_assert_close(
		right_line.width,
		band.edge_line_width,
		"720-wide viewport: right edge fixed width"
	)
	_assert(
		left_line.width_curve == null,
		"720-wide viewport: left edge has a width curve"
	)
	_assert(
		right_line.width_curve == null,
		"720-wide viewport: right edge has a width curve"
	)


func _assert(condition: bool, message: String) -> void:
	if condition:
		return

	_failures.append(message)


func _assert_close(actual: float, expected: float, label: String) -> void:
	_assert(
		is_equal_approx(actual, expected),
		"%s: expected %.4f, got %.4f" % [label, expected, actual]
	)
