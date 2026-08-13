extends SceneTree


const OUTPUT_DIR := "res://tests/artifacts"
const TEST_CASES := {
	"straight": [
		[640.0, 719.0],
		[640.0, 620.0],
		[640.0, 500.0],
		[640.0, 340.0],
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
		[288.0, 500.0],
		[220.0, 480.0],
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
		[992.0, 500.0],
		[1060.0, 480.0],
	],
}


func _initialize() -> void:
	call_deferred("_run")


func _run() -> void:
	root.size = Vector2i(1280, 720)

	var background := ColorRect.new()
	background.color = Color(0.025, 0.035, 0.045, 1.0)
	root.add_child(background)
	background.set_anchors_preset(Control.PRESET_TOP_LEFT)
	background.size = Vector2(1280.0, 720.0)

	var band := NavigationBandRenderer.new()
	root.add_child(band)
	band.set_anchors_preset(Control.PRESET_TOP_LEFT)
	band.position = Vector2.ZERO
	band.size = Vector2(1280.0, 720.0)

	var directory_error := DirAccess.make_dir_recursive_absolute(
		ProjectSettings.globalize_path(OUTPUT_DIR)
	)
	if directory_error != OK:
		push_error("Could not create render-test output directory: %s" % directory_error)
		quit(1)
		return

	await process_frame

	for case_name in TEST_CASES:
		band.call("_apply_curve", TEST_CASES[case_name], 1280, 720)
		await process_frame
		await process_frame
		RenderingServer.force_draw(false)

		var image := root.get_texture().get_image()
		var output_path := "%s/%s.png" % [OUTPUT_DIR, case_name]
		var save_error := image.save_png(output_path)
		if save_error != OK:
			push_error("Could not save %s: %s" % [output_path, save_error])
			quit(1)
			return

	print("NavigationBand render tests saved to ", OUTPUT_DIR)
	quit(0)
