extends Control

## HUD 信息显示

@onready var fps_label: Label = $FPSLabel if has_node("FPSLabel") else null

func _process(_delta):
	if fps_label:
		fps_label.text = "FPS: %d" % Engine.get_frames_per_second()
