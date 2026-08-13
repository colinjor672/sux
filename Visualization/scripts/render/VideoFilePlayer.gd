extends VideoStreamPlayer

@export_file("*.ogv", "*.webm", "*.mp4", "*.mov") var video_path := "res://IMG.mov"

func _ready():
	set_anchors_preset(Control.PRESET_FULL_RECT)
	expand = true

	var s := load(video_path)
	if s == null:
		push_error("无法加载视频文件: " + video_path)
		return

	stream = s
	stop()

func start_video():
	if stream != null:
		play()

func get_video_time() -> float:
	return stream_position
