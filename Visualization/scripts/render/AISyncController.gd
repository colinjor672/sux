extends Node


@export var prebuffer_seconds := 0.60
@export var ai_ttl := 0.35
@export var debug_sync := true

@onready var video_player: VideoStreamPlayer = $"../VideoPlayer"
@onready var mask_layer = $"../MaskLayer"
@onready var navigation_band = $"../NavigationBand"
@onready var ship_layer = $"../ShipLayer"

var _ai_buffer: Array = []
var _video_started := false
var _last_applied_key := -1

func _ready():

	video_player.stop()
	_video_started = false

	if debug_sync:
		print("[AISync] 等待 Python AI 数据预缓冲...")

func on_ai_packet(packet: Dictionary):
	# Python 发来的 packet 必须包含 video_time
	if not packet.has("video_time"):
		return

	_ai_buffer.append(packet)

	# 控制缓存长度，避免无限增长
	while _ai_buffer.size() > 300:
		_ai_buffer.pop_front()

	# 第一阶段同步：
	# 等 Python 至少发到 prebuffer_seconds 后，Godot 再开始播放视频。
	# 这样 Godot 播放时，AI 数据已经提前准备好了。
	if not _video_started:
		var newest_time := _get_newest_video_time()
		if newest_time >= prebuffer_seconds:
			video_player.play()
			_video_started = true
			if debug_sync:
				print("[AISync] 视频开始播放，AI预缓冲=", newest_time)

func _process(_delta):
	if not _video_started:
		return

	var current_video_time := video_player.stream_position
	var packet := _get_packet_for_video_time(current_video_time)

	if packet.is_empty():
		return

	var key := int(packet.get("mask_frame_id", packet.get("frame_id", -1)))
	if key == _last_applied_key:
		return

	_last_applied_key = key

	_apply_ai_packet(packet)

	if debug_sync:
		var pt := float(packet.get("video_time", 0.0))
		var diff := current_video_time - pt
		print(
			"[AISync] video=", snapped(current_video_time, 0.001),
			" packet=", snapped(pt, 0.001),
			" diff=", snapped(diff, 0.001),
			" key=", key
		)

func _get_newest_video_time() -> float:
	if _ai_buffer.is_empty():
		return 0.0

	var newest := 0.0
	for p in _ai_buffer:
		newest = maxf(newest, float(p.get("video_time", 0.0)))

	return newest

func _get_packet_for_video_time(video_time: float) -> Dictionary:
	if _ai_buffer.is_empty():
		return {}

	var best: Dictionary = {}
	var best_diff := 999999.0

	for p in _ai_buffer:
		var t := float(p.get("video_time", 0.0))
		var diff := absf(t - video_time)

		if diff < best_diff:
			best_diff = diff
			best = p

	if best.is_empty():
		return {}

	if best_diff > ai_ttl:
		return {}

	return best

func _apply_ai_packet(packet: Dictionary):
	if mask_layer != null:
		mask_layer.update_masks(packet)

	if navigation_band != null:
		navigation_band.set_curve_from_packet(packet)

	if ship_layer != null:
		# 这里根据你 ShipLayer 的接口改
		# 如果你的函数叫 update_ships_from_packet，就用你的函数名
		if ship_layer.has_method("update_from_packet"):
			ship_layer.update_from_packet(packet)
		elif ship_layer.has_method("update_ships"):
			var ships = packet.get("ships", packet.get("ships_data", []))
			ship_layer.update_ships(ships)
