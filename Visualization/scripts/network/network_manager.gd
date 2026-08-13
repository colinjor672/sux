extends Node

@onready var tcp_client: NavTCPClient = $NavTCPClient
@onready var ui_root = get_node_or_null("../Node2D/Camera2D/CanvasLayer/UIRoot")
@onready var shm_reader = get_node_or_null("../ShmVideoReader")

func _ready():
	tcp_client.nav_packet_received.connect(_on_nav_data)

	# 兜底：@onready 可能因为节点初始化顺序找不到，用绝对路径再试
	if shm_reader == null:
		shm_reader = get_node_or_null("../Node2D/ShmVideoReader")

	if shm_reader:
		shm_reader.video_frame_received.connect(_on_video_frame)
		print("[Manager] ShmVideoReader 已连接 ✓")
	else:
		push_error("[Manager] ShmVideoReader 未找到！")

	print("[Manager] 当前节点路径: ", get_path())

	if ui_root == null:
		push_error("[Manager] UIRoot 未找到！路径: ../Node2D/Camera2D/CanvasLayer/UIRoot")
		print("[Manager] 父节点: ", get_parent().get_path() if get_parent() else "null")
	else:
		print("[Manager] UIRoot 已找到: ", ui_root.get_path())

func _on_nav_data(packet: Dictionary):
	if ui_root == null:
		return

	var band = ui_root.get_node_or_null("NavigationBand")
	if band and band.has_method("set_curve_from_packet"):
		band.set_curve_from_packet(packet)

	var mask = ui_root.get_node_or_null("MaskLayer")
	if mask and mask.has_method("update_masks"):
		mask.update_masks(packet)

	var ships = ui_root.get_node_or_null("ShipLayer")
	if ships and ships.has_method("update_ships"):
		ships.update_ships(packet)



func _on_video_frame(_frame_id: int, w: int, h: int, img: Image):

	if ui_root == null:
		print("[Manager] 黑屏原因: ui_root is null")
		return

	var video = ui_root.get_node_or_null("VideoDisplay")
	if video == null:
		print("[Manager] 黑屏原因: VideoDisplay 节点找不到，UIRoot 子节点有: ")
		for child in ui_root.get_children():
			print("  - ", child.name)
		return

	if not video.has_method("update_frame"):
		print("[Manager] 黑屏原因: VideoDisplay 没有 update_frame 方法")
		return

	video.update_frame(w, h, img)
