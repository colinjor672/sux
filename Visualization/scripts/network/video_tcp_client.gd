class_name VideoTCPClient
extends Node

## TCP 视频帧接收客户端（替代 ShmVideoReader）
## 协议：每帧 = [4B msg_len][4B frame_id][4B width][4B height][N bytes JPEG]

signal video_frame_received(frame_id: int, width: int, height: int, img: Image)

@export var host := "127.0.0.1"
@export var port := 8766

const HEADER_SIZE := 16          # 4+4+4+4
const OUTER_SIZE  := 4           # 消息总长度前缀
const RECONNECT_INTERVAL_MS := 2000

var _tcp := StreamPeerTCP.new()
var _buf := PackedByteArray()
var _next_reconnect_ms := 0
var _frame_count := 0

func _ready():
	_connect()

func _connect() -> void:
	_tcp = StreamPeerTCP.new()
	_buf.clear()
	var err := _tcp.connect_to_host(host, port)
	if err != OK:
		push_error("[VideoTCP] 连接发起失败: %d" % err)

func _check_connection() -> void:
	_tcp.poll()
	var status := _tcp.get_status()
	if status == StreamPeerTCP.STATUS_CONNECTED or status == StreamPeerTCP.STATUS_CONNECTING:
		return

	var now := Time.get_ticks_msec()
	if now < _next_reconnect_ms:
		return
	_next_reconnect_ms = now + RECONNECT_INTERVAL_MS
	print("[VideoTCP] 重连视频服务器 %s:%d" % [host, port])
	_connect()

func _process(_delta):
	_check_connection()
	_poll()

func _poll():
	_tcp.poll()
	if _tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return

	var available := _tcp.get_available_bytes()
	if available <= 0:
		return

	var result := _tcp.get_data(available)
	if result[0] != OK:
		return

	_buf.append_array(result[1])
	_parse_frames()

func _parse_frames():
	while true:
		# 需要至少 4 字节读取消息总长度
		if _buf.size() < OUTER_SIZE:
			break

		var msg_len := _buf.decode_u32(0)
		if msg_len < HEADER_SIZE:
			# 无效消息，丢弃 4 字节重试
			_buf = _buf.slice(OUTER_SIZE)
			continue

		var total_needed := OUTER_SIZE + msg_len
		if _buf.size() < total_needed:
			break

		# 提取完整帧
		var payload := _buf.slice(OUTER_SIZE, total_needed)
		_buf = _buf.slice(total_needed)

		if payload.size() < HEADER_SIZE:
			continue

		var frame_id  := payload.decode_u32(0)
		var width     := payload.decode_u32(4)
		var height    := payload.decode_u32(8)
		var data_size := payload.decode_u32(12)

		var jpeg_data := payload.slice(HEADER_SIZE)
		if jpeg_data.size() != data_size:
			push_warning("[VideoTCP] JPEG 数据大小不匹配: %d != %d" % [jpeg_data.size(), data_size])
			continue

		var img := Image.new()
		var err := img.load_jpg_from_buffer(jpeg_data)
		if err != OK:
			push_warning("[VideoTCP] JPEG 解码失败: %d" % err)
			continue

		_frame_count += 1
		if _frame_count % 100 == 0:
			print("[VideoTCP] 已接收 %d 帧" % _frame_count)

		video_frame_received.emit(frame_id, width, height, img)
