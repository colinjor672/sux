class_name NavTCPClient
extends Node

signal nav_packet_received(packet: Dictionary)

@export var nav_host := "127.0.0.1"
@export var nav_port := 8765
@export var config_file_name := "config.json"

var _nav_tcp := StreamPeerTCP.new()
var _nav_buf := PackedByteArray()

const RECONNECT_INTERVAL_MS := 2000
const OUTER_HEADER := 4
const INNER_HEADER := 16

var _next_nav_reconnect_ms := 0

func _ready():
	_load_external_config()
	_connect_servers()

func _load_external_config() -> void:
	var config_path := _get_external_config_path()

	if not FileAccess.file_exists(config_path):
		push_warning("配置文件不存在，使用默认地址：%s。配置路径：%s" % [nav_host, config_path])
		return

	var file := FileAccess.open(config_path, FileAccess.READ)
	if file == null:
		push_error("无法打开配置文件：%s，错误码：%s" % [config_path, FileAccess.get_open_error()])
		return

	var json_text := file.get_as_text()
	var parsed = JSON.parse_string(json_text)

	if not parsed is Dictionary:
		push_error("config.json 格式错误，必须是 JSON 对象")
		return

	var config: Dictionary = parsed
	nav_host = str(config.get("nav_host", nav_host))
	nav_port = int(config.get("nav_port", nav_port))

func _get_external_config_path() -> String:
	if OS.has_feature("editor"):
		return ProjectSettings.globalize_path("res://%s" % config_file_name)
	return OS.get_executable_path().get_base_dir().path_join(config_file_name)

func _connect_servers() -> void:
	_nav_tcp = StreamPeerTCP.new()
	_nav_buf.clear()
	var nav_error := _nav_tcp.connect_to_host(nav_host, nav_port)
	if nav_error != OK:
		push_error("导航连接发起失败: %d" % nav_error)

func _check_nav_connection() -> void:
	_nav_tcp.poll()
	var status := _nav_tcp.get_status()
	# 已连接或正在连接，不需要重连
	if status == StreamPeerTCP.STATUS_CONNECTED or status == StreamPeerTCP.STATUS_CONNECTING:
		return

	# 没连上才重连
	var now := Time.get_ticks_msec()
	if now < _next_nav_reconnect_ms:
		return

	_next_nav_reconnect_ms = now + RECONNECT_INTERVAL_MS
	print("[TCP] 重连导航服务器: ", nav_host, ":", nav_port)

	_nav_tcp = StreamPeerTCP.new()
	_nav_buf.clear()
	var err := _nav_tcp.connect_to_host(nav_host, nav_port)
	if err != OK:
		push_warning("[TCP] 导航重连发起失败: %d" % err)

func _process(_delta):
	_check_nav_connection()
	_poll_nav()

func _poll_nav():
	_nav_tcp.poll()
	if _nav_tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return

	var available := _nav_tcp.get_available_bytes()
	if available <= 0:
		return

	var result := _nav_tcp.get_data(available)
	if result[0] != OK:
		return

	_nav_buf.append_array(result[1])
	_parse_nav_messages()

func _parse_nav_messages():
	while true:
		if _nav_buf.size() < OUTER_HEADER:
			break

		var message_length := _nav_buf.decode_u32(0)
		var total_needed := OUTER_HEADER + message_length

		if _nav_buf.size() < total_needed:
			break

		var payload := _nav_buf.slice(OUTER_HEADER, total_needed)
		_nav_buf = _nav_buf.slice(total_needed)

		if payload.size() < INNER_HEADER:
			continue

		var json_len := payload.decode_u32(12)
		if payload.size() < INNER_HEADER + json_len:
			continue

		var json_bytes := payload.slice(INNER_HEADER, INNER_HEADER + json_len)
		var json_str := json_bytes.get_string_from_utf8()

		if json_str.length() == 0:
			continue

		var parsed = JSON.parse_string(json_str)
		if parsed != null and parsed is Dictionary:
			nav_packet_received.emit(parsed)
