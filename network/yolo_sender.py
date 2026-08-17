import json
import socket
import struct
import threading
import time

from utils.thread_utils import set_thread_name  # 若无此模块可注释或自行实现


class YoloSender:
    MAGIC = 0x4655534E
    VERSION = 1
    MSG_YOLO_DETECTIONS = 2
    HEADER_STRUCT = struct.Struct(">IHHIQII")
    SHIP_CLASS_ID = 0
    BRIDGE_PIER_CLASS_ID = 1
    SEND_CLASS_IDS = frozenset((SHIP_CLASS_ID, BRIDGE_PIER_CLASS_ID))

    def __init__(
        self,
        host="0.0.0.0",          # 监听所有网卡
        port=9000,
        max_lag_frames=5,
        reconnect_interval=1.0,  # 此处用于等待新客户端连接间隔
        connect_timeout=2.0,     # 用于 accept 超时（实际上用不到，保留兼容）
        send_timeout=1.0,
    ):
        self.host = str(host)
        self.port = int(port)
        self.max_lag_frames = int(max_lag_frames)
        self.reconnect_interval = float(reconnect_interval)
        self.connect_timeout = float(connect_timeout)  # 未使用，可忽略
        self.send_timeout = float(send_timeout)

        self._lock = threading.Lock()
        self._listen_sock = None       # 监听 socket
        self._client_sock = None       # 当前连接的客户端 socket
        self._slot = None              # 待发送数据
        self._event = threading.Event()
        self._send_thread = None
        self._accept_thread = None
        self._seq = 0

        self._running = False
        self.client_count = 0          # 统计已连接客户端数量（累计）
        self.sent = 0
        self.dropped = 0
        self._latest_frame = -1

    def start(self):
        """启动服务端，开始监听并启动后台线程"""
        if self._running:
            return

        self._running = True

        # 创建监听 socket
        self._listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen_sock.bind((self.host, self.port))
        self._listen_sock.listen(5)
        # 设置 accept 超时，便于 stop 时退出阻塞
        self._listen_sock.settimeout(1.0)

        # 启动接受连接线程
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            daemon=True,
            name="YoloAccept",
        )
        self._accept_thread.start()

        # 启动发送线程
        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name="YoloSend",
        )
        self._send_thread.start()

        print(f"[YoloSender] listening on {self.host}:{self.port}")

    def stop(self):
        """停止服务，关闭所有连接和线程"""
        self._running = False
        self._event.set()   # 唤醒发送线程

        # 关闭监听 socket（让 accept 退出）
        with self._lock:
            if self._listen_sock:
                try:
                    self._listen_sock.close()
                except OSError:
                    pass
                self._listen_sock = None

        # 关闭当前客户端连接
        self._close_client_socket()

        # 等待线程结束
        for thr in (self._accept_thread, self._send_thread):
            if thr and thr.is_alive() and threading.current_thread() is not thr:
                thr.join(timeout=self.send_timeout + 2.0)

        print(
            f"[YoloSender] stopped | sent={self.sent} "
            f"dropped={self.dropped} | client_count={self.client_count}"
        )

    def send_ships(self, ships, frame_idx, width, height, timestamp=None):
        """外部接口：放入待发送数据（线程安全）"""
        if not self._running:
            return

        frame_timestamp = time.time() if timestamp is None else float(timestamp)

        with self._lock:
            self._latest_frame = max(self._latest_frame, int(frame_idx))

            if self._slot is not None:
                self.dropped += 1

            self._slot = (
                list(ships) if ships else [],
                int(frame_idx),
                int(width),
                int(height),
                frame_timestamp,
            )

        self._event.set()

    # ---------- 内部线程函数 ----------

    def _accept_loop(self):
        """循环接受客户端连接（只保持一个连接）"""
        set_thread_name("YoloAccept")
        while self._running:
            try:
                if self._listen_sock is None:
                    break
                client_sock, addr = self._listen_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                # 可能被关闭
                break

            if not self._running:
                client_sock.close()
                break

            # 如果有旧客户端，先关闭（只支持一个客户端）
            with self._lock:
                if self._client_sock is not None:
                    old = self._client_sock
                    self._client_sock = None
                    # 不在这里关闭 old，避免死锁，放到后面
                else:
                    old = None

            if old:
                try:
                    old.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                old.close()

            # 设置新客户端
            client_sock.settimeout(self.send_timeout)
            with self._lock:
                self._client_sock = client_sock
                self.client_count += 1

            print(f"[YoloSender] client connected from {addr} (total: {self.client_count})")
            # 唤醒发送线程
            self._event.set()

    def _send_loop(self):
        """主发送循环：检查客户端连接，发送待发送数据"""
        set_thread_name("YoloSend")
        while self._running:
            # 等待数据或连接事件
            self._event.wait(timeout=0.1)
            self._event.clear()

            if not self._running:
                break

            # 获取当前客户端
            sock = None
            with self._lock:
                sock = self._client_sock

            if sock is None:
                # 无客户端，继续等待（accept 线程会唤醒我们）
                continue

            # 取数据
            with self._lock:
                item = self._slot
                self._slot = None
                latest = self._latest_frame

            if item is None:
                continue

            ships, frame_idx, _width, _height, frame_timestamp = item

            # 丢弃过时帧
            if latest - frame_idx > self.max_lag_frames:
                self.dropped += 1
                continue

            try:
                packet = self._build_packet(ships, frame_timestamp, self._seq)
                sock.sendall(packet)
            except (socket.error, OSError) as exc:
                self.dropped += 1
                print(f"[YoloSender] send failed: {exc}, closing client")
                self._close_client_socket()   # 断开连接，等待重连
                continue

            self._seq = (self._seq + 1) & 0xFFFFFFFF
            self.sent += 1

    # ---------- 辅助方法 ----------

    def _close_client_socket(self):
        """安全关闭当前客户端 socket（线程安全）"""
        with self._lock:
            sock = self._client_sock
            self._client_sock = None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
            print("[YoloSender] client disconnected")

    @classmethod
    def _build_packet(cls, ships, timestamp, seq):
        """构建船只和桥墩共用的 YOLO 检测数据包。"""
        frame_timestamp = float(timestamp)
        detections = []
        for target in ships:
            class_id = int(target.get("class_id", cls.SHIP_CLASS_ID))
            if class_id not in cls.SEND_CLASS_IDS:
                continue

            detections.append(
                {
                    "id": int(target.get("ship_id", target.get("id", 0))),
                    "x1": float(target["bbox"][0]),
                    "y1": float(target["bbox"][1]),
                    "x2": float(target["bbox"][2]),
                    "y2": float(target["bbox"][3]),
                    "score": float(target.get("conf", target.get("score", 0.0))),
                    "class_id": class_id,
                    "timestamp": frame_timestamp,
                }
            )

        payload_obj = {
            "timestamp": frame_timestamp,
            "yolo_detections": detections,
        }
        payload = json.dumps(
            payload_obj,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        header = cls.HEADER_STRUCT.pack(
            cls.MAGIC,
            cls.VERSION,
            cls.MSG_YOLO_DETECTIONS,
            len(payload),
            int(frame_timestamp * 1_000_000),
            int(seq) & 0xFFFFFFFF,
            0,
        )
        return header + payload
