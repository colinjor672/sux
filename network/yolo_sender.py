import socket
import threading
import time
import json as _json
from typing import List

from utils.thread_utils import set_thread_name

class YoloSender:
    def __init__(self, host='0.0.0.0', port=9000, max_lag_frames=5):
        self.host = host
        self.port = port
        self.max_lag_frames = int(max_lag_frames)

        self._lock = threading.Lock()
        self._clients: List[socket.socket] = []

        self._slot = None
        self._event = threading.Event()

        self._running = False
        self.client_count = 0
        self.sent = 0
        self.dropped = 0
        self._latest_frame = -1

    def start(self):
        self._running = True
        threading.Thread(
            target=self._server_loop,
            daemon=True,
            name="YoloSenderAccept"
        ).start()
        threading.Thread(
            target=self._send_loop,
            daemon=True,
            name="YoloSenderSend"
        ).start()

    def stop(self):
        self._running = False
        self._event.set()
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except:
                    pass
            self._clients.clear()

        print(
            f"[YoloSender] 停止 | sent={self.sent} "
            f"dropped={self.dropped}"
        )

    def send_ships(self, ships, frame_idx, width, height):
        if not self._running:
            return

        with self._lock:
            self._latest_frame = max(self._latest_frame, int(frame_idx))

            if self._slot is not None:
                self.dropped += 1

            self._slot = (
                list(ships) if ships else [],
                int(frame_idx),
                int(width),
                int(height),
            )

        self._event.set()

    def _send_loop(self):
        set_thread_name("YoloSend")
        while self._running:
            self._event.wait(timeout=0.05)
            self._event.clear()

            if not self._running:
                break

            with self._lock:
                item = self._slot
                self._slot = None
                latest = self._latest_frame

            if item is None:
                continue

            ships, frame_idx, width, height = item

            if latest - frame_idx > self.max_lag_frames:
                self.dropped += 1
                continue

            self._broadcast(ships, frame_idx, width, height)

    def _broadcast(self, ships, frame_idx, width, height):
        with self._lock:
            if not self._clients:
                return

        msg = {
            "type": "yolo_detections",
            "frame_id": frame_idx,
            "timestamp": time.time(),
            "image_width": width,
            "image_height": height,
            "detections": [{
                "ship_id": int(s.get("ship_id", 0)),
                "label": str(s.get("label", "ship")),
                "bbox": [int(v) for v in s["bbox"]],
                "center": [float(v) for v in s["center"]],
                "confidence": float(s.get("conf", 0)),
            } for s in ships]
        }

        data = (_json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")

        dead = []

        with self._lock:
            for c in self._clients:
                try:
                    c.sendall(data)
                except Exception:
                    dead.append(c)

            for c in dead:
                try:
                    self._clients.remove(c)
                except ValueError:
                    pass
                try:
                    c.close()
                except:
                    pass

            self.client_count = len(self._clients)

        self.sent += 1

    def _server_loop(self):
        set_thread_name("YoloAccept")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        srv.bind((self.host, self.port))
        srv.listen(5)

        print(f"[YoloSender] 监听 {self.host}:{self.port}")

        while self._running:
            try:
                client, addr = srv.accept()
                client.settimeout(0.05)

                with self._lock:
                    self._clients.append(client)
                    self.client_count = len(self._clients)

                print(f"[YoloSender] 🔗 融合端已连接: {addr}")

            except socket.timeout:
                continue
            except OSError:
                break

        try:
            srv.close()
        except:
            pass