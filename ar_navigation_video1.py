import os, sys

TRT_LIB = r"D:/BaiduNetdiskDownload/TensorRT-8.6.1.6/lib"
os.environ['PATH'] = TRT_LIB + os.pathsep + os.environ['PATH']
try:
    os.add_dll_directory(TRT_LIB)
except Exception:
    pass

# 2. 添加 PyTorch 的 CUDA 库路径 
TORCH_LIB = r"C:/Users/admin/miniconda4/envs/yolo11/Lib/site-packages/torch/lib"
os.environ['PATH'] = TORCH_LIB + os.pathsep + os.environ['PATH']
try:
    os.add_dll_directory(TORCH_LIB)
except Exception:
    pass

# 3. 添加 cuDNN 库路径 (conda 装的)
CUDNN_LIB = r"C:/Users/admin/miniconda4/envs/yolo11/Library/bin"
os.environ['PATH'] = CUDNN_LIB + os.pathsep + os.environ['PATH']
try:
    os.add_dll_directory(CUDNN_LIB)
except Exception:
    pass

_POLY_MAGIC = 0x594C4F50  
def _setup_onnxruntime_cuda():
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
            return torch_lib
    except Exception as e:
        print(f"⚠️ 注入 CUDA 路径失败: {e}")
    return None

_torch_lib = _setup_onnxruntime_cuda()
if _torch_lib:
    print(f"✓ 已注入 PyTorch CUDA/cuDNN 库: {_torch_lib}")


import torch.nn.functional as F
import argparse
import time
import queue
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List
from threading import Thread
import threading
import cv2
import numpy as np
import socket
from data_server import NavigationDataServer, prepare_frame_data
from ultralytics import YOLO
import struct
import math
import json as _json
import torch
from typing import Optional, Dict, Any
import tensorrt as trt
# 模型 & 超参
SHIP_ENGINE_PATH = (
    r"C:/Users/admin\Desktop/ultralytics-v11"
    r"/runs/bridge_ship2/yolo11n_bridge_ship2"
    r"/weights/best.engine"
)

SHIP_CONF = 0.45
SHIP_IOU = 0.45
SHIP_CLASS_NAMES = ["ship"]
SHIP_INPUT_SIZE = 1280

if not os.path.isfile(SHIP_ENGINE_PATH):
    raise FileNotFoundError(
        f"找不到YOLO TensorRT Engine：{SHIP_ENGINE_PATH}"
    )

SHIP_MODEL = YOLO(
    SHIP_ENGINE_PATH,
    task="detect",
)

warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try: cv2.setLogLevel(0)
except: pass

NUM_CLASSES         = 3
BRIDGE_CLASS_ID     = 0
WATER_CLASS_ID      = 1
BACKGROUND_CLASS_ID = 2

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

nav_server: Optional["NavigationDataServer"] = None


@dataclass
class RenderConfig:
    water_color:   Tuple[int, int, int] = (255, 220, 120)
    bridge_color:  Tuple[int, int, int] = (0, 0, 255)
    ar_near_color: Tuple[int, int, int] = (255, 230, 80)
    ar_far_color:  Tuple[int, int, int] = (255, 120, 0)
    water_alpha:   float = 0.14
    bridge_alpha:  float = 0.25


class TRTInferencer:
    def __init__(self, engine_path: str,
                 input_h: int = 432,
                 input_w: int = 768):
        self._logger = trt.Logger(trt.Logger.WARNING)
        self.input_h = input_h
        self.input_w = input_w

        print(f"加载 TRT 分割模型: {engine_path}  ({input_h}×{input_w})")
        self._engine = self._load(engine_path)
        self._ctx    = self._engine.create_execution_context()
        self._alloc()
        print("✓ TRTInferencer 初始化完成")

    def _load(self, path: str):
        runtime = trt.Runtime(self._logger)
        with open(path, 'rb') as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _alloc(self):
        self._bufs: List[Tuple[str, torch.Tensor]] = []
        self._out_tensors: List[torch.Tensor] = []
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            tdtype = torch.float16 if dtype == np.float16 else torch.float32
            t = torch.empty(shape, dtype=tdtype, device='cuda')
            self._bufs.append((name, t))
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self._out_tensors.append(t)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        
        stream = torch.cuda.current_stream()

        in_name, in_buf = self._bufs[0]
        in_buf.copy_(x, non_blocking=True)

        for name, tensor in self._bufs:
            self._ctx.set_tensor_address(
                name,
                tensor.data_ptr(),
            )

        ok = self._ctx.execute_async_v3(
            stream.cuda_stream
        )

        if not ok:
            raise RuntimeError(
                "TensorRT execute_async_v3执行失败"
            )

        # 此处不同步，由AsyncSegInferencer统一同步
        return self._out_tensors[0]

    def shutdown(self):
        pass

class TemporalMaskSmootherGPU:
    def __init__(self, alpha=0.7, threshold=0.5):
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self._ema = None

    def update(self, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=torch.float32)

        if self._ema is None or self._ema.shape != m.shape:
            self._ema = m.clone()
        else:
            self._ema.mul_(self.alpha).add_(m, alpha=(1.0 - self.alpha))

        if mask.dtype == torch.uint8:
            return (self._ema >= self.threshold).to(dtype=torch.uint8)
        elif mask.dtype == torch.int32:
            return self._ema.round().to(dtype=torch.int32)
        else:
            return (self._ema >= self.threshold).to(dtype=mask.dtype)

#  I/O 异步类
class AsyncVideoWriter:
    def __init__(self, video_writer, maxsize: int = 6):
        self.writer    = video_writer
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.running   = True
        self.dropped   = 0
        self.processed = 0
        self.thread = Thread(target=self._run, daemon=True, name="AsyncWriter")
        self.thread.start()

    def submit(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while self.running:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self.writer.write(item)
                self.processed += 1
            except Exception as e:
                print(f"⚠️ VideoWriter 错误: {e}")

    def shutdown(self):
        self.running = False
        try: self._q.put_nowait(None)
        except: pass
        self.thread.join(timeout=5)
        print(f"  📊 写盘统计: 处理 {self.processed} 帧, 丢弃 {self.dropped} 帧")


# 编码器选择（优先硬件）
_encoder = None
_encoder_name = "raw"

#try:
    #from pynvjpeg import NvJpeg
    #_encoder = NvJpeg()
    #_encoder_name = "nvjpeg"
#except ImportError:

try:
    from turbojpeg import TurboJPEG

    if sys.platform.startswith("win"):
        _encoder = TurboJPEG(lib_path=r"C:/libjpeg-turbo64/bin/turbojpeg.dll")
    else:
        _encoder = TurboJPEG()

    _encoder_name = "turbojpeg"

except (ImportError, RuntimeError, OSError) as e:
    print(f"⚠️ TurboJPEG 不可用，改用 cv2.imencode: {e}")
    _encoder = None
    _encoder_name = "cv2"

print(f"[VideoStream] 编码器: {_encoder_name}")


def _encode_jpeg(bgr_frame: np.ndarray, quality: int = 85) -> bytes:
    """根据可用编码器选择最快的 JPEG 编码方式"""
    if _encoder_name == "nvjpeg":
        return _encoder.encode(bgr_frame, quality)
    elif _encoder_name == "turbojpeg":
        return _encoder.encode(bgr_frame, quality=quality)
    else:
        # cv2 fallback（最慢）
        ret, buf = cv2.imencode('.jpg', bgr_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ret else b''


class AsyncTCPVideoStreamer:

    def __init__(self,
                 nav_server_ref,
                 stream_scale: float = 0.33,
                 send_interval: int = 1,
                 jpeg_quality: int = 85,
                 min_send_interval_ms: float = 18.0):
        self.nav_server = nav_server_ref
        self.stream_scale = float(np.clip(stream_scale, 0.1, 1.0))
        self.send_interval = max(1, int(send_interval))
        self.jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._slot = None
        self._event = threading.Event()
        self.running = True
        self.dropped = 0
        self.sent = 0
        self._send_ms_ema = 5.0
        self._min_interval = min_send_interval_ms / 1000.0
        self._last_send_time = 0.0

        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="VideoSend")
        self._send_thread.start()

        print(f"  视频流: encoder={_encoder_name}, scale={self.stream_scale:.2f}, "
              f"quality={self.jpeg_quality}, interval={self.send_interval}")

    def submit(self, frame: np.ndarray, frame_idx: int,
               orig_w: int, orig_h: int):
        # 自适应跳帧
        if self._send_ms_ema > 50.0:
            effective = max(self.send_interval, 3)
        elif self._send_ms_ema > 25.0:
            effective = max(self.send_interval, 2)
        else:
            effective = self.send_interval

        if frame_idx % effective != 0:
            return

        with self._lock:
            if self._slot is not None:
                self.dropped += 1
            self._slot = (frame, frame_idx, orig_w, orig_h)
        self._event.set()

    def _send_loop(self):
        while self.running:
            fired = self._event.wait(timeout=0.05)
            self._event.clear()
            if not fired:
                continue

            with self._lock:
                item, self._slot = self._slot, None
            if item is None:
                continue

            # 限流
            now = time.perf_counter()
            elapsed = now - self._last_send_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            frame, frame_idx, orig_w, orig_h = item
            t0 = time.perf_counter()

            try:
                # 缩放
                if self.stream_scale < 0.99:
                    nw = max(1, int(orig_w * self.stream_scale))
                    nh = max(1, int(orig_h * self.stream_scale))
                    send_frame = cv2.resize(
                        frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                else:
                    send_frame, nw, nh = frame, orig_w, orig_h

                # 编码（硬件 or CPU）
                jpeg_bytes = _encode_jpeg(send_frame, self.jpeg_quality)

                if jpeg_bytes:
                    self.nav_server.send_jpeg_video_frame(
                        jpeg_bytes, frame_idx, nw, nh)
                    self.sent += 1
                    self._last_send_time = time.perf_counter()

            except Exception as e:
                if self.sent % 200 == 1:
                    print(f"⚠️ VideoSend: {e}")

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._send_ms_ema = 0.15 * dt_ms + 0.85 * self._send_ms_ema

    def shutdown(self):
        self.running = False
        self._event.set()
        if hasattr(self, '_send_thread'):
            self._send_thread.join(timeout=2)
        print(f"  视频流统计: encoder={_encoder_name} sent={self.sent} "
              f"drop={self.dropped} latency≈{self._send_ms_ema:.1f}ms")


class FileVideoSender:
    """
    通过文件 I/O 将 RGBA 视频帧发送给 Godot 的 ShmVideoReader。

    Godot 的 ShmVideoReader 在 GDExtension 不可用时（Windows），
    回退到 FileAccess 模式读取以下文件：
      /dev/shm/godot_input_ctrl  → 8字节：write_index(u32 LE) + frame_id(u32 LE)
      /dev/shm/godot_input_0.raw → RGBA 640×360×4
      /dev/shm/godot_input_1.raw → RGBA 640×360×4（双缓冲）

    Windows 上 /dev/shm/ 会被 Godot 解析为 C:\\dev\\shm\\
    """

    INPUT_W = 640
    INPUT_H = 360
    INPUT_SIZE = INPUT_W * INPUT_H * 4  # RGBA = 921600 字节

    def __init__(self):
        self._write_index = 0
        self._frame_count = 0
        self._lock = threading.Lock()
        self._running = False
        self._error_count = 0

        # 路径：Windows 上使用 Godot 的 user:// 目录，与 ShmVideoReader.gd 对齐
        # Godot user:// 在 Windows 上映射到 %APPDATA%\Godot\app_userdata\<项目名>\
        if sys.platform.startswith("win"):
            appdata = os.environ.get('APPDATA', '')
            self._shm_dir = os.path.join(appdata, 'Godot', 'app_userdata', 'Visualization')
        else:
            # Jetson 上用 /dev/shm/（GDExtension 模式，不会走到文件回退）
            self._shm_dir = "/dev/shm"

        self._ctrl_path = os.path.join(self._shm_dir, "godot_input_ctrl")
        self._slot_0_path = os.path.join(self._shm_dir, "godot_input_0.raw")
        self._slot_1_path = os.path.join(self._shm_dir, "godot_input_1.raw")

    def start(self) -> bool:
        try:
            os.makedirs(self._shm_dir, exist_ok=True)
            self._running = True
            print(f"[FileVideoSender] ✓ 文件视频发送已启动 → {self._shm_dir}")
            print(f"  尺寸: {self.INPUT_W}×{self.INPUT_H} RGBA ({self.INPUT_SIZE} 字节/帧)")
            print(f"  双缓冲: {self._slot_0_path}")
            print(f"           {self._slot_1_path}")
            print(f"  控制:   {self._ctrl_path}")
            return True
        except Exception as e:
            print(f"[FileVideoSender] 启动失败: {e}")
            return False

    def send(self, bgr_frame: np.ndarray):
        """接收 BGR 任意尺寸帧 → 缩放到 640×360 → 转 RGBA → 写入文件"""
        if not self._running or bgr_frame is None:
            return

        with self._lock:
            self._write_index += 1
            self._frame_count += 1
            wi = self._write_index
            fid = self._frame_count

        try:
            # 缩放到 640×360
            if bgr_frame.shape[1] != self.INPUT_W or bgr_frame.shape[0] != self.INPUT_H:
                resized = cv2.resize(bgr_frame, (self.INPUT_W, self.INPUT_H),
                                     interpolation=cv2.INTER_LINEAR)
            else:
                resized = bgr_frame

            # BGR → RGBA
            rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
            rgba_bytes = rgba.tobytes()

            # 双缓冲写入
            slot_path = self._slot_0_path if (wi % 2) == 0 else self._slot_1_path

            with open(slot_path, 'wb') as f:
                f.write(rgba_bytes)

            # 更新控制文件（8字节：write_index + frame_id，均为 u32 LE）
            with open(self._ctrl_path, 'wb') as f:
                f.write(struct.pack('<II', wi, fid))

            if fid == 1:
                print(f"[FileVideoSender] 首帧已写入 {slot_path} ({len(rgba_bytes)} 字节)")
            elif fid % 200 == 0:
                print(f"[FileVideoSender] 已发送 {fid} 帧")

        except Exception as e:
            self._error_count += 1
            if self._error_count <= 3:
                print(f"[FileVideoSender] 写入失败: {e}")

    def stop(self):
        self._running = False
        print(f"[FileVideoSender] 已停止 (共 {self._frame_count} 帧, 错误 {self._error_count})")


class AsyncNavDataSender:
    def __init__(self, nav_server, mask_send_scale: float = 0.5,
             use_polygon_encoding: bool = True,
             polygon_epsilon: float = 0.005):
        self.nav_server      = nav_server
        self.mask_send_scale = float(max(0.25, min(1.0, mask_send_scale)))
        self.use_polygon_encoding = use_polygon_encoding
        self.polygon_epsilon      = polygon_epsilon
        self._poly_bytes_last     = 0
        self._lock   = threading.Lock()
        self._slot   = None
        self._event  = threading.Event()
        self.running = False
        self.dropped = 0
        self.sent    = 0

        self._send_method = self._find_send_method()

        self._thread = threading.Thread(
            target=self._run,
            name="AsyncNavSender",
            daemon=True
        )
        self.running = True
        self._thread.start()

    def _find_send_method(self):
        candidates = [
            'send_prepared_nav',
            'send_nav_data',
            'send_frame_data',
            'broadcast_nav',
            'send',
        ]
        for name in candidates:
            m = getattr(self.nav_server, name, None)
            if m is not None and callable(m):
                print(f"[AsyncNavSender] ✓ 使用发送方法: nav_server.{name}()")
                return m
        available = [
            attr for attr in dir(self.nav_server)
            if not attr.startswith('_') and callable(getattr(self.nav_server, attr))
        ]
        print(f"[AsyncNavSender] ✗ 未找到任何发送方法！可用方法: {available}")
        return None

    def submit(self, width, height, curve, frame_id, timestamp,
        water_mask, bridge_mask, ships_data,
        ship_src_w=None, ship_src_h=None,
        video_time: float = 0.0,
        sync_mode: str = "file_time",
        source_fps: float = 30.0,
        mask_frame_id: int = -1,
        mask_video_time: float = 0.0):
        if not self.running:
            return
        ship_src_w = int(ship_src_w) if ship_src_w is not None else int(width)
        ship_src_h = int(ship_src_h) if ship_src_h is not None else int(height)

        item = (
            width, height,
            curve.copy() if (curve is not None and len(curve) > 0) else curve,
            frame_id, timestamp,
            water_mask,
            bridge_mask,
            list(ships_data) if ships_data else [],
            ship_src_w,
            ship_src_h,
            float(video_time),
            str(sync_mode),
            float(source_fps),
            int(mask_frame_id),
            float(mask_video_time),
        )
        with self._lock:
            if self._slot is not None:
                self.dropped += 1
            self._slot = item
        self._event.set()

    def shutdown(self, timeout: float = 2.0):
        self.running = False
        self._event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("[AsyncNavSender] ⚠️ 后台线程未在超时内退出")
        else:
            print(f"[AsyncNavSender] ✓ 已停止 | 发送={self.sent} 丢弃={self.dropped}")
        if self.use_polygon_encoding and self._poly_bytes_last > 0:
            ratio = 460800 // max(1, self._poly_bytes_last)
            print(f"  多边形末帧大小: {self._poly_bytes_last}B  压缩率≈{ratio}x")
    
    def _run(self):
        
        while self.running:
            triggered = self._event.wait(timeout=0.05)
            self._event.clear()
            if not self.running:
                break

            with self._lock:
                item       = self._slot
                self._slot = None
            if item is None:
                continue

            (
                width, height,
                curve, frame_id, timestamp,
                water_mask, bridge_mask, ships_data,
                ship_src_w, ship_src_h,
                video_time, sync_mode, source_fps,
                mask_frame_id, mask_video_time,
            ) = item

            sc = self.mask_send_scale
            if sc < 1.0:
                sw = max(1, int(width  * sc))
                sh = max(1, int(height * sc))
                wm = cv2.resize(water_mask,  (sw, sh),
                                interpolation=cv2.INTER_NEAREST)
                bm = cv2.resize(bridge_mask, (sw, sh),
                                interpolation=cv2.INTER_NEAREST)
                if curve is not None and len(curve) > 0:
                    cx_s = sw / float(width)
                    cy_s = sh / float(height)
                    c = curve.astype(np.float32).copy()
                    c[:, 0] = np.clip(c[:, 0] * cx_s, 0, sw - 1)
                    c[:, 1] = np.clip(c[:, 1] * cy_s, 0, sh - 1)
                    curve = c.astype(np.int32)
                
                
                send_w, send_h = sw, sh
            else:
                wm = water_mask.copy()
                bm = bridge_mask.copy()
                send_w, send_h = width, height
            if ships_data:
                sx_ship = send_w / float(ship_src_w)
                sy_ship = send_h / float(ship_src_h)

                ships_data = _scale_ships_for_send(
                    ships_data,
                    sx_ship,
                    sy_ship,
                    dst_w=send_w,
                    dst_h=send_h,
                )

                
            try:
                if self._send_method is None:
                    self.dropped += 1
                    continue

                if self.use_polygon_encoding:
                    w_polys, b_polys = _extract_polygons_json(
                        wm, bm,
                        epsilon_ratio = self.polygon_epsilon,
                        min_area_px   = max(100, int(send_w * send_h * 0.001)),
                    )

                    self._poly_bytes_last = (
                        sum(len(p) for p in w_polys) +
                        sum(len(p) for p in b_polys)
                    ) * 12

                    empty = np.zeros((1, 1), dtype=np.uint8)
                    msg = prepare_frame_data(
                        width       = send_w,
                        height      = send_h,
                        curve       = curve,
                        frame_id    = frame_id,
                        timestamp   = timestamp,
                        water_mask  = empty,
                        bridge_mask = empty,
                        ships_data  = ships_data,
                    )
                    msg["source_type"] = "file"
                    msg["sync_mode"] = sync_mode

                    # 这个是 Godot 用来对齐视频播放进度的核心字段
                    # 建议用 mask_video_time，因为 mask/curve 来自分割帧
                    msg["video_time"] = float(mask_video_time)

                    msg["send_video_time"] = float(video_time)
                    msg["source_fps"] = float(source_fps)

                    msg["mask_frame_id"] = int(mask_frame_id)
                    msg["mask_video_time"] = float(mask_video_time)

                    msg["send_frame_id"] = int(frame_id)
                    msg["sent_wall_time"] = time.time()
                    msg['water_polygons']  = w_polys
                    msg['bridge_polygons'] = b_polys
                    msg['has_polygons']    = True
                else:
                    msg = prepare_frame_data(
                        width       = send_w,
                        height      = send_h,
                        curve       = curve,
                        frame_id    = frame_id,
                        timestamp   = timestamp,
                        water_mask  = wm,
                        bridge_mask = bm,
                        ships_data  = ships_data,
                    )

                try:
                    self._send_method(msg, frame_id)
                except TypeError:
                    self._send_method(msg)
                self.sent += 1

            except OSError as e:
                print(f"[AsyncNavSender] 网络异常(丢帧): {e}")
                self.dropped += 1
            except Exception as e:
                import traceback
                print(f"[AsyncNavSender] ✗ 未知异常 (帧{frame_id}): {e}")
                traceback.print_exc()
                self.dropped += 1


class AsyncYOLODetector:
    def __init__(self, model, imgsz=1024, conf=0.5, iou=0.45,
                 device="cuda:0", class_names=None, detect_every=7 ,allowed_class_ids=None):
        self.model = model
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_names = class_names or ['ship']
        self.detect_every = detect_every
        self.allowed_class_ids = allowed_class_ids
        self._allowed_class_ids_gpu = None
        self._allowed_class_ids_device = None
        self._input_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = -1

        self._result_lock = threading.Lock()
        self._latest_result: list = []
        self._result_frame_id: int = -1

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._new_frame_event = threading.Event()

        self.infer_fps = 0.0
        self.frames_submitted = 0
        self.frames_processed = 0
        self.frames_skipped = 0

        self.start()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._infer_loop, daemon=True,
                                        name="YOLO-Async")
        self._thread.start()
        print(f"[AsyncYOLO] ✓ 后台推理线程已启动 "
              f"(conf={self.conf}, iou={self.iou}, detect_every={self.detect_every})")

    def shutdown(self):
        self.stop()

    def stop(self):
        self._running = False
        self._new_frame_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        print(f"[AsyncYOLO] 已停止 | 处理={self.frames_processed} 跳过={self.frames_skipped}")

    def submit(self, frame: np.ndarray, frame_id: int):
        if frame_id % self.detect_every != 0:
            return

        with self._input_lock:
            old_id = self._latest_frame_id
            self._latest_frame = frame.copy()
            self._latest_frame_id = frame_id
            self.frames_submitted += 1

            if old_id != -1 and old_id != frame_id:
                self.frames_skipped += 1

        self._new_frame_event.set()

    def get_result(self) -> list:
        with self._result_lock:
            return self._latest_result

    def _infer_loop(self):
        last_time = time.time()
        frame_count = 0

        while self._running:
            self._new_frame_event.wait(timeout=0.1)
            self._new_frame_event.clear()

            if not self._running:
                break

            with self._input_lock:
                frame = self._latest_frame
                frame_id = self._latest_frame_id
                self._latest_frame = None

            if frame is None:
                continue

            t0 = time.time()
            # _infer_loop 里，约第 ~260 行
            try:
                results = self.model(
                    frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                    classes=self.allowed_class_ids,
                )
                # ✅ 改这里：_parse_to_ships → boxes_to_ships
                ships = self.boxes_to_ships(results[0], frame_id)
            except Exception as e:
                print(f"[AsyncYOLO] 推理异常: {e}")
                continue

            ships = _remove_duplicate_boxes(ships, iou_thresh=0.4)

            with self._result_lock:
                self._latest_result = ships
                self._result_frame_id = frame_id

            self.frames_processed += 1
            frame_count += 1

            now = time.time()
            elapsed = now - last_time
            if elapsed >= 1.0:
                self.infer_fps = frame_count / elapsed
                frame_count = 0
                last_time = now

    def boxes_to_ships(self, result, frame_id: int):
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return []

        raw = boxes.data

        # 普通检测：
        # [x1, y1, x2, y2, conf, cls]
        #
        # 跟踪模式：
        # [x1, y1, x2, y2, track_id, conf, cls]
        if boxes.is_track:
            det_gpu = torch.cat(
                (
                    raw[:, :4],
                    raw[:, -2:],
                ),
                dim=1,
            )
        else:
            det_gpu = raw[:, :6]

        # 在 GPU 上提前过滤类别，避免把无用检测框传回 CPU
        if self.allowed_class_ids is not None:
            if (
                self._allowed_class_ids_gpu is None
                or self._allowed_class_ids_device != det_gpu.device
            ):
                self._allowed_class_ids_gpu = torch.tensor(
                    sorted(self.allowed_class_ids),
                    device=det_gpu.device,
                    dtype=torch.int64,
                )
                self._allowed_class_ids_device = det_gpu.device

            class_ids_gpu = det_gpu[:, 5].to(torch.int64)

            keep = torch.isin(
                class_ids_gpu,
                self._allowed_class_ids_gpu,
            )

            det_gpu = det_gpu[keep]

        if det_gpu.shape[0] == 0:
            return []

        # 唯一一次 GPU → CPU
        # 这里会发生同步，但 Python/JSON 最终必须使用 CPU 数据
        det_cpu = (
            det_gpu
            .detach()
            .to(
                device="cpu",
                dtype=torch.float32,
            )
            .numpy()
        )

        ships = []

        for row in det_cpu:
            x1 = float(row[0])
            y1 = float(row[1])
            x2 = float(row[2])
            y2 = float(row[3])
            conf = float(row[4])
            cls_id = int(row[5])

            if cls_id < len(self.class_names):
                label = self.class_names[cls_id]
            else:
                label = result.names.get(
                    cls_id,
                    f"class_{cls_id}",
                )

            ships.append({
                "ship_id": len(ships),
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "center": [
                    (x1 + x2) * 0.5,
                    (y1 + y2) * 0.5,
                ],
                "conf": conf,
                "speed": 0.0,
                "bearing": 0.0,
                "distance": 0.0,
                "threat_level": 0,
                "hasSpeedBearing": False,
                "source_frame_id": frame_id,
            })

        return ships

    def __repr__(self):
        return (f"AsyncYOLODetector(infer_fps={self.infer_fps:.1f}, "
                f"processed={self.frames_processed}, "
                f"skipped={self.frames_skipped})")
    



def _remove_duplicate_boxes(ships, iou_thresh=0.4):
    if len(ships) <= 1:
        return ships
    ships_sorted = sorted(ships, key=lambda s: s['conf'], reverse=True)
    keep = []
    for s in ships_sorted:
        is_dup = any(_calc_iou(s['bbox'], k['bbox']) > iou_thresh for k in keep)
        if not is_dup:
            keep.append(s)
    for i, s in enumerate(keep):
        s['ship_id'] = i
    return keep


def _calc_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def should_draw_nav_band(ships, external_ships, max_distance):
    distances = []
    for s in ships:
        d = float(s.get("distance", -1))
        if d > 0:
            distances.append(d)
    for s in external_ships:
        d = float(s.get("bridge_pier_distance", -1))
        if d > 0:
            distances.append(d)

    if not distances:
        return True

    return min(distances) <= max_distance

def encode_mask_polygons(water_mask, bridge_mask,
                          epsilon_ratio=0.005,
                          min_area_px=200,
                          max_polys=4):
    h, w    = water_mask.shape
    epsilon = max(1.0, epsilon_ratio * max(w, h))

def _extract_polygons_json(water_mask, bridge_mask,
                            epsilon_ratio=0.005,
                            min_area_px=200,
                            max_polys=4):
    h, w = water_mask.shape
    epsilon = max(1.0, epsilon_ratio * max(w, h))

    def _extract(mask):
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        polys = []
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:max_polys]:
            if cv2.contourArea(cnt) < min_area_px:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
            pts = [
                [int(p[0]), int(p[1])]
                for p in approx.reshape(-1, 2)
            ]
            if len(pts) >= 3:
                polys.append(pts)
        return polys

    return _extract(water_mask), _extract(bridge_mask)
    

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

class PreprocessGPU:
    def __init__(self, target_h=432, target_w=768):
        self.target_h = target_h
        self.target_w = target_w

        self.buf = torch.empty(
            (1, 3, target_h, target_w),
            dtype=torch.float16,
            device="cuda",
        )

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float16,
            device="cuda",
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float16,
            device="cuda",
        ).view(1, 3, 1, 1)

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
        # 先缩放，减少后续颜色转换处理的像素数
        img = cv2.resize(
            frame_bgr,
            (self.target_w, self.target_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 只对640×384做颜色转换
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB,
        )

        t = torch.from_numpy(img).to(
            device="cuda",
            dtype=torch.float16,
            non_blocking=True,
        )

        t = t.permute(2, 0, 1).unsqueeze(0)
        t.mul_(1.0 / 255.0)
        t.sub_(self.mean).div_(self.std)

        self.buf.copy_(t)
        return self.buf
    
def get_raw_masks(frame_bgr, preprocess, inferencer):
    """
    获取原始分割结果，并用软概率做优先级调度
    替代原来的 get_raw_masks
    """
    x      = preprocess(frame_bgr)
    logits = inferencer.infer(x)
    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    
    # shape: [1, 3, H, W]  →  softmax 得到各类概率
    probs = torch.softmax(logits.float(), dim=1)[0]
    # probs[BRIDGE_CLASS_ID], probs[WATER_CLASS_ID], probs[BACKGROUND_CLASS_ID]
    
    p_bridge = probs[BRIDGE_CLASS_ID]   # [H, W]
    p_water  = probs[WATER_CLASS_ID]    # [H, W]
    
    # ── 核心逻辑 ──────────────────────────────────────────────
    # 1. 水面置信度足够高的像素，不允许被桥梁抢走
    #    即使 argmax 给了桥梁，也强制判为水面
    WATER_PROTECT_THRESH  = 0.30   # 水面概率超过此值就保护
    BRIDGE_COMMIT_THRESH  = 0.55   # 桥梁概率必须足够高才算桥梁
    
    # 水面掩码：水面概率 > 阈值
    water_gpu = (p_water >= WATER_PROTECT_THRESH).to(torch.uint8)
    
    # 桥梁掩码：桥梁概率高 且 该像素不是受保护的水面
    bridge_raw = (p_bridge >= BRIDGE_COMMIT_THRESH)
    bridge_gpu = (bridge_raw & (water_gpu == 0)).to(torch.uint8)
    # ──────────────────────────────────────────────────────────
    
    return water_gpu, bridge_gpu

class AsyncSegInferencer:
    def __init__(self, inferencer, preprocess,
                 mask_h: int = 432, mask_w: int = 768):
        self.inferencer = inferencer
        self.preprocess = preprocess
        self.mask_h = mask_h
        self.mask_w = mask_w

        self._water_smoother  = TemporalMaskSmootherGPU(alpha=0.55, threshold=0.40)
        self._bridge_smoother = TemporalMaskSmootherGPU(alpha=0.85, threshold=0.40)
        self._nav_smoother    = TemporalMaskSmootherGPU(alpha=0.70, threshold=0.40)
        self._prev_nav_gpu    = None

        self._input_lock     = threading.Lock()
        self._input_frame    = None
        self._input_frame_id = -1
        self._new_frame      = threading.Event()

        self._output_lock      = threading.Lock()
        self._water_np:  Optional[np.ndarray] = None
        self._bridge_np: Optional[np.ndarray] = None
        self._nav_np:    Optional[np.ndarray] = None
        self._curve_np: Optional[np.ndarray] = None
        self._output_frame_id = -1

        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                         name="AsyncSeg")
        self._thread.start()
        print("[AsyncSeg] ✓ 分割推理后台线程已启动")

    def submit(self, frame_bgr: np.ndarray, frame_id: int):
        with self._input_lock:
            self._input_frame    = frame_bgr
            self._input_frame_id = frame_id
        self._new_frame.set()

    def get_result(self):
        with self._output_lock:
            if self._water_np is None:
                return None
            return (self._water_np, self._bridge_np,
                    self._nav_np, self._output_frame_id,
                    self._curve_np)

    def shutdown(self):
        self._running = False
        self._new_frame.set()
        self._thread.join(timeout=3)

    def _loop(self):
        self._stream = torch.cuda.Stream()
        while self._running:
            self._new_frame.wait(timeout=0.1)
            self._new_frame.clear()
            if not self._running:
                break

            with self._input_lock:
                frame = self._input_frame
                fid   = self._input_frame_id
                self._input_frame = None

            if frame is None:
                continue

            try:
                with torch.cuda.stream(self._stream):
                    water_gpu, bridge_gpu = get_raw_masks(
                        frame, self.preprocess, self.inferencer)

                    water_gpu  = self._water_smoother.update(water_gpu)
                    bridge_gpu = self._bridge_smoother.update(bridge_gpu)

                    water_gpu  = refine_water_mask(water_gpu)
                    bridge_gpu = refine_bridge_mask(
                        bridge_gpu, water_gpu,
                        self.mask_h, self.mask_w,
                        margin_ratio=0.85,      # 恢复原来的值
                        max_height_ratio=0.65,
                        min_aspect_ratio=1.2,
                        max_area_ratio=0.5,
                    )

                    # 🔥 最终保证：桥梁掩码不能覆盖水面掩码
                    bridge_gpu = bridge_gpu & (water_gpu == 0)

                    nav_gpu = self._nav_smoother.update(water_gpu)
                    nav_gpu = stabilize_nav_mask_gpu(nav_gpu, self._prev_nav_gpu)
                    self._prev_nav_gpu = nav_gpu.clone()

                    # ★ GPU 上直接计算导航曲线，避免 CPU 运算
                    curve_gpu = build_navigation_curve_gpu(
                        nav_gpu,
                        water_mask=water_gpu,
                        n_rows=50,
                    )

                self._stream.synchronize()

                # ★ 一次性下载所有结果到 CPU
                masks_gpu = torch.stack(
                    [water_gpu, bridge_gpu, nav_gpu],
                    dim=0,
                ).contiguous()

                masks_np = masks_gpu.cpu().numpy()

                w_np = masks_np[0]
                b_np = masks_np[1]
                n_np = masks_np[2]

                curve_np = tensor_to_numpy_int32(curve_gpu)

                with self._output_lock:
                    self._water_np        = w_np
                    self._bridge_np       = b_np
                    self._nav_np          = n_np
                    self._curve_np        = curve_np
                    self._output_frame_id = fid

            except Exception as e:
                print(f"[AsyncSeg] 推理异常: {e}")

class FusionReceiver:
    HEADER_FMT = ">IHHIQII"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    MAGIC = 0x4655534E

    def __init__(
        self,
        host,
        port,
        reconnect_interval=10.0,
        connect_timeout=5.0,
        recv_timeout=2.0,
        data_ttl=0.5,
        max_payload_size=10 * 1024 * 1024,
    ):
        self.host = str(host)
        self.port = int(port)

        # 连接失败或连接断开后，等待10秒重新连接
        self.reconnect_interval = float(reconnect_interval)

        # 建立TCP连接超时时间
        self.connect_timeout = float(connect_timeout)

        # 已连接后，单次recv等待超时时间
        self.recv_timeout = float(recv_timeout)

        # 融合数据默认有效期
        self.data_ttl = float(data_ttl)

        # 防止异常包声明超大payload
        self.max_payload_size = int(max_payload_size)

        self._lock = threading.Lock()
        self._ships: list = []
        self._last_update_time = 0.0

        self._running = False
        self._connected = False

        self._sock = None
        self._thread = None

        # 用Event代替固定time.sleep，
        # stop时可以立即唤醒重连等待
        self._stop_event = threading.Event()

    def start(self):
        if self._running:
            print("[Fusion] 接收线程已经启动")
            return

        self._running = True
        self._connected = False
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._recv_loop,
            daemon=True,
            name="FusionRecv",
        )
        self._thread.start()

        print(
            f"[Fusion] 接收线程已启动，"
            f"断线后每 {self.reconnect_interval:.0f} 秒重连"
        )

    def stop(self):
        self._running = False
        self._connected = False
        self._stop_event.set()

        sock = self._sock
        self._sock = None

        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass

            try:
                sock.close()
            except OSError:
                pass

        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=3.0)

        self._clear_ships()

        print("[Fusion] 已停止")

    def get_ships(self, ttl=None):
        if ttl is None:
            ttl = self.data_ttl

        now = time.time()

        with self._lock:
            if (
                self._last_update_time <= 0.0
                or now - self._last_update_time > float(ttl)
            ):
                self._ships = []
                return []

            # 外层复制列表，避免调用者修改内部列表
            return list(self._ships)

    def is_connected(self):
        return self._connected

    def _clear_ships(self):
        with self._lock:
            self._ships = []
            self._last_update_time = 0.0

    def _close_socket(self):
        sock = self._sock
        self._sock = None

        if sock is None:
            return

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass

        try:
            sock.close()
        except OSError:
            pass

    def _wait_before_reconnect(self):

        if not self._running:
            return

        print(
            f"[Fusion] {self.reconnect_interval:.0f} 秒后重新尝试连接..."
        )

        self._stop_event.wait(
            timeout=self.reconnect_interval
        )

    def _recv_loop(self):
        while self._running:
            disconnected_reason = None

            try:
                print(
                    f"[Fusion] 正在连接 "
                    f"{self.host}:{self.port} ..."
                )

                sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                )

                # 减少小数据包发送延迟
                sock.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )

                # 可选TCP保活
                sock.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_KEEPALIVE,
                    1,
                )

                sock.settimeout(self.connect_timeout)
                self._sock = sock

                sock.connect((self.host, self.port))

                if not self._running:
                    break

                self._connected = True

                print(
                    f"[Fusion] ✅ 已连接 "
                    f"{self.host}:{self.port}"
                )

                sock.settimeout(self.recv_timeout)

                buf = bytearray()

                while self._running:
                    try:
                        data = sock.recv(65536)

                        # recv返回空字节表示对端正常关闭
                        if not data:
                            disconnected_reason = "对端已关闭连接"
                            break

                        buf.extend(data)
                        buf = self._parse_buffer(buf)

                    except socket.timeout:
                        # 暂时没有数据不代表断线
                        continue

                    except ConnectionResetError:
                        disconnected_reason = "连接被对端重置"
                        break

                    except ConnectionAbortedError:
                        disconnected_reason = "连接被中止"
                        break

                    except OSError as e:
                        if self._running:
                            disconnected_reason = f"接收异常：{e}"
                        break

            except socket.timeout:
                if self._running:
                    disconnected_reason = (
                        f"连接超时，超过 "
                        f"{self.connect_timeout:.1f} 秒"
                    )

            except ConnectionRefusedError as e:
                if self._running:
                    disconnected_reason = f"对端拒绝连接：{e}"

            except OSError as e:
                if self._running:
                    disconnected_reason = f"连接失败：{e}"

            except Exception as e:
                if self._running:
                    disconnected_reason = f"未知异常：{e}"

            finally:
                self._connected = False
                self._close_socket()
                self._clear_ships()

            if not self._running:
                break

            if disconnected_reason:
                print(f"[Fusion] ⚠️ {disconnected_reason}")

            self._wait_before_reconnect()

    def _parse_buffer(self, buf):
        if not isinstance(buf, bytearray):
            buf = bytearray(buf)

        magic_bytes = struct.pack(">I", self.MAGIC)

        while len(buf) >= self.HEADER_SIZE:
            # 包头不以MAGIC开头，查找下一个MAGIC
            if buf[:4] != magic_bytes:
                idx = buf.find(magic_bytes, 1)

                if idx >= 0:
                    del buf[:idx]
                else:
                    # MAGIC是4字节，保留最后3字节，
                    # 防止MAGIC刚好跨两个recv数据块
                    if len(buf) > 3:
                        del buf[:-3]
                    break

                continue

            try:
                (
                    magic,
                    version,
                    msg_type,
                    payload_len,
                    ts_us,
                    seq,
                    reserved,
                ) = struct.unpack_from(
                    self.HEADER_FMT,
                    buf,
                    0,
                )

            except struct.error:
                break

            # payload_len是无符号整数，无需判断小于0
            if payload_len > self.max_payload_size:
                print(
                    f"[Fusion] ⚠️ 非法payload长度："
                    f"{payload_len} 字节，跳过当前MAGIC"
                )

                # 移除当前MAGIC，继续寻找后续正常包
                del buf[:4]
                continue

            total_size = self.HEADER_SIZE + payload_len

            # 当前缓存还不是完整包
            if len(buf) < total_size:
                break

            payload = bytes(
                buf[self.HEADER_SIZE:total_size]
            )

            # 从缓存移除当前完整包
            del buf[:total_size]

            try:
                packet = _json.loads(
                    payload.decode("utf-8")
                )

                obstacles = packet.get("obstacles", [])

                if not isinstance(obstacles, list):
                    print(
                        "[Fusion] ⚠️ obstacles不是列表，"
                        "本包已忽略"
                    )
                    continue

                ships = self._convert(obstacles)

                with self._lock:
                    self._ships = ships
                    self._last_update_time = time.time()

            except UnicodeDecodeError as e:
                print(f"[Fusion] UTF-8解析失败：{e}")

            except _json.JSONDecodeError as e:
                print(f"[Fusion] JSON解析失败：{e}")

            except Exception as e:
                print(f"[Fusion] 数据处理失败：{e}")

        return buf

    def _convert(self, obstacles):
        ships = []

        for obs in obstacles:
            if not isinstance(obs, dict):
                continue

            try:
                x1 = int(obs.get("pixel_x1", -1))
                y1 = int(obs.get("pixel_y1", -1))
                x2 = int(obs.get("pixel_x2", -1))
                y2 = int(obs.get("pixel_y2", -1))

                if (
                    x1 < 0
                    or y1 < 0
                    or x2 <= x1
                    or y2 <= y1
                ):
                    continue

                north_vel = float(obs["north_vel"])
                east_vel = float(obs["east_vel"])

                speed = math.hypot(
                    north_vel,
                    east_vel,
                )

                bearing = (
                    math.degrees(
                        math.atan2(
                            east_vel,
                            north_vel,
                        )
                    )
                    % 360.0
                )

                distance = float(obs["distance"])
                yaw = float(obs["yaw"])

                bridge_pier_distance = float(
                    obs.get(
                        "bridge_pier_distance",
                        obs.get(
                            "pier_distance",
                            obs.get(
                                "bridge_distance",
                                -1.0,
                            ),
                        ),
                    )
                )

                class_id = int(
                    obs.get("class_id", 0)
                )

                confidence = float(
                    obs.get("confidence", 0.0)
                )

                numeric_values = (
                    north_vel,
                    east_vel,
                    distance,
                    yaw,
                    bridge_pier_distance,
                    confidence,
                )
                if not all(math.isfinite(value) for value in numeric_values):
                    continue

            except (
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                # 当前目标字段异常，只跳过这一条
                continue

            # 距离无效时，不应被误判为最高危险等级
            if distance > 0:
                if distance < 30:
                    threat_level = 2
                elif distance < 80:
                    threat_level = 1
                else:
                    threat_level = 0
            else:
                threat_level = 0

            ships.append({
                "ship_id": len(ships),
                "label": f"Target-{class_id}",
                "bbox": [x1, y1, x2, y2],
                "center": [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ],
                "conf": confidence,
                "speed": round(speed, 2),
                "bearing": round(bearing, 1),
                "distance": round(distance, 1),
                "north_vel": north_vel,
                "east_vel": east_vel,
                "yaw": yaw,
                "threat_level": threat_level,
                "hasSpeedBearing": True,
                "bridge_pier_distance": round(
                    bridge_pier_distance,
                    1,
                ),
            })

        return ships


def _ship_center(ship):
    if not isinstance(ship, dict):
        return None

    try:
        center = ship.get("center")
        if center is not None and len(center) >= 2:
            x, y = float(center[0]), float(center[1])
        else:
            bbox = ship.get("bbox")
            if bbox is None or len(bbox) < 4:
                return None
            x = (float(bbox[0]) + float(bbox[2])) * 0.5
            y = (float(bbox[1]) + float(bbox[3])) * 0.5
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _fusion_values(ship):
    if not isinstance(ship, dict):
        return None

    try:
        values = {
            key: float(ship[key])
            for key in ("north_vel", "east_vel", "distance", "yaw")
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def _optional_float(ship, key, default):
    try:
        value = float(ship.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def merge_ship_data(yolo_ships, external_ships, max_match_distance=80.0):
    if not yolo_ships:
        return []

    for ship in yolo_ships:
        ship["speed"] = 0.0
        ship["bearing"] = 0.0
        ship["distance"] = 0.0
        ship["north_vel"] = 0.0
        ship["east_vel"] = 0.0
        ship["yaw"] = 0.0
        ship["bridge_pier_distance"] = -1.0
        ship["hasSpeedBearing"] = False
        ship["has_fusion_data"] = False

    if not external_ships or not isinstance(external_ships, (list, tuple)):
        return yolo_ships

    try:
        match_limit = float(max_match_distance)
    except (TypeError, ValueError, OverflowError):
        return yolo_ships
    if not math.isfinite(match_limit) or match_limit < 0.0:
        return yolo_ships

    valid_external = []
    for external_ship in external_ships:
        center = _ship_center(external_ship)
        values = _fusion_values(external_ship)
        if center is not None and values is not None:
            valid_external.append((external_ship, center, values))

    candidates = []
    for yolo_index, yolo_ship in enumerate(yolo_ships):
        center = _ship_center(yolo_ship)
        if center is None:
            continue
        cx, cy = center

        for external_index, (_, external_center, _) in enumerate(valid_external):
            ex, ey = external_center
            distance = math.hypot(cx - ex, cy - ey)
            if distance <= match_limit:
                candidates.append((distance, yolo_index, external_index))

    matched_yolo = set()
    matched_external = set()
    for _, yolo_index, external_index in sorted(candidates):
        if yolo_index in matched_yolo or external_index in matched_external:
            continue

        yolo_ship = yolo_ships[yolo_index]
        external_ship, _, fusion_values = valid_external[external_index]
        yolo_ship["speed"] = _optional_float(external_ship, "speed", 0.0)
        yolo_ship["bearing"] = _optional_float(external_ship, "bearing", 0.0)
        yolo_ship.update(fusion_values)
        yolo_ship["bridge_pier_distance"] = _optional_float(
            external_ship, "bridge_pier_distance", -1.0
        )
        yolo_ship["hasSpeedBearing"] = True
        yolo_ship["has_fusion_data"] = True

        matched_yolo.add(yolo_index)
        matched_external.add(external_index)

    return yolo_ships


def refine_water_mask(mask_gpu: torch.Tensor) -> torch.Tensor:
    m = mask_gpu.float().unsqueeze(0).unsqueeze(0)
    m = F.max_pool2d(m, 5, stride=1, padding=2)
    m = -F.max_pool2d(-m, 5, stride=1, padding=2)
    m = -F.max_pool2d(-m, 7, stride=1, padding=3)
    m = F.max_pool2d(m, 7, stride=1, padding=3)
    return m.squeeze(0).squeeze(0).to(torch.uint8)


def refine_bridge_mask(bridge_gpu, water_gpu, mask_h, mask_w,
                       margin_ratio=0.85,
                       max_height_ratio=0.65,   # 原来是 0.20
                       min_aspect_ratio=1.2,
                       max_area_ratio=0.5):
    h, w = bridge_gpu.shape[:2]

    # 1. 空间约束（纯 GPU）
    # 找水面最高行：用 GPU 逐行求和
    water_row_sum = water_gpu.float().sum(dim=1)  # shape: (h,)
    water_has = (water_row_sum > 0)

    # 找第一个有水的行
    water_indices = torch.nonzero(water_has, as_tuple=False)
    if water_indices.numel() == 0:
        bridge_gpu.zero_()
        return bridge_gpu

    water_top = int(water_indices[0].item())
    water_bottom = int(water_indices[-1].item())

    margin = int(h * margin_ratio)
    bridge_top_limit = max(0, water_top - margin)

    bridge_gpu[:bridge_top_limit, :] = 0
    bridge_gpu[water_bottom:, :] = 0

    # 2. 形态学清理
    m = bridge_gpu.float().unsqueeze(0).unsqueeze(0)

    # 开运算：去小噪声
    m = -F.max_pool2d(-m, 5, stride=1, padding=2)  # erode
    m = F.max_pool2d(m, 5, stride=1, padding=2)    # dilate

    # 高度过滤：用垂直方向的投影
    # 如果某列的桥梁像素高度占比超过阈值，可能是楼体
    col_sum = m.squeeze().sum(dim=0)  # shape: (w,)
    max_bridge_h = int(h * max_height_ratio)
    bad_cols = (col_sum > max_bridge_h).float()

    # 如果大面积列都超高，说明是楼体，整块去掉
    bad_ratio = bad_cols.sum() / float(w)
    if bad_ratio > 0.3:
        # 超过 30% 的列都超高 → 整个区域可能是楼体
        bridge_gpu.zero_()
        return bridge_gpu

    # 宽高比过滤：用行列投影近似
    row_sum = m.squeeze().sum(dim=1)  # shape: (h,)
    active_rows = (row_sum > 0)
    active_cols_mask = (col_sum > 0)

    if active_rows.any() and active_cols_mask.any():
        row_indices = torch.nonzero(active_rows, as_tuple=False)
        col_indices = torch.nonzero(active_cols_mask, as_tuple=False)

        bbox_h = int(row_indices[-1].item() - row_indices[0].item()) + 1
        bbox_w = int(col_indices[-1].item() - col_indices[0].item()) + 1

        if bbox_h > 0:
            aspect = bbox_w / float(bbox_h)
            if aspect < min_aspect_ratio:
                bridge_gpu.zero_()
                return bridge_gpu

        # 面积过滤
        area_ratio = float(m.sum().item()) / (h * w)
        if area_ratio > max_area_ratio:
            bridge_gpu.zero_()
            return bridge_gpu

    result = m.squeeze().to(torch.uint8)
    return result


class CurveSmootherCPU:
    def __init__(self, alpha=0.18):
        self.alpha = float(alpha)
        self._ema = None

    def update(self, curve: np.ndarray) -> np.ndarray:
        if curve is None or len(curve) == 0:
            return np.empty((0, 2), dtype=np.int32)

        c = curve.astype(np.float32)

        if self._ema is None or self._ema.shape != c.shape:
            self._ema = c.copy()
        else:
            self._ema = self.alpha * self._ema + (1.0 - self.alpha) * c

        return np.round(self._ema).astype(np.int32)


def build_navigation_curve(nav_mask: np.ndarray,
                           water_mask: np.ndarray = None,
                           n_rows: int = 50,
                           min_row_pixels: int = 20,
                           smooth_k: int = 13) -> np.ndarray:
    if nav_mask is None:
        return np.empty((0, 2), dtype=np.int32)

    mask = nav_mask > 0
    h, w = mask.shape

    row_counts = mask.sum(axis=1)
    valid_rows = np.flatnonzero(row_counts >= 1)

    if valid_rows.size < 10:
        return np.empty((0, 2), dtype=np.int32)

    #顶部取水面 mask 的最高行，而非 nav_mask 的最高行
    if water_mask is not None:
        water_rows = np.flatnonzero((water_mask > 0).any(axis=1))
        if water_rows.size > 0:
            water_top = max(0, int(water_rows[0]) - 1)
        else:
            water_top = max(0, int(valid_rows[0]) - 1)
    else:
        water_top = max(0, int(valid_rows[0]) - 1)

    top_row = water_top
    bot_row = h - 1

    span = bot_row - top_row
    top_row = top_row + int(span * 0.20)

    # 均匀采样 n_rows 行（从底到顶）
    ys = np.linspace(bot_row, top_row, n_rows, dtype=np.int32)

    rows = mask[ys].astype(np.float32)
    counts = rows.sum(axis=1)

    xs = np.arange(w, dtype=np.float32)
    cx = rows @ xs / np.maximum(counts, 1.0)

    valid = counts >= min_row_pixels

    # 无效行沿用上一行中心
    prev = w * 0.5
    for i in range(n_rows):
        if valid[i]:
            prev = cx[i]
        else:
            cx[i] = prev

    # 限制跳变，防止曲线抖动
    max_step = max(8, int(w * 0.035))
    for i in range(1, n_rows):
        cx[i] = np.clip(cx[i], cx[i - 1] - max_step, cx[i - 1] + max_step)

    # 轻量 1D 平滑
    if smooth_k > 1 and n_rows >= smooth_k:
        if smooth_k % 2 == 0:
            smooth_k += 1
        pad = smooth_k // 2
        kernel = np.ones(smooth_k, dtype=np.float32) / smooth_k
        cx = np.convolve(np.pad(cx, (pad, pad), mode="edge"),
                         kernel,
                         mode="valid")

    result = np.empty((n_rows, 2), dtype=np.int32)
    result[:, 0] = np.clip(cx, 0, w - 1).astype(np.int32)
    result[:, 1] = np.clip(ys, 0, h - 1).astype(np.int32)

    return result

# ═══════════════════════════════════════════════════════════════
# GPU 版本几何计算（PyTorch CUDA tensors，全部在 GPU 上完成）
# ═══════════════════════════════════════════════════════════════

def build_navigation_curve_gpu(
    nav_mask: torch.Tensor,
    water_mask: Optional[torch.Tensor] = None,
    n_rows: int = 50,
    min_row_pixels: int = 20,
    smooth_k: int = 13,
) -> torch.Tensor:
    """GPU 版本：从导航 mask 计算导航曲线"""
    if nav_mask is None or not nav_mask.any():
        return torch.empty((0, 2), dtype=torch.int32, device=nav_mask.device)

    device = nav_mask.device
    mask = nav_mask > 0
    h, w = mask.shape

    row_counts = mask.sum(dim=1)
    valid_rows = torch.where(row_counts >= 1)[0]

    if valid_rows.numel() < 10:
        return torch.empty((0, 2), dtype=torch.int32, device=device)

    if water_mask is not None and water_mask.any():
        water_any = (water_mask > 0).any(dim=1)
        water_rows = torch.where(water_any)[0]
        if water_rows.numel() > 0:
            water_top = max(0, int(water_rows[0].item()) - 1)
        else:
            water_top = max(0, int(valid_rows[0].item()) - 1)
    else:
        water_top = max(0, int(valid_rows[0].item()) - 1)

    top_row = water_top
    bot_row = h - 1
    span = bot_row - top_row
    top_row = top_row + int(span * 0.20)

    ys = torch.linspace(bot_row, top_row, n_rows, dtype=torch.int32, device=device)
    ys = ys.clamp(0, h - 1)

    rows = mask[ys].float()
    counts = rows.sum(dim=1)

    xs = torch.arange(w, dtype=torch.float32, device=device)
    cx = (rows @ xs) / counts.clamp(min=1.0)

    valid = counts >= min_row_pixels

    prev = w * 0.5
    for i in range(n_rows):
        if valid[i]:
            prev = cx[i].item()
        else:
            cx[i] = prev

    max_step = max(8, int(w * 0.035))
    for i in range(1, n_rows):
        cx[i] = cx[i].clamp(cx[i - 1] - max_step, cx[i - 1] + max_step)

    if smooth_k > 1 and n_rows >= smooth_k:
        if smooth_k % 2 == 0:
            smooth_k += 1
        kernel = torch.ones(1, 1, smooth_k, dtype=torch.float32, device=device) / smooth_k
        cx_padded = F.pad(cx.view(1, 1, -1), (smooth_k // 2, smooth_k // 2), mode='replicate')
        cx = F.conv1d(cx_padded, kernel).view(-1)

    result = torch.empty((n_rows, 2), dtype=torch.int32, device=device)
    result[:, 0] = cx.clamp(0, w - 1).to(torch.int32)
    result[:, 1] = ys.clamp(0, h - 1).to(torch.int32)

    return result


def tensor_to_numpy_int32(t: torch.Tensor) -> np.ndarray:
    """将 GPU tensor 转为 CPU numpy int32"""
    if t is None or t.numel() == 0:
        return np.empty((0, 2), dtype=np.int32)
    return t.cpu().numpy().astype(np.int32)


def stabilize_nav_mask_gpu(nav_mask_gpu: torch.Tensor,
                              prev_nav_mask_gpu: Optional[torch.Tensor],
                              min_area_ratio: float = 0.003) -> torch.Tensor:
    """stabilize v4：用 soft gate 替代条件分支，零同步。"""
    h, w = nav_mask_gpu.shape
    min_a = float(max(1000, int(h * w * min_area_ratio)))

    area = nav_mask_gpu.float().sum()  # 标量 tensor，不调 .item()

    if prev_nav_mask_gpu is not None:
        #  soft blend：面积太小时平滑过渡到前一帧
        # gate ∈ [0, 1]：面积充足时≈1（用当前帧），面积不足时≈0（用前帧）
        gate = torch.sigmoid((area - min_a) * 0.005)
        nav = nav_mask_gpu.float() * gate + prev_nav_mask_gpu.float() * (1.0 - gate)
    else:
        nav = nav_mask_gpu.float()

    # 闭运算
    nav = nav.unsqueeze(0).unsqueeze(0)
    nav = F.max_pool2d(nav, 5, stride=1, padding=2)
    nav = -F.max_pool2d(-nav, 5, stride=1, padding=2)
    nav = nav.squeeze()

    return (nav > 0.5).to(torch.uint8)


def row_mask_center(mask, y, default_x):
    h, _  = mask.shape
    y     = int(np.clip(y, 0, h - 1))
    xs    = np.where(mask[y, :] == 1)[0]
    if xs.size < 20:
        return int(default_x)
    return int((np.percentile(xs, 18) + np.percentile(xs, 82)) / 2)


def smooth_curve(points, k=6, iterations=2):
    if points is None or len(points) < 2 * k + 1:
        return points.astype(np.int32)
    out = points.astype(np.float32)
    for _ in range(iterations):
        new = out.copy()
        for i in range(1, len(points) - 1):
            s, e  = max(0, i - k), min(len(points), i + k + 1)
            new[i] = np.mean(out[s:e], axis=0)
        out = new
    return out.astype(np.int32)


def scale_curve_xy(curve: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if curve is None or len(curve) == 0:
        return np.empty((0, 2), dtype=np.int32)
    out = curve.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out.astype(np.int32)


def _scale_ships_for_display(ships: list, sx: float, sy: float) -> list:
    if not ships or (sx == 1.0 and sy == 1.0):
        return ships
    result = []
    for s in ships:
        ss = dict(s)
        x1, y1, x2, y2 = s['bbox']
        ss['bbox']   = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        ss['center'] = [s['center'][0] * sx, s['center'][1] * sy]
        result.append(ss)
    return result

def _scale_ships_for_send(ships: list, sx: float, sy: float,
                          dst_w: int = None, dst_h: int = None) -> list:
    if not ships:
        return []

    result = []

    for s in ships:
        if 'bbox' not in s or s['bbox'] is None or len(s['bbox']) < 4:
            continue

        ss = dict(s)

        x1, y1, x2, y2 = [float(v) for v in s['bbox']]

        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        if dst_w is not None:
            x1 = float(np.clip(x1, 0, dst_w - 1))
            x2 = float(np.clip(x2, 0, dst_w - 1))

        if dst_h is not None:
            y1 = float(np.clip(y1, 0, dst_h - 1))
            y2 = float(np.clip(y2, 0, dst_h - 1))

        ss['bbox'] = [x1, y1, x2, y2]

        if 'center' in s and s['center'] is not None and len(s['center']) >= 2:
            cx = float(s['center'][0]) * sx
            cy = float(s['center'][1]) * sy

            if dst_w is not None:
                cx = float(np.clip(cx, 0, dst_w - 1))
            if dst_h is not None:
                cy = float(np.clip(cy, 0, dst_h - 1))

            ss['center'] = [cx, cy]
        else:
            ss['center'] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        result.append(ss)

    return result

def overlay_masks(frame_bgr, water_mask, bridge_mask, cfg,
                  curve=None, frame_idx=0, fps_text="",
                  draw_curve=True, draw_masks=True, ships=None):
    out = frame_bgr.copy()

    if ships:
        h_img, w_img = out.shape[:2]
        for s in ships:
            x1, y1, x2, y2 = [int(v) for v in s['bbox']]
            spd    = s.get('speed',    0)
            brg    = s.get('bearing',  0)
            dist   = s.get('distance', 0)
            threat = s.get('threat_level', 0)

            if threat >= 2:
                box_color, tag = (0, 0, 255),   "DANGER"
            elif threat >= 1:
                box_color, tag = (0, 165, 255),  "CAUTION"
            else:
                box_color, tag = (0, 255, 0),    "SAFE"

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

            lines  = [f"{tag}  DIST:{dist:.1f}m",
                      f"SPD:{spd:.1f}m/s  BRG:{brg:.0f}"]
            line_h = 22
            bg_h   = line_h * len(lines) + 6
            bg_y   = max(0, y1 - bg_h)

            dx1 = max(0, min(x1, w_img - 1))
            dx2 = max(dx1 + 1, min(x2, w_img))
            dy1 = max(0, min(bg_y, h_img - 1))
            dy2 = max(dy1 + 1, min(bg_y + bg_h, h_img))

            if dy2 > dy1 and dx2 > dx1:
                roi_copy = out[dy1:dy2, dx1:dx2].copy()
                cv2.rectangle(out, (dx1, dy1), (dx2, dy2), box_color, -1)
                blended = cv2.addWeighted(
                    out[dy1:dy2, dx1:dx2], 0.7, roi_copy, 0.3, 0)
                if blended is not None:
                    out[dy1:dy2, dx1:dx2] = blended

            for li, line in enumerate(lines):
                ty = bg_y + 16 + li * line_h
                if 0 < ty < h_img and 0 < x1 < w_img:
                    cv2.putText(out, line, (x1 + 4, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(out, line, (x1 + 4, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                (0, 0, 0), 1, cv2.LINE_AA)

            if spd > 0.5:
                cx  = (x1 + x2) // 2
                cy  = (y1 + y2) // 2
                arrow_len = min(60, int(spd * 8))
                rad = math.radians(brg)
                ex  = int(cx + arrow_len * math.sin(rad))
                ey  = int(cy - arrow_len * math.cos(rad))
                cv2.arrowedLine(out, (cx, cy), (ex, ey),
                                (0, 255, 255), 2, tipLength=0.3)

    if draw_masks:
        def overlay_fast(mask, color_bgr, alpha):
            if mask is None or not mask.any():
                return
            color_layer = np.zeros_like(out, dtype=np.uint8)
            color_layer[mask == 1] = color_bgr
            cv2.addWeighted(color_layer, alpha, out, 1.0, 0, dst=out)
        overlay_fast(water_mask,  cfg.water_color,  cfg.water_alpha)
        overlay_fast(bridge_mask, cfg.bridge_color, cfg.bridge_alpha)

    if draw_curve and curve is not None and len(curve) >= 2:
        pts = curve.reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, (0, 255, 128), 2, cv2.LINE_AA)

    if fps_text:
        cv2.putText(out, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
    return out

#  TCP 服务启动
def start_tcp_server():
    global nav_server
    nav_server = NavigationDataServer(
        host="0.0.0.0", port=8765, video_port=8766)
    nav_server.start()


def process_video(input_video, output_video,
                  seg_weights,
                  device_str, enable_streaming=True,
                  codec="mp4v",
                  enable_video_stream=True,
                  video_jpeg_quality=85,
                  video_stream_scale=0.5,
                  video_stream_interval=1,
                  infer_every_n=3,
                  show_window=False,
                  show_scale=0.5,
                  blur_kernel=5,
                  use_jpeg_stream=True,
                  draw_curve_on_python=False,
                  draw_masks_on_python=False,
                  ship_host="127.0.0.1",
                  ship_port=55103,
                  yolo_send_port=9000,
                  save_video=False,
                  nav_send_every=3,
                  mask_send_scale=0.5,
                  nav_draw_distance=100.0):

    cfg = RenderConfig()

    MASK_H, MASK_W = 432, 768

    need_render = save_video
    show_scale  = float(np.clip(show_scale, 0.25, 1.0))

    print(f"\n推理尺寸  : {MASK_W}×{MASK_H}")
    print(f"掩码策略  : 每 {infer_every_n} 帧推理一次，其余复用")
    print(f"本地渲染  : {'开' if need_render else '关(最高性能)'}")
    print(f"Python绘制: 曲线={'开' if draw_curve_on_python else '关'} "
          f"掩码={'开' if draw_masks_on_python else '关'}")
    print(f"视频流    : 文件 I/O → Godot ShmVideoReader (RGBA 640×360)")

    # TCP 服务
    if enable_streaming:
        Thread(target=start_tcp_server, daemon=True).start()
        time.sleep(1.5)
        print(f"TCP 服务器已启动:")
        print(f"  导航数据: tcp://0.0.0.0:8765")
        print(f"  视频帧  : 文件 I/O (RGBA 640×360 → Godot ShmVideoReader)")

    # 单预处理器
    preprocess = PreprocessGPU(target_h=MASK_H, target_w=MASK_W)

    # TRT 推理器
    inferencer = TRTInferencer(
        engine_path=seg_weights,
        input_h=MASK_H,
        input_w=MASK_W,
    )

    #YOLO 异步检测器
    SHIP_CLASS_IDS = [0]

    yolo_detector = AsyncYOLODetector(
        model=SHIP_MODEL,
        imgsz=SHIP_INPUT_SIZE,
        conf=SHIP_CONF,
        iou=SHIP_IOU,
        device="cuda:0",
        class_names=SHIP_CLASS_NAMES,
        detect_every=5,
        allowed_class_ids=SHIP_CLASS_IDS,
    )


    # YoloSender
    yolo_sender = None
    if yolo_send_port > 0:
        yolo_sender = YoloSender(host='0.0.0.0', port=yolo_send_port)
        yolo_sender.start()
        print(f"[YoloSender] YOLO结果发送端口: {yolo_send_port}")

    #  FusionReceiver 
    fusion_receiver = None
    if ship_host and ship_host.strip():
        fusion_receiver = FusionReceiver(ship_host, ship_port)
        fusion_receiver.start()

    # 视频源 
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_video}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #视频写出
    writer       = None
    async_writer = None
    if save_video:
        fourcc       = cv2.VideoWriter_fourcc(*codec)
        writer       = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        async_writer = AsyncVideoWriter(writer, maxsize=6)
        print(f"视频保存: 开 → {output_video}")
    else:
        print("视频保存: 关")

    tcp_video_streamer = None
    file_video_sender = None
    if enable_video_stream:
        # Windows / 非 Jetson：通过文件 I/O 发送 RGBA 帧给 Godot ShmVideoReader
        # Godot 的 ShmVideoReader 在无 GDExtension 时回退到 FileAccess 模式
        file_video_sender = FileVideoSender()
        if not file_video_sender.start():
            file_video_sender = None
            print("⚠️ FileVideoSender 启动失败，视频将不发送到 Godot")

    # 导航数据异步发送器
    nav_sender = None
    if enable_streaming:
        nav_sender = AsyncNavDataSender(
            nav_server      = nav_server,
            mask_send_scale = 1.0,
        )

    # 预热
    dummy = torch.randn(1, 3, MASK_H, MASK_W, dtype=torch.float16, device='cuda')
    for _ in range(3):
        inferencer.infer(dummy)
    print("✓ TRT 预热完成")

    # 异步分割推理器
    async_seg = AsyncSegInferencer(
        inferencer = inferencer,
        preprocess = preprocess,
        mask_h     = MASK_H,
        mask_w     = MASK_W,
    )

    # 曲线平滑器（轻量 CPU 操作，留在主循环）
    curve_smoother = CurveSmootherCPU(alpha=0.18)

    # CPU 端缓存（每帧使用的"当前可用结果"）
    water_small_np      = None
    bridge_small_np     = None
    nav_small_np        = None
    curve_small         = np.empty((0, 2), dtype=np.int32)

    cached_curve        = np.empty((0, 2), dtype=np.int32)
    cached_curve_send   = np.empty((0, 2), dtype=np.int32)
    cached_water_send   = None
    cached_bridge_send  = None
    cached_water_large  = None
    cached_bridge_large = None

    send_scale = float(np.clip(mask_send_scale, 0.2, 1.0))
    send_w     = max(1, int(width  * send_scale))
    send_h     = max(1, int(height * send_scale))

    frame_idx   = 0
    frame_times = []
    last_processed_seg_fid = -1
    last_sent_nav_seg_fid = -1

    print(f"视频: {width}×{height} @ {fps:.1f}fps, 共 {total_frames} 帧")
    print(f"[AsyncSeg] ✓ 推理完全异步 | 主循环零等待")
    print(f"开始处理...\n")
    t_start = time.time()

    try:
        while True:
            t_frame = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            video_time = float(frame_idx - 1) / max(1e-6, float(fps))

            # ★ 步骤 0：视频帧发送到 Godot（通过文件 I/O，ShmVideoReader 读取）
            if file_video_sender:
                file_video_sender.send(frame)

            # ★ 步骤 1：提交推理（非阻塞，丢给后台线程）
            if frame_idx % infer_every_n == 0:
                async_seg.submit(frame, frame_idx)

            yolo_detector.submit(frame, frame_idx)

            # ★ 步骤 2：取最新分割结果（非阻塞，有新的就更新缓存）
            seg_result = async_seg.get_result()

            if seg_result is not None:
                new_water_np, new_bridge_np, new_nav_np, seg_fid, new_curve_np = seg_result

                # 只有拿到新分割结果时，才执行曲线构建和mask缩放
                if seg_fid != last_processed_seg_fid:
                    last_processed_seg_fid = seg_fid

                    water_small_np = new_water_np
                    bridge_small_np = new_bridge_np
                    nav_small_np = new_nav_np

                    # ★ 曲线已在 GPU 上计算完成，只需 CPU 端平滑
                    curve_small = new_curve_np
                    curve_small = curve_smoother.update(curve_small)

                    sx_up = width / float(MASK_W)
                    sy_up = height / float(MASK_H)

                    if curve_small is not None and len(curve_small) > 0:
                        cached_curve = scale_curve_xy(
                            curve_small,
                            sx_up,
                            sy_up,
                        )
                        cached_curve[:, 0] = np.clip(
                            cached_curve[:, 0], 0, width - 1
                        )
                        cached_curve[:, 1] = np.clip(
                            cached_curve[:, 1], 0, height - 1
                        )
                    else:
                        cached_curve = np.empty(
                            (0, 2),
                            dtype=np.int32,
                        )

                    cached_water_send = cv2.resize(
                        water_small_np,
                        (send_w, send_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    cached_bridge_send = cv2.resize(
                        bridge_small_np,
                        (send_w, send_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    sx_send = send_w / float(MASK_W)
                    sy_send = send_h / float(MASK_H)

                    if curve_small is not None and len(curve_small) > 0:
                        cached_curve_send = scale_curve_xy(
                            curve_small,
                            sx_send,
                            sy_send,
                        )
                        cached_curve_send[:, 0] = np.clip(
                            cached_curve_send[:, 0],
                            0,
                            send_w - 1,
                        )
                        cached_curve_send[:, 1] = np.clip(
                            cached_curve_send[:, 1],
                            0,
                            send_h - 1,
                        )
                    else:
                        cached_curve_send = np.empty(
                            (0, 2),
                            dtype=np.int32,
                        )

                    if need_render and save_video:
                        cached_water_large = cv2.resize(
                            water_small_np,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                        cached_bridge_large = cv2.resize(
                            bridge_small_np,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )

            # ★ 步骤 3：船只数据汇总
            yolo_ships     = yolo_detector.get_result()
            external_ships = fusion_receiver.get_ships() if fusion_receiver else []
            ships_final    = merge_ship_data(yolo_ships, external_ships)
            # ★ 步骤 4：YoloSender 推送
            if yolo_sender and yolo_ships:
                yolo_sender.send_ships(yolo_ships, frame_idx, width, height)

            # 步骤 5：TCP 导航数据发送
            # 只在新的分割帧出现后发送一次，避免重复提取同一张 mask 的 polygon
            has_new_nav_mask = (
                last_processed_seg_fid > 0
                and last_processed_seg_fid != last_sent_nav_seg_fid
            )

            if (nav_sender
                    and cached_water_send is not None
                    and cached_bridge_send is not None
                    and has_new_nav_mask):

                mask_frame_id = int(last_processed_seg_fid)
                mask_video_time = float(mask_frame_id - 1) / max(1e-6, float(fps))

                nav_sender.submit(
                    width       = send_w,
                    height      = send_h,
                    curve = cached_curve_send,
                    frame_id    = frame_idx,
                    timestamp   = video_time,

                    water_mask  = cached_water_send,
                    bridge_mask = cached_bridge_send,
                    ships_data  = ships_final,
                    ship_src_w  = width,
                    ship_src_h  = height,

                    video_time      = video_time,
                    sync_mode       = "file_time",
                    source_fps      = fps,
                    mask_frame_id   = mask_frame_id,
                    mask_video_time = mask_video_time,
                )

                last_sent_nav_seg_fid = last_processed_seg_fid

            # ★ 步骤 6：本地渲染（仅用于保存视频，不再弹窗显示）
            if need_render:
                elapsed  = time.time() - t_start
                cur_fps  = frame_idx / elapsed if elapsed > 0 else 0.0
                fps_text = (
                    f"FPS:{cur_fps:.1f} Frame:{frame_idx}/{total_frames} "
                    f"Ships:{len(ships_final)}"
                )

                if save_video and async_writer:
                    wm_disp = (cached_water_large  if cached_water_large  is not None
                               else np.zeros((height, width), dtype=np.uint8))
                    bm_disp = (cached_bridge_large if cached_bridge_large is not None
                               else np.zeros((height, width), dtype=np.uint8))

                    vis_full = overlay_masks(
                        frame,
                        water_mask  = wm_disp,
                        bridge_mask = bm_disp,
                        cfg         = cfg,
                        curve       = cached_curve,
                        frame_idx   = frame_idx,
                        fps_text    = fps_text,
                        draw_curve  = draw_curve_on_python,
                        draw_masks  = draw_masks_on_python,
                        ships       = ships_final,
                    )
                    async_writer.submit(vis_full)

            # ★ 步骤 7：帧耗时统计 + 帧率控制
            t_cost = time.time() - t_frame
            frame_times.append(t_cost)
            if len(frame_times) > 120:
                frame_times.pop(0)

            if frame_idx % 100 == 0 and frame_idx > 0:
                avg_ms   = np.mean(frame_times) * 1000
                real_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                print(
                    f"  帧 {frame_idx:5d}/{total_frames} | "
                    f"avg {avg_ms:.1f}ms ({real_fps:.1f}fps) | "
                    f"ships={len(ships_final)}"
                )

            frame_interval     = 1.0 / fps
            elapsed_this_frame = time.time() - t_frame
            sleep_time         = frame_interval - elapsed_this_frame
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")
    finally:
        print("\n正在释放资源...")
        cap.release()
        async_seg.shutdown()

        if async_writer:
            async_writer.shutdown()
        if writer:
            writer.release()
        if file_video_sender:
            file_video_sender.stop()
        if nav_sender:
            nav_sender.shutdown()
        if yolo_detector:
            yolo_detector.shutdown()
        if yolo_sender:
            yolo_sender.stop()
        if fusion_receiver:
            fusion_receiver.stop()
        if inferencer:
            inferencer.shutdown()

        elapsed = time.time() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\n处理完成:")
        print(f"  总帧数  : {frame_idx}")
        print(f"  总耗时  : {elapsed:.1f}s")
        print(f"  平均FPS : {avg_fps:.1f}")
        if frame_times:
            print(f"  帧耗时  : avg={np.mean(frame_times)*1000:.1f}ms "
                  f"min={np.min(frame_times)*1000:.1f}ms "
                  f"max={np.max(frame_times)*1000:.1f}ms")


def parse_args():
    p = argparse.ArgumentParser(description="Navigation")
    p.add_argument("--input",          default="bridge10.mp4")
    p.add_argument("--output",         default="output_nav.mp4")
    p.add_argument("--seg",            default="best1.trt")          # 单模型权重
    p.add_argument("--device",         default="cuda")
    p.add_argument("--no-stream",      action="store_true")
    p.add_argument("--no-video-stream",action="store_true")
    p.add_argument("--no-window",      action="store_true", default=True,
                   help="已弃用：本地窗口显示已移除，视频通过 TCP 发送到 Godot")
    p.add_argument("--save-video",     action="store_true")
    p.add_argument("--use-jpeg",       action="store_true", default=True)
    p.add_argument("--no-jpeg",        action="store_true")
    p.add_argument("--nav-send-every", type=int,   default=2)
    p.add_argument("--mask-send-scale",type=float, default=0.5)
    p.add_argument("--show-scale",     type=float, default=0.5)
    p.add_argument("--infer-every",    type=int,   default=5)
    p.add_argument("--jpeg-quality",   type=int,   default=85)
    p.add_argument("--nav-draw-distance", type=float, default=100.0,
                   help="距离超过该值时不绘制导航带")
    p.add_argument("--stream-scale",   type=float, default=0.5)
    p.add_argument("--stream-interval",type=int,   default=1)
    p.add_argument("--ship-host",      default="127.0.0.1")
    p.add_argument("--ship-port",      type=int,   default=55103)
    p.add_argument("--yolo-port",      type=int,   default=9000)
    p.add_argument("--draw-curve",     action="store_true")
    p.add_argument("--draw-masks",     action="store_true")
    p.add_argument("--codec",          default="mp4v")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        input_video          = args.input,
        output_video         = args.output,
        seg_weights          = args.seg,
        device_str           = args.device,
        enable_streaming     = not args.no_stream,
        codec                = args.codec,
        mask_send_scale      = args.mask_send_scale,
        enable_video_stream  = not args.no_video_stream,
        video_jpeg_quality   = args.jpeg_quality,
        video_stream_scale   = args.stream_scale,
        video_stream_interval= args.stream_interval,
        infer_every_n        = args.infer_every,
        show_window          = False,  # 本地窗口显示已移除，用 Godot 显示
        show_scale           = args.show_scale,
        draw_curve_on_python = args.draw_curve,
        draw_masks_on_python = args.draw_masks,
        ship_host            = args.ship_host,
        ship_port            = args.ship_port,
        yolo_send_port       = args.yolo_port,
        save_video           = args.save_video,
        nav_draw_distance    = args.nav_draw_distance,
    )
