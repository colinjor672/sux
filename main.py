import os

# Keep CPU helper libraries from creating a worker pool for small GPU-bound
# operations. This must be set before importing OpenCV/PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import warnings
import subprocess
import time
import signal
import sys
import stat
import math

import cv2

from env_setup import setup_environment, setup_onnxruntime_cuda
from config import SHIP_ENGINE_PATH
from video_pipeline import process_video
from godot_launcher import start_godot, stop_godot


def parse_args():
    p = argparse.ArgumentParser(description="Navigation")
    #cap = cv2.VideoCapture(input_video)  # input_video 是 "video.mp4" 文件
    p.add_argument("--input", default="rtsp://admin:gzsia2011@192.168.1.164:554/Streaming/Channels/101")#rtsp://admin:gzsia2011@192.168.1.164:554/Streaming/Channels/101
    p.add_argument("--output", default="output_nav.mp4")
    p.add_argument("--seg",    default="best_fp16.engine")
    p.add_argument("--device", default="cuda")
    p.add_argument("--godot",
                   default="/home/jetson/visualization/Visualization1.arm64",
                   help="Godot 导出的可执行程序路径")

    p.add_argument("--no-godot",        action="store_true", help="不自动启动 Godot")
    p.add_argument("--no-EasyDarwin",   action="store_true", help="不自动启动 EasyDarwin")
    p.add_argument("--no-stream",       action="store_true")
    p.add_argument("--no-video-stream", action="store_true")
    window = p.add_mutually_exclusive_group()
    window.add_argument("--window", dest="show_window", action="store_true",
                        help="启用 Python 本地预览渲染")
    window.add_argument("--no-window", dest="show_window", action="store_false",
                        help="关闭 Python 本地预览渲染（默认）")
    p.set_defaults(show_window=False)
    p.add_argument("--save-video",      action="store_true")

    p.add_argument("--nav-send-every",    type=int,   default=2)
    p.add_argument("--mask-send-scale",   type=float, default=0.5)
    p.add_argument("--show-scale",        type=float, default=0.5)
    p.add_argument("--playback-fps", type=float, default=0.0,
                   help="处理帧率上限；0 表示本地文件使用源帧率、RTSP 不限帧")
    p.add_argument("--cpu-threads", type=int, default=1,
                   help="OpenCV/PyTorch CPU 工作线程数（GPU 推理建议为 1）")
    p.add_argument("--nav-draw-distance", type=float, default=30.0,
                   help="GNSS 导航带前视距离，米")
    p.add_argument("--nav-scale", type=float, default=0.5,
                   help="投映后导航带的等比缩放，(0,1]；1.0 为原尺寸")
    p.add_argument("--nav-projection-hz", type=float, default=10.0,
                   help="GNSS projection updates per second; other frames reuse the cached curve")
    p.add_argument("--calib-ini", default="calibration.ini",
                   help="相机-LiDAR 标定文件")
    p.add_argument("--lidar-height", type=float, default=1.6,
                   help="LiDAR 距水面高度，米")
    p.add_argument("--nav-heading-offset", type=float, default=0.0,
                   help="航向安装修正角，度")
    p.add_argument("--mqtt-host", default="127.0.0.1",
                   help="本地 MQTT Broker 地址")
    p.add_argument("--prediction-topic", default="v1/11/prediction/result",
                   help="避障服务路径结果 MQTT Topic")
    p.add_argument("--gnss-topic", default="v1/11/sensor/gnss/gnss_01",
                   help="实时船只经纬度 MQTT Topic")
    p.add_argument("--mqtt-username", default=os.environ.get("MQTT_USERNAME"))
    p.add_argument("--mqtt-password", default=os.environ.get("MQTT_PASSWORD"))

    p.add_argument("--ship-host", default="127.0.0.1")
    p.add_argument("--ship-port", type=int, default=55103)
    p.add_argument("--yolo-port", type=int, default=9000)

    p.add_argument("--draw-curve", action="store_true")
    p.add_argument("--draw-masks", action="store_true")
    p.add_argument("--codec",      default="mp4v")

    p.add_argument(
        "--EasyDarwin",
        default="/home/jetson/programs/easydarwin/EasyDarwin",
        help="EasyDarwin 可执行文件路径"
    )

    return p.parse_args()


# ──────────────────────────────────────────────
#  工具：自动给可执行文件加权限
# ──────────────────────────────────────────────
def ensure_executable(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    current = os.stat(path).st_mode
    if not (current & stat.S_IXUSR):
        print(f"[权限] 自动添加执行权限: {path}")
        os.chmod(path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return True


# ──────────────────────────────────────────────
#  EasyDarwin 管理
# ──────────────────────────────────────────────
_EasyDarwin_proc = None


def start_EasyDarwin(EasyDarwin_path: str) -> None:
    global _EasyDarwin_proc

    if not ensure_executable(EasyDarwin_path):
        
        return

    # 检查是否已经在跑
    result = subprocess.run(["pgrep", "-x", "EasyDarwin"], capture_output=True)
    if result.returncode == 0:
        print("[EasyDarwin] 已在运行，跳过启动")
        return

    print(f"[EasyDarwin] 启动: {EasyDarwin_path}")
    _EasyDarwin_proc = subprocess.Popen(
        [EasyDarwin_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(1.5)
    
    print("[EasyDarwin] RTSP 默认地址 → rtsp://127.0.0.1:15544/stream")


def stop_EasyDarwin() -> None:
    global _EasyDarwin_proc
    if _EasyDarwin_proc and _EasyDarwin_proc.poll() is None:
        print(f"[EasyDarwin] 停止 PID={_EasyDarwin_proc.pid}")
        _EasyDarwin_proc.terminate()
        _EasyDarwin_proc = None
        print("[EasyDarwin] 已停止")


# ──────────────────────────────────────────────
#  退出清理
# ──────────────────────────────────────────────
def _cleanup(signum=None, frame=None):
    print("\n[main] 正在清理，请稍候...")
    stop_godot()
    stop_EasyDarwin()
    sys.exit(0)


# ──────────────────────────────────────────────
#  主函数
# ──────────────────────────────────────────────
def main():
    # EGL 无头渲染，放最前面
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES",
                          "compute,video,utility,graphics")

    setup_environment()
    setup_onnxruntime_cuda()

    warnings.filterwarnings("ignore")
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass

    args = parse_args()

    if not 0.0 < args.nav_scale <= 1.0:
        raise ValueError("--nav-scale 必须在 (0, 1] 范围内")
    if not math.isfinite(args.nav_draw_distance) or args.nav_draw_distance <= 0:
        raise ValueError("--nav-draw-distance 必须大于 0")
    if not math.isfinite(args.nav_projection_hz) or args.nav_projection_hz <= 0:
        raise ValueError("--nav-projection-hz must be greater than 0")
    if not math.isfinite(args.playback_fps) or args.playback_fps < 0:
        raise ValueError("--playback-fps 必须是大于或等于 0 的有限数值")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads 必须大于或等于 1")
    if not args.mqtt_host.strip():
        raise ValueError("--mqtt-host 不能为空")
    if not args.prediction_topic.strip() or not args.gnss_topic.strip():
        raise ValueError("MQTT Topic 不能为空")

    cv2.setNumThreads(args.cpu_threads)
    try:
        import torch
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Inter-op threads can only be configured before parallel work starts.
        pass
    print(f"CPU 工作线程上限: {args.cpu_threads}")

    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)
    # 终端被关闭（挂断）时也走清理，避免 Godot/EasyDarwin 残留
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _cleanup)

    # 1. 启动 EasyDarwin (RTSP 服务器)
    if not args.no_EasyDarwin:
        start_EasyDarwin(args.EasyDarwin)

    # 2. 启动 Godot 后台无头渲染
    if not args.no_godot and args.godot:
        if not ensure_executable(args.godot):
            print(f"[Godot] 找不到文件: {args.godot}，跳过启动")
        else:
            start_godot(args.godot, rtsp_push_url="rtsp://127.0.0.1:15544/stream", wait_seconds=2.0)

    # 3. 检查 TRT Engine
    if not os.path.isfile(SHIP_ENGINE_PATH):
        raise FileNotFoundError(
            f"找不到 YOLO TensorRT Engine：{SHIP_ENGINE_PATH}"
        )

    # 4. 启动推理主循环
    try:
        process_video(
            input_video=args.input,
            output_video=args.output,
            seg_weights=args.seg,
            device_str=args.device,

            enable_streaming=not args.no_stream,
            enable_video_stream=not args.no_video_stream,

            codec=args.codec,
            mask_send_scale=args.mask_send_scale,

            show_window=args.show_window,
            show_scale=args.show_scale,
            playback_fps=args.playback_fps,

            draw_curve_on_python=args.draw_curve,
            draw_masks_on_python=args.draw_masks,

            ship_host=args.ship_host,
            ship_port=args.ship_port,
            yolo_send_port=args.yolo_port,

            save_video=args.save_video,
            nav_send_every=args.nav_send_every,
            nav_draw_distance=args.nav_draw_distance,
            nav_scale=args.nav_scale,
            nav_projection_hz=args.nav_projection_hz,
            calib_ini=args.calib_ini,
            lidar_height=args.lidar_height,
            nav_heading_offset=args.nav_heading_offset,
            mqtt_host=args.mqtt_host,
            prediction_topic=args.prediction_topic,
            gnss_topic=args.gnss_topic,
            mqtt_username=args.mqtt_username,
            mqtt_password=args.mqtt_password,
        )
    except Exception as e:
        import traceback
        print(f"[main] 异常: {e}")
        traceback.print_exc()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
