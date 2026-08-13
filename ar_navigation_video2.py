"""
轻量 stub：仅供本地离线预览 draw_gnss_navigation_band_patched.py。

完整版（TCP 服务器 + TensorRT 分割 + YOLO）在
ultralytics-v11/ar_navigation_video2.py，需要 GPU 环境。
本 stub 跳过 TCP 发送，只在本机窗口显示导航线。
"""
import threading

nav_server = None


def start_tcp_server():
    print("[stub] 跳过 TCP 服务器（本地预览模式，导航线仅本地窗口显示）")


# 兼容旧调用：如果其他代码直接调用 nav_server 方法，这里安全返回 None
def send_prepared_nav(*args, **kwargs):
    return None
