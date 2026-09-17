"""实时/流式缺陷检测（Step 8b）。

把「一次性调用」的检测升级为「持续流式处理」：
    输入源 -> 按帧读取 -> 限频推理 -> 结果去抖 -> 状态变化回调 -> 输出带框画面

设计要点与对应需求：
  1) 输入源与启停：FrameSource 抽象（摄像头 / 视频文件 / 网络流 / 合成帧源）；
     StreamDetector 提供 start / pause / resume / stop 显式控制。
  2) 控频与避免重复处理：process_frame() 内部按 detect_interval 限频，
     间隔内的帧直接复用上次检测结果，不重复推理；run() 再用 target_fps 限制循环节奏。
  3) 结果稳定性：ResultStabilizer 把检测结果转成「量化签名」（类别 + 框中心网格量化），
     连续 stable_frames 次相同才确认，且只在签名发生变化时触发回调（去重 + 去抖）。
  4) 资源释放：daemon 线程 + threading.Event 控制；stop() 置位 -> join -> 释放帧源；
     幂等且支持 with 语法，避免线程残留与内存泄漏。

安全：`open_source(..., allowed_roots=...)` 可把合成帧源的路径限制在白名单根目录内，
用于挡住 `/stream?source=` 的任意文件读取（详见 api._resolve_source）。

中文目录注意：读图统一走 src.utils.imageio.load_image
（cv2.imread / np.tofile 在含中文的路径下会静默失败）。
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..utils.imageio import load_image

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------
# 输入源
# --------------------------------------------------------------------------
class FrameSource:
    """帧源基类。"""

    is_live = False  # 直播源（摄像头/网络流）为 True，文件和合成源为 False

    def read(self):
        """返回 (ok, frame_bgr)。"""
        raise NotImplementedError

    def release(self) -> None:
        pass

    def reset(self) -> None:
        """循环播放时回到起点。"""


class CaptureSource(FrameSource):
    """基于 cv2.VideoCapture：支持摄像头索引、视频文件、RTSP/HTTP 流。"""

    def __init__(self, source, loop: bool = False):
        import cv2  # 惰性导入，避免无视频需求时也拉起 cv2

        self.source = source
        self.loop = loop
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开输入源: {source}")
        # 直播源判定：整数索引（摄像头）或字符串 URL
        self.is_live = isinstance(source, int) or (
            isinstance(source, str)
            and source.lower().startswith(("rtsp://", "http://", "https://", "rtmp://"))
        )

    def read(self):
        import cv2

        ok, frame = self.cap.read()
        if not ok and self.loop and not self.is_live:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return (ok and frame is not None), frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SyntheticSource(FrameSource):
    """合成帧源：循环播放一组图片，用于无头验证与演示（不需要摄像头）。

    hold: 每张图连续输出多少帧。真实摄像头/产线画面相邻帧几乎不变，
          而「每帧换一张图」是极端场景（签名每帧都变，去抖必然不触发）。
          hold>1 用来模拟「静止场景」，才能观测到稳定状态变化事件。
    """

    def __init__(self, image_paths: Sequence[str], repeat: int = 1, hold: int = 1):
        self.paths = list(image_paths)
        if not self.paths:
            raise ValueError("合成帧源需要至少一张图片")
        self.repeat = max(int(repeat), 1)
        self.hold = max(int(hold), 1)
        self._idx = 0
        self._round = 0
        self._held = 0

    def read(self):
        if self._round >= self.repeat:
            return False, None
        frame = load_image(self.paths[self._idx])
        self._held += 1
        if self._held >= self.hold:      # 同一张图播够 hold 帧才换下一张
            self._held = 0
            self._idx += 1
            if self._idx >= len(self.paths):
                self._idx = 0
                self._round += 1
        if frame is None:
            return False, None
        return True, frame

    def reset(self) -> None:
        self._idx = 0
        self._round = 0
        self._held = 0


def _list_images(target: str) -> list[str]:
    if os.path.isdir(target):
        return sorted(
            os.path.join(target, f) for f in os.listdir(target)
            if f.lower().endswith(_IMAGE_EXTS)
        )
    return [target]


def _path_allowed(path: str, roots: Sequence[Path]) -> bool:
    """路径是否落在任一允许根目录内。

    空串/纯空白直接拒绝：`Path("")` 会 resolve 成当前工作目录，
    会把「空路径」误判成「允许」（与 utils.paths.is_path_allowed 保持一致）。
    """
    try:
        if not str(path).strip():
            return False
        resolved = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return False
    for root in roots:
        try:
            if resolved == root or resolved.is_relative_to(root):
                return True
        except OSError:
            continue
    return False


def open_source(source, loop: bool = False, hold: int = 1,
                allowed_roots: Sequence[Path] | None = None) -> FrameSource:
    """把各种写法统一成 FrameSource。

    source 可以是：
      - int                     ：摄像头索引（0 通常是内置摄像头）
      - "synthetic:dir_or_path" ：合成帧源（循环播放给定图片，便于测试）
      - 其它 str / Path         ：视频文件路径 或 rtsp/http 流地址
    hold          仅对合成帧源生效（每张图连续播多少帧）。
    allowed_roots 若给定，则合成帧源的路径必须落在这些根目录内，否则抛 PermissionError。
    """
    if isinstance(source, SyntheticSource):
        return source
    if isinstance(source, str) and source.startswith("synthetic:"):
        target = source.split(":", 1)[1]
        if allowed_roots is not None and not _path_allowed(target, allowed_roots):
            raise PermissionError(f"路径不在允许的根目录内: {target}")
        paths = _list_images(target)
        if not paths:
            raise FileNotFoundError(f"合成帧源没有可用图片: {target}")
        return SyntheticSource(paths, repeat=10_000 if loop else 1, hold=hold)
    return CaptureSource(source, loop=loop)


# --------------------------------------------------------------------------
# 结果去抖
# --------------------------------------------------------------------------
class ResultStabilizer:
    """把逐帧检测结果稳定成「状态」，只在状态真正变化时放行。

    - 量化签名：类别 + 框中心落在哪个网格（pos_tolerance 控制网格粒度），
      避免同一目标因为几个像素的抖动就被当成新状态。
    - 连续 stable_frames 次相同才确认（去抖），过滤偶发漏检/误检造成的闪烁。
    """

    def __init__(self, stable_frames: int = 3, pos_tolerance: float = 0.08):
        self.stable_frames = max(int(stable_frames), 1)
        self.pos_tolerance = pos_tolerance
        self._pending = None
        self._pending_count = 0
        self._emitted = None
        # 统计用
        self.updates = 0
        self.emits = 0

    def signature(self, dets: Iterable[dict], w: int, h: int) -> tuple:
        tol = max(self.pos_tolerance, 1e-6)
        items = []
        for d in dets:
            x1, y1, x2, y2 = d["xyxy"]
            cx = int(((x1 + x2) / 2) / max(w, 1) / tol)
            cy = int(((y1 + y2) / 2) / max(h, 1) / tol)
            items.append((int(d["cls"]), cx, cy))
        return tuple(sorted(items))

    def update(self, dets: list[dict], w: int, h: int) -> bool:
        """喂入一帧结果，返回 True 表示「稳定状态发生变化，应当回调」。"""
        self.updates += 1
        sig = self.signature(dets, w, h)
        if sig == self._pending:
            self._pending_count += 1
        else:
            self._pending = sig
            self._pending_count = 1
        if self._pending_count >= self.stable_frames and sig != self._emitted:
            self._emitted = sig
            self.emits += 1
            return True
        return False

    @property
    def stable_signature(self) -> tuple:
        return self._emitted if self._emitted is not None else ()


# --------------------------------------------------------------------------
# 实时检测器
# --------------------------------------------------------------------------
# 参数合法区间（接口层与内部都做钳制，避免调用方传负值绕过限频）
FPS_RANGE = (1.0, 120.0)
INTERVAL_RANGE = (0.001, 60.0)
HOLD_RANGE = (1, 10_000)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


class StreamDetector:
    """实时流式检测器：限频推理 + 结果去抖 + 状态回调 + 线程化启停。

    典型用法：
        det = StreamDetector(predictor, source=0, target_fps=15,
                             detect_interval=0.1, on_change=cb)
        det.start(); ...; det.stop()
    """

    def __init__(self, predictor, source, *,
                 target_fps: float = 15.0,
                 detect_interval: float = 0.1,
                 stable_frames: int = 3,
                 pos_tolerance: float = 0.08,
                 conf: float = 0.25,
                 iou: float = 0.5,
                 loop: bool = True,
                 hold: int = 1,
                 allowed_roots: Sequence[Path] | None = None,
                 on_change: Callable[[list[dict], np.ndarray], None] | None = None,
                 draw: Callable[[np.ndarray, list[dict]], np.ndarray] | None = None):
        self.predictor = predictor
        self.source_arg = source
        # 钳制：负数/0 会让限频与节流形同虚设（旧实现实测 detect_interval=-1 时推理占比 88.9%）
        self.target_fps = _clamp(target_fps, *FPS_RANGE)
        self.detect_interval = _clamp(detect_interval, *INTERVAL_RANGE)
        self.conf = _clamp(conf, 0.0, 1.0)
        self.iou = _clamp(iou, 0.0, 1.0)
        self.stable_frames = max(int(stable_frames), 1)
        self.hold = int(_clamp(hold, *HOLD_RANGE))
        self.loop = loop
        self.allowed_roots = tuple(allowed_roots) if allowed_roots else None
        self.on_change = on_change
        self._draw = draw

        self.stabilizer = ResultStabilizer(stable_frames=self.stable_frames,
                                          pos_tolerance=pos_tolerance)

        self._source: FrameSource | None = None
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_dets: list[dict] = []   # 对外发布的最新结果（锁内读写）
        self._last_dets: list[dict] = []     # 限频复用的上一次推理结果
        self._last_detect_ts = 0.0
        self._frames_read = 0
        self._frames_inferred = 0

    # ---------------- 单帧处理（可被外部驱动直接调用） ----------------
    def process_frame(self, frame: np.ndarray):
        """处理一帧。返回 (标注帧, 检测结果, 状态是否变化)。

        限频逻辑：距上次推理不足 detect_interval 时，直接复用上次结果，
        不调用模型 —— 这是「避免重复处理造成性能浪费」的关键。
        """
        h, w = frame.shape[:2]
        now = time.time()

        changed = False
        if now - self._last_detect_ts >= self.detect_interval:
            dets = self.predictor.predict_image(frame, conf=self.conf, iou=self.iou)
            self._last_detect_ts = now
            self._last_dets = dets
            self._frames_inferred += 1
            changed = self.stabilizer.update(dets, w, h)
            if changed and self.on_change is not None:
                self.on_change(dets, frame)
        else:
            dets = self._last_dets

        annotated = self._draw(frame, dets) if self._draw is not None else frame
        with self._lock:
            self._latest_frame = annotated
            self._latest_dets = dets
        return annotated, dets, changed

    # ---------------- 线程化运行 ----------------
    def _loop(self) -> None:
        frame_period = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        try:
            while not self._stop_evt.is_set():
                if self._pause_evt.is_set():
                    time.sleep(0.02)
                    continue

                t0 = time.time()
                ok, frame = self._source.read()
                if not ok or frame is None:
                    if self.source.is_live:
                        time.sleep(0.05)   # 直播源偶发丢帧：稍等重试
                        continue
                    break                  # 文件/合成源读完即结束
                self._frames_read += 1
                self.process_frame(frame)

                if frame_period > 0:
                    rest = frame_period - (time.time() - t0)
                    if rest > 0:
                        time.sleep(rest)
        finally:
            if self._source is not None:
                self._source.release()

    def start(self) -> "StreamDetector":
        """启动。幂等：已在运行则直接返回。"""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._source = open_source(self.source_arg, loop=self.loop, hold=self.hold,
                                   allowed_roots=self.allowed_roots)
        self._stop_evt.clear()
        self._pause_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="stream-detector", daemon=True)
        self._thread.start()
        return self

    @property
    def source(self) -> FrameSource | None:
        return self._source

    def pause(self) -> None:
        """暂停（线程仍在，只是不读帧不推理）。"""
        self._pause_evt.set()

    def resume(self) -> None:
        self._pause_evt.clear()

    @property
    def paused(self) -> bool:
        return self._pause_evt.is_set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        """停止并释放资源。幂等，可重复调用。"""
        self._stop_evt.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=timeout)
        # 兜底：线程若卡在阻塞 read 上，显式释放帧源促使 read 返回
        if self._source is not None:
            try:
                self._source.release()
            except Exception:
                pass
        self._thread = None
        self._source = None

    def latest(self):
        """取最新一帧与结果（供 MJPEG / 轮询接口使用），线程安全。"""
        with self._lock:
            return self._latest_frame, list(self._latest_dets)

    def stats(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "frames_read": self._frames_read,
            "frames_inferred": self._frames_inferred,
            "infer_ratio": (self._frames_inferred / self._frames_read
                            if self._frames_read else 0.0),
            "stabilizer_updates": self.stabilizer.updates,
            "stabilizer_emits": self.stabilizer.emits,
            "latest_detections": len(self._latest_dets),
        }

    # ---------------- 上下文管理 ----------------
    def __enter__(self) -> "StreamDetector":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
