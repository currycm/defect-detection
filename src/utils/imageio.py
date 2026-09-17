"""图像读写与画框工具（全项目唯一实现）。

以前 `_imread_unicode` / `_imwrite_unicode` 在 augment / stream / error_analysis /
ui / 多个 scripts 里各写一份，颜色表也有两份，画框函数有两套且行为不一致
（`utils/visualize.draw_boxes` 缺标签右边界钳制，靠右的框文字会溢出图像）。
本模块收敛为唯一实现。

Windows 中文路径：`cv2.imread` / `cv2.imwrite` / `np.tofile` 对含非 ASCII 的路径
会**静默失败**（读返回 None、写不报错但文件不生成）。统一走
    读: np.fromfile -> cv2.imdecode
    写: cv2.imencode -> Path.write_bytes
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# 每个类别一个固定颜色（BGR），便于肉眼区分
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 128, 0),    # crazing          蓝
    1: (0, 200, 0),      # inclusion        绿
    2: (0, 0, 255),      # patches          红
    3: (0, 220, 220),    # pitted_surface   黄
    4: (255, 0, 255),    # rolled-in_scale  品红
    5: (255, 200, 0),    # scratches        青
}

_DEFAULT_COLOR = (0, 255, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_image(path: str | os.PathLike) -> np.ndarray | None:
    """中文路径安全读图，返回 BGR ndarray；失败返回 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path: str | os.PathLike, img: np.ndarray) -> bool:
    """中文路径安全写图；成功返回 True。"""
    ext = os.path.splitext(str(path))[1].lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        Path(path).write_bytes(buf.tobytes())
    except OSError:
        return False
    return True


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> bytes:
    """编码为 JPEG 字节（供 MJPEG 流式响应使用）。"""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def draw_detections(frame_bgr: np.ndarray, dets: Sequence[dict],
                    thickness: int = 2,
                    names: Sequence[str] | None = None) -> np.ndarray:
    """低开销画框（cv2.putText），每帧调用也扛得住流式帧率。

    - 标签优先取 det["name"]，没有则用 names[det["cls"]] 回退，都没有就只画类别号；
    - 标签底色按实测文字宽高绘制，并对 x/y 做**边界钳制**，
      避免靠右/靠上的框标签溢出或裁切（旧 draw_boxes 就缺这一步）；
    - 需要中文标注请用界面层的 PIL 版本（帧率要求低、可读性优先）。
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for d in dets:
        xyxy = d.get("xyxy")
        if xyxy is None:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        cls = int(d.get("cls", -1))
        color = CLASS_COLORS.get(cls, _DEFAULT_COLOR)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        name = d.get("name")
        if name is None and names is not None and 0 <= cls < len(names):
            name = names[cls]
        label = f"{name if name is not None else cls} {float(d.get('conf', 0.0)):.2f}"

        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        tx = min(max(x1, 0), max(w - tw - 4, 0))
        ty = max(y1 - th - 4, 0)
        cv2.rectangle(out, (tx, ty), (tx + tw + 4, ty + th + 4), color, -1)
        cv2.putText(out, label, (tx + 2, ty + th + 1),
                    _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_boxes(image: np.ndarray, boxes: Sequence[dict],
               names: Sequence[str], thickness: int = 2) -> np.ndarray:
    """兼容旧签名 `draw_boxes(image, boxes, names)`，内部转调 draw_detections。"""
    return draw_detections(image, boxes, thickness=thickness, names=names)
