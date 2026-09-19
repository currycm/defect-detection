"""缺陷推理封装（基于 ultralytics / PyTorch，作为 ONNX 不可用时的回退路径）。

与 `onnx_predictor.ONNXDefectPredictor` 保持**同一份输出契约**：
    {"xyxy": [x1, y1, x2, y2], "conf": float, "cls": int, "name": str}
（此前本类不返回 `name`，导致 API 的 `Detection(**d)` 校验失败、/detect 必然 500。）

中文路径：读图统一走 np.fromfile + cv2.imdecode（cv2.imread 在含中文的路径下静默返回 None）。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ultralytics import YOLO

from ..utils.imageio import load_image
from ..utils.logger import get_logger

LOG = get_logger("predictor")


class DefectPredictor:
    """封装 YOLO 推理与后处理，纯函数式、可独立测试。"""

    def __init__(self, weights: str, class_names: Sequence[str], conf: float = 0.25,
                 iou: float = 0.5):
        self.model = YOLO(weights)
        self.class_names = list(class_names)
        self.conf = conf
        self.iou = iou

    def predict_image(self, image: np.ndarray, conf: float | None = None,
                      iou: float | None = None) -> list[dict]:
        """对 numpy 图像推理，返回检测结果列表。

        conf / iou 按调用传入（不再落到实例属性上改共享状态——那会让并发请求互相篡改阈值，
        并且旧实现在回退路径上会静默忽略 iou）。
        """
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou

        res = self.model.predict(image, conf=conf, iou=iou, verbose=False)[0]
        dets: list[dict] = []
        for b in res.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(b.conf[0]),
                "cls": cls,
                "name": (self.class_names[cls]
                         if 0 <= cls < len(self.class_names) else str(cls)),
            })
        return dets

    def predict_path(self, path: str) -> list[dict]:
        """对图片路径推理（中文路径安全）。"""
        img = load_image(path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片（或格式不支持）: {path}")
        return self.predict_image(img)
