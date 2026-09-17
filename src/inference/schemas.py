"""API 数据契约（Pydantic）。

历史上两个 predictor 的返回字典不一致（ONNX 版带 `name`，ultralytics 版不带），
导致 torch 回退路径的 `Detection(**d)` 直接 pydantic ValidationError（/detect 必 500）。
现在统一从 `to_detection()` 收口，`name` 也给了默认值做二次容错。
"""
from __future__ import annotations

from typing import Mapping, Sequence

from pydantic import BaseModel


class Detection(BaseModel):
    xyxy: list[float]
    conf: float
    cls: int
    # 有默认值：即使某个 producer 漏了 name 也只是降级，而不是整个接口 500
    name: str = ""


class DetectResponse(BaseModel):
    detections: list[Detection]
    count: int


def to_detection(det: Mapping, class_names: Sequence[str] | None = None) -> Detection:
    """把任意 predictor 输出的字典规整成 Detection（唯一收口点）。

    - `name` 缺失时按 cls 从类别表补齐；
    - 坐标/置信度统一转成 python float（避免 numpy 标量导致 JSON 序列化失败）。
    """
    cls = int(det["cls"])
    name = det.get("name")
    if not name:
        if class_names is not None and 0 <= cls < len(class_names):
            name = class_names[cls]
        else:
            name = str(cls)
    return Detection(
        xyxy=[float(v) for v in det["xyxy"]],
        conf=float(det.get("conf", 0.0)),
        cls=cls,
        name=str(name),
    )
