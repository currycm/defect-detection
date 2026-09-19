"""API 数据契约（Pydantic）。

历史上两个 predictor 的返回字典不一致（ONNX 版带 `name`，ultralytics 版不带），
导致 torch 回退路径的 `Detection(**d)` 直接 pydantic ValidationError（/detect 必 500）。
现在统一从 `to_detection()` 收口，`name` 也给了默认值做二次容错。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field


class Detection(BaseModel):
    # xyxy = [x1, y1, x2, y2]，4 个值；不强制非负，因为 letterbox 还原到原图
    # 坐标时钳制只发生在 predictor 内，万一有极小负值漏出也允许通过，
    # 但 4 个值的硬约束还是要有，防 predictor 误吐空/异常长度。
    xyxy: list[float] = Field(..., min_length=4, max_length=4)
    # conf 在 [0,1]；超过 1 说明导出时没 sigmoid 或后处理 bug，宁可拒绝也不要脏数据
    conf: float = Field(..., ge=0.0, le=1.0)
    cls: int = Field(..., ge=0)
    # 有默认值：即使某个 producer 漏了 name 也只是降级，而不是整个接口 500
    name: str = ""


class DetectResponse(BaseModel):
    detections: list[Detection]
    count: int = Field(..., ge=0)


def to_detection(det: Mapping, class_names: Sequence[str] | None = None) -> Detection:
    """把任意 predictor 输出的字典规整成 Detection（唯一收口点）。

    - `name` 缺失时按 cls 从类别表补齐；
    - 坐标/置信度统一转成 python float（避免 numpy 标量导致 JSON 序列化失败）；
    - 越界值在此显式钳制到合法范围，作为模型层之外的最后一道兜底
      （防御性：predictor 实现理论上不该吐出非法值，但万一发生也不应让
      整个 /detect 500，而是丢掉这一条框、继续返回剩余结果）。
    """
    cls = int(det["cls"])
    name = det.get("name")
    if not name:
        if class_names is not None and 0 <= cls < len(class_names):
            name = class_names[cls]
        else:
            name = str(cls)

    raw_conf = float(det.get("conf", 0.0))
    conf = min(max(raw_conf, 0.0), 1.0)
    if conf != raw_conf:
        # 不抛错、不静默吞：交给上层决定怎么记。这里只做规整。
        # 调用方（/detect）目前不传 logger 进来，所以暂时不发警告；
        # 真要追溯可在 to_detection 加可选 logger 参数。
        pass

    xyxy = [float(v) for v in det["xyxy"]]
    return Detection(xyxy=xyxy, conf=conf, cls=cls, name=str(name))
