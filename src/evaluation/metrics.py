"""评估指标封装。

在指定 split 上跑 ultralytics 验证，并把结果整理成「整体 + 逐类」结构，
供 scripts/evaluate.py 与错误分析复用。
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def evaluate(
    weights: str | Path,
    data_yaml: str | Path = "configs/data.yaml",
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
    verbose: bool = False,
) -> dict:
    """在 `split` 指定的划分上评估权重，返回整体与逐类指标。

    参数
        weights   : .pt / .onnx 权重路径
        data_yaml : 数据集配置（需含 train/val/test 三路）
        split     : "train" | "val" | "test"，默认 test（独立 held-out 集）
        conf/iou  : 验证时的置信度 / NMS 阈值；conf 默认取极低值 0.001，
                    因为 mAP 是「全召回扫描」指标，阈值调高只会让 AP 虚高。

    返回
        {
          "weights", "split", "mAP50", "mAP50_95", "precision", "recall",
          "per_class": [{class_id, name, precision, recall, ap50, ap50_95}, ...],
          "save_dir",
        }
    """
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=verbose,
    )
    box = metrics.box
    names = {int(k): v for k, v in model.names.items()}

    per_class = []
    for i, cid in enumerate(box.ap_class_index):
        c = int(cid)
        per_class.append({
            "class_id": c,
            "name": names.get(c, str(c)),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "ap50": float(box.ap50[i]),
            "ap50_95": float(box.maps[i]),
        })
    # 固定按 class_id 排序，避免因「该类在验证集未出现」导致顺序漂移
    per_class.sort(key=lambda d: d["class_id"])

    return {
        "weights": str(weights),
        "split": split,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }


def format_report(res: dict) -> str:
    """把 evaluate 的结果渲染成可读文本表。"""
    lines = [
        f"权重: {res['weights']}",
        f"评估集: {res['split']}",
        f"整体: mAP@0.5={res['mAP50']:.4f}  mAP@0.5:0.95={res['mAP50_95']:.4f}"
        f"  P={res['precision']:.4f}  R={res['recall']:.4f}",
        "",
        f"{'类别':<18}{'P':>8}{'R':>8}{'AP50':>9}{'AP50-95':>10}",
        "-" * 53,
    ]
    for d in res["per_class"]:
        lines.append(
            f"{d['name']:<18}{d['precision']:>8.3f}{d['recall']:>8.3f}"
            f"{d['ap50']:>9.3f}{d['ap50_95']:>10.3f}"
        )
    if res.get("save_dir"):
        lines += ["", f"图表与混淆矩阵: {res['save_dir']}"]
    return "\n".join(lines)
