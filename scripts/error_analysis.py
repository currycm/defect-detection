"""Step 6 入口：难例 / 错误分析。

在 val 集上对最优权重（exp_aug640）逐张推理，与 GT 比对后导出：
  - summary.csv    逐类 TP / FN / FP / 混淆统计
  - confusion.csv  含 background 的 7x7 混淆矩阵
  - <class>/*.jpg  重点类（crazing、pitted_surface）的错误样本可视化
"""
import os

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.evaluation.error_analysis import export_error_cases  # noqa: E402

if __name__ == "__main__":
    export_error_cases(
        weights=os.path.join(PROJECT, "runs", "detect", "runs",
                             "exp_aug640", "weights", "best.pt"),
        data_root=os.path.join(PROJECT, "data", "processed"),
        split="val",
        imgsz=640,
        conf=0.25,
        iou_thr=0.5,
        out_dir=os.path.join(PROJECT, "runs", "error_analysis"),
        focus=("crazing", "pitted_surface"),
        max_save=8,
    )
