"""阈值调优的无偏验证：在 test 集（训练/调参全程未用过）上确认。

阈值是在 val 集上选出来的，直接报 val 上的提升属于「乐观估计」。
这里在 test 集上复评，给出诚实结论。

用法：python scripts/verify_threshold_test.py
"""
import os

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from ultralytics import YOLO  # noqa: E402

DATA_YAML = os.path.join(PROJECT, "configs", "data.yaml")
WEIGHT = os.path.join(PROJECT, "runs", "detect", "runs", "exp_aug640", "weights", "best.pt")

# (标签, conf, iou)  —— 对比「ultralytics 默认」与「val 上选出的最优」
CONFIGS = [
    ("默认 (conf=0.001, iou=0.7)", 0.001, 0.7),
    ("调优 (conf=0.001, iou=0.5)", 0.001, 0.5),
    ("高 Precision 档 (conf=0.5, iou=0.5)", 0.5, 0.5),
]


def main():
    model = YOLO(WEIGHT)
    print("在 test 集上验证（该集未参与训练与调参）\n")
    print(f"{'配置':<34}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for label, conf, iou in CONFIGS:
        r = model.val(data=DATA_YAML, imgsz=640, conf=conf, iou=iou,
                      split="test", workers=0, verbose=False, plots=False)
        p, rec = float(r.box.mp), float(r.box.mr)
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        print(f"{label:<34}{p:>8.3f}{rec:>8.3f}{f1:>8.3f}"
              f"{float(r.box.map50):>9.3f}{float(r.box.map):>10.3f}")


if __name__ == "__main__":
    main()
