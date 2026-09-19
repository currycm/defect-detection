"""Step 6b：置信度 / NMS 阈值调优（零训练成本）。

背景：Step6 错误分析发现 FP(203) > FN(139)，模型「过度预测」——
说明当前工作点偏激进，抬高置信度阈值有望砍掉大量误检、换取更高 precision。

做法（两阶段，避免全网格爆炸）：
  阶段1：固定 NMS iou=0.5，扫 conf
  阶段2：在最优 conf 上，扫 NMS iou
每档都用 ultralytics 的 val 在统一 val 集上评估，记录 P / R / F1 / mAP50 / mAP50-95。

输出：runs/threshold_tuning/sweep.csv，并打印「最佳 F1 工作点」与「最佳 mAP50-95 工作点」。
"""
import csv
import os

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from ultralytics import YOLO

from src.utils import paths

# 经 paths 绝对化：ultralytics 解析相对 `path` 时以 cwd 为准，不可依赖
DATA_YAML = str(paths.runtime_data_yaml())
WEIGHT = os.path.join(PROJECT, "runs", "detect", "runs", "exp_aug640", "weights", "best.pt")
OUT_DIR = os.path.join(PROJECT, "runs", "threshold_tuning")

CONF_GRID = [0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
IOU_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def evaluate(model, conf: float, iou: float, imgsz: int = 640) -> dict:
    r = model.val(
        data=DATA_YAML, imgsz=imgsz, conf=conf, iou=iou,
        split="val", workers=0, verbose=False, plots=False,
    )
    p, rec = float(r.box.mp), float(r.box.mr)
    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
    return {
        "conf": conf, "iou": iou, "P": p, "R": rec, "F1": f1,
        "mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = YOLO(WEIGHT)
    rows: list[dict] = []

    print("阶段1：扫 conf（固定 NMS iou=0.5）")
    for c in CONF_GRID:
        m = evaluate(model, c, 0.5)
        rows.append(m)
        print(f"  conf={c:<6} P={m['P']:.3f} R={m['R']:.3f} "
              f"F1={m['F1']:.3f} mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50_95']:.3f}")

    best_conf_row = max(rows, key=lambda r: r["F1"])
    best_conf = best_conf_row["conf"]
    print(f"\n  最佳 F1 的 conf = {best_conf} (F1={best_conf_row['F1']:.3f})")

    print(f"\n阶段2：在 conf={best_conf} 上扫 NMS iou")
    for i in IOU_GRID:
        m = evaluate(model, best_conf, i)
        m["stage"] = "iou"
        rows.append(m)
        print(f"  iou={i:<6} P={m['P']:.3f} R={m['R']:.3f} "
              f"F1={m['F1']:.3f} mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50_95']:.3f}")

    with open(os.path.join(OUT_DIR, "sweep.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["stage", "conf", "iou", "P", "R", "F1", "mAP50", "mAP50-95"])
        for r in rows:
            w.writerow([r.get("stage", "conf"), r["conf"], r["iou"],
                        f"{r['P']:.4f}", f"{r['R']:.4f}", f"{r['F1']:.4f}",
                        f"{r['mAP50']:.4f}", f"{r['mAP50_95']:.4f}"])

    b_f1 = max(rows, key=lambda r: r["F1"])
    b_map = max(rows, key=lambda r: r["mAP50_95"])
    b_map50 = max(rows, key=lambda r: r["mAP50"])
    print("\n=== 最优工作点 ===")
    print(f"  最佳 F1      : conf={b_f1['conf']} iou={b_f1['iou']} "
          f"F1={b_f1['F1']:.3f} (P={b_f1['P']:.3f} R={b_f1['R']:.3f}) mAP50={b_f1['mAP50']:.3f}")
    print(f"  最佳 mAP50-95: conf={b_map['conf']} iou={b_map['iou']} mAP50-95={b_map['mAP50_95']:.3f} "
          f"(P={b_map['P']:.3f} R={b_map['R']:.3f})")
    print(f"  最佳 mAP50   : conf={b_map50['conf']} iou={b_map50['iou']} mAP50={b_map50['mAP50']:.3f} "
          f"(P={b_map50['P']:.3f} R={b_map50['R']:.3f})")
    print(f"\nCSV: {os.path.join(OUT_DIR, 'sweep.csv')}")


if __name__ == "__main__":
    main()
