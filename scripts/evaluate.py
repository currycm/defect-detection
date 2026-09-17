"""评估入口：在指定划分（默认独立 test 集）上评估最优权重。

用法：
    python scripts/evaluate.py                        # test 集 + exp_aug640/best.pt
    python scripts/evaluate.py --split val
    python scripts/evaluate.py --weights runs/detect/runs/exp/weights/best.pt --split test

修复记录（M7）：原实现把权重写死为 `runs/exp/weights/best.pt`，
但实际训练产物在 `runs/detect/runs/exp_aug640/weights/best.pt`
（ultralytics 会在 runs/ 下再套一层 detect/），路径不存在会直接抛错。
现在默认值取自 src.utils.paths.BEST_PT，并支持 --weights 覆盖。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import evaluate, format_report  # noqa: E402
from src.utils import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLOv8 缺陷检测评估")
    ap.add_argument("--weights", default=str(paths.BEST_PT),
                    help="权重路径（默认: %(default)s）")
    ap.add_argument("--data", default=str(paths.DATA_YAML),
                    help="data.yaml 路径（默认: %(default)s）")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="评估划分（默认 test，即独立 held-out 集）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(
            f"未找到权重: {args.weights}\n"
            f"提示：训练产物通常在 runs/detect/runs/<exp>/weights/best.pt，"
            f"可用 --weights 指定；先跑 python scripts/train.py 生成。"
        )

    res = evaluate(
        args.weights, data_yaml=args.data, split=args.split,
        imgsz=args.imgsz, conf=args.conf, iou=args.iou,
    )
    print(format_report(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
