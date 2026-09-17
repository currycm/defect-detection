"""训练入口。

用法：
    python scripts/train.py                              # imgsz=640, name=exp（推荐）
    python scripts/train.py --imgsz 320 --name exp_aug   # Step5 那次消融实验的配置

消融规范：一次只改一个变量（要么改分辨率、要么改增强），否则无法归因。
项目根由 scripts/_bootstrap 统一解析（可用环境变量 DEFECT_PROJECT_ROOT 覆盖）。

注意（L11）：imgsz 默认值改为 640 —— 训练/评估/部署三条链路实际都用 640，
此前默认 320 属历史残留，容易让人以为线上跑的是 320。
"""
from __future__ import annotations

import argparse

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.training.train import train  # noqa: E402
from src.utils import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLOv8 缺陷检测训练")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default="exp", help="实验名（决定 runs/detect/runs/<name>）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default=str(paths.PRETRAINED_PT),
                    help="初始权重（默认: %(default)s）")
    ap.add_argument("--data", default=str(paths.DATA_YAML))
    ap.add_argument("--hyp", default=str(paths.HYP_YAML))
    args = ap.parse_args()

    # 权重用绝对路径：相对路径在 cwd 异常时会退化成「去 GitHub 下载」
    save_dir = train(
        model=args.weights,
        data_yaml=args.data,
        hyp_yaml=args.hyp,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
    )
    print("权重目录:", save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
