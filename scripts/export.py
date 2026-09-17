"""导出 ONNX 入口（Step 7）。

把最优权重（paths.BEST_PT，即 exp_aug640/weights/best.pt）导出为 ONNX，
放到 weights/best.onnx，供 FastAPI 服务用 ONNXRuntime 加载（无需 torch）。

用法：
    python scripts/export.py
    python scripts/export.py --weights runs/detect/runs/exp/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.models.wrapper import export_onnx, load_model  # noqa: E402
from src.utils import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 YOLOv8 权重为 ONNX")
    ap.add_argument("--weights", default=str(paths.BEST_PT),
                    help="输入 .pt 权重（默认: %(default)s）")
    ap.add_argument("--out", default=str(paths.ONNX_PATH),
                    help="输出 .onnx 路径（默认: %(default)s）")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(
            f"未找到权重: {args.weights}\n"
            f"提示：训练产物通常在 runs/detect/runs/<exp>/weights/best.pt，"
            f"先用 --weights 指定；或先跑 python scripts/train.py。"
        )

    model = load_model(args.weights)
    out = export_onnx(model, args.out, imgsz=args.imgsz,
                      dynamic=False, simplify=False)
    print(f"已放置到: {out}")
    print(f"文件大小: {Path(out).stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
