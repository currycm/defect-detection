"""数据准备入口：转换 NEU-DET 标注 + 划分 train/val/test。

用法：
    python scripts/prepare_data.py                # 转换 + 切分
    python scripts/prepare_data.py --no-split     # 只转换，不切分
"""
from __future__ import annotations

import argparse

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.constants import CLASS_NAMES
from src.data.convert import convert_all
from src.data.split import split
from src.utils import paths


def main() -> int:
    ap = argparse.ArgumentParser(description="NEU-DET 数据准备")
    ap.add_argument("--raw", default=str(paths.RAW_DIR))
    ap.add_argument("--out", default=str(paths.PROCESSED_DIR))
    ap.add_argument("--no-split", action="store_true",
                    help="只做 XML->YOLO 转换，不切分 val/test")
    args = ap.parse_args()

    counts = convert_all(args.raw, args.out, CLASS_NAMES)
    if not counts:
        return 1

    if not args.no_split:
        # split() 自带幂等保护：test 目录已有内容时会跳过而非重复切分
        split(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
