"""从 provided validation 切分独立 test 集，得到 train/val/test。

NEU-DET 原始仅提供 train/validation。为获得严格独立的 held-out 评估集，
这里把 convert 产出的 images/val 按 test_ratio 切分为 val + test，
train 保持不变（数据.yaml 中 train/val/test 三路均可用）。
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from ..utils.io import ensure_dir
from ..utils.logger import get_logger

LOG = get_logger("split")


def split(processed_dir: str, test_ratio: float = 0.5, seed: int = 42) -> None:
    """把 data/processed/images/val 按 test_ratio 切出 test，其余留作 val。

    切分在类别层面近似均衡：先整体打乱再取前 test_ratio 比例，因源数据
    每类样本数相同（各 60 张），结果天然均衡。
    """
    base = Path(processed_dir)
    val_img = base / "images" / "val"
    val_lbl = base / "labels" / "val"
    if not val_img.exists():
        LOG.error("未找到 %s，请先运行 convert", val_img)
        return

    # 幂等保护（M5）：split 用 shutil.move 切走文件，本身不可重复执行。
    # 若 test 目录已有内容，说明切分过了，再跑会把 val 继续掏空，
    # 因此这里直接跳过并提示，避免误操作破坏数据集。
    test_img = base / "images" / "test"
    if test_img.exists() and any(test_img.iterdir()):
        LOG.warning("检测到 %s 已有 %d 个文件，切分已完成，跳过（如需重切请先清空 test）",
                    test_img, len(list(test_img.iterdir())))
        return

    imgs = sorted(p for p in val_img.glob("*") if p.is_file())
    # 用局部 Random 实例而非 random.seed()：后者会改动**全局**随机状态，
    # 影响同进程内其它模块（含测试）的可复现性。
    rng = random.Random(seed)
    rng.shuffle(imgs)
    n_test = int(len(imgs) * test_ratio)
    test_files = imgs[:n_test]

    od_i = ensure_dir(base / "images" / "test")
    od_l = ensure_dir(base / "labels" / "test")
    n_moved = 0
    for f in test_files:
        shutil.move(str(f), str(od_i / f.name))
        lbl = val_lbl / (f.stem + ".txt")
        if lbl.exists():
            shutil.move(str(lbl), str(od_l / lbl.name))
        n_moved += 1

    n_val = sum(1 for p in val_img.glob("*") if p.is_file())
    LOG.info("切分完成：train 保持不变；val=%d test=%d（从原 validation 切出 %d 张）",
             n_val, n_moved, len(test_files))
