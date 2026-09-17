"""Step 5 离线弱类增强入口。

用法：
    python scripts/augment.py
"""
from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.data.augment import augment_weak_classes  # noqa: E402
from src.utils import paths  # noqa: E402

if __name__ == "__main__":
    res = augment_weak_classes(str(paths.PROCESSED_DIR))
    print("Step5 新增弱类样本:", res)
