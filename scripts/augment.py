"""Step 5 离线弱类增强入口。

用法：
    python scripts/augment.py
"""
from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.data.augment import augment_weak_classes
from src.utils import paths
from src.utils.logger import get_logger

LOG = get_logger("augment")

if __name__ == "__main__":
    res = augment_weak_classes(str(paths.PROCESSED_DIR))
    LOG.info("Step5 新增弱类样本: %s", res)
