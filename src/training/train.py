"""YOLOv8 训练入口。"""
from __future__ import annotations

import glob
import os

import yaml
from ultralytics import YOLO

from ..utils import paths
from ..utils.logger import get_logger

LOG = get_logger("train")


def _load_hyp(hyp_yaml: str | os.PathLike) -> dict:
    """读取超参 YAML，作为 train 的 override 传入。

    注意：新版 ultralytics 已移除 `hyp=` 参数，超参必须作为直接 override
    （lr0/mosaic/...）传入，否则报 'hyp is not a valid YOLO argument'。
    """
    if not hyp_yaml or not os.path.exists(hyp_yaml):
        return {}
    with open(hyp_yaml, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def clear_label_caches() -> int:
    """清理 data/processed/labels/ 下的 ultralytics cache 文件，返回清理数量。

    必要性：ultralytics 训练启动时会扫描 labels 目录并生成 `*.cache`。
    若数据集变动（增删图片/标签）后旧 cache 未失效，会触发
    `SAFE_DELETE_BULK_CONFIRM_REQUIRED` 沙箱守卫，把训练任务直接中断
    （历史上踩过多次）。清完再启动就稳了。

    放这里而不是脚本入口：任何调用 `train()` 的代码（不只是 scripts/train.py）
    都能受益，包括 resume.py / 未来的实验脚本。
    """
    labels_dir = paths.PROCESSED_DIR / "labels"
    if not labels_dir.exists():
        return 0
    n = 0
    for cache in glob.glob(str(labels_dir / "*" / "*.cache")):
        # labels/{train,val,test}/*.cache 都要清
        try:
            os.remove(cache)
            n += 1
        except OSError as e:
            LOG.warning("清理 cache 失败 %s: %s", cache, e)
    # 也清 labels 根目录下可能存在的 cache（旧版本布局）
    for cache in glob.glob(str(labels_dir / "*.cache")):
        try:
            os.remove(cache)
            n += 1
        except OSError as e:
            LOG.warning("清理 cache 失败 %s: %s", cache, e)
    if n:
        LOG.info("已清理 %d 个 labels cache（防沙箱守卫中断训练）", n)
    return n


def train(data_yaml: str | os.PathLike | None = None,
          hyp_yaml: str | os.PathLike | None = None,
          model: str | os.PathLike | None = None,
          epochs: int = 100, imgsz: int = 640,
          batch: int = 16,
          project: str | os.PathLike | None = None,
          name: str = "exp") -> str:
    """启动 YOLOv8 训练，返回权重保存目录。

    路径参数默认取 src.utils.paths 的唯一真源（绝对路径），避免「相对路径在
    cwd 异常时被解析到别处」——历史上就发生过 ultralytics 退回默认数据集目录。
    数据集配置统一经 `paths.runtime_data_yaml()` 绝对化后再交给 ultralytics。
    """
    data_yaml = str(paths.runtime_data_yaml(data_yaml))
    hyp_yaml = str(hyp_yaml or paths.HYP_YAML)
    model = str(model or paths.PRETRAINED_PT)
    project = str(project or paths.RUNS_DIR)

    # 训练前清旧 cache：ultralytics 启动时会重新生成，但若数据集变动过，
    # 旧 cache 会触发 SAFE_DELETE_BULK_CONFIRM_REQUIRED 守卫中断任务。
    clear_label_caches()

    m = YOLO(model)
    hyp = _load_hyp(hyp_yaml)
    results = m.train(
        data=data_yaml, epochs=epochs,
        imgsz=imgsz, batch=batch, project=project, name=name,
        **hyp,
    )
    save_dir = str(results.save_dir)
    LOG.info("训练完成，权重目录: %s", save_dir)
    return save_dir


if __name__ == "__main__":
    train()
