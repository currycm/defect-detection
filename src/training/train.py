"""YOLOv8 训练入口。"""
from __future__ import annotations

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
    with open(hyp_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
