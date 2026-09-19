"""模型加载与导出（隔离 ultralytics API）。

`ultralytics` 只做「惰性导入」：本模块的导入不应把 torch/ultralytics
拉起来（否则任何 `import src.models` 都要等好几秒，单测也被拖慢）。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ultralytics import YOLO


def load_model(weights: str | Path) -> YOLO:
    """加载 YOLO 模型（预训练权重或训练产出权重）。"""
    from ultralytics import YOLO

    return YOLO(str(weights))


def export_onnx(model, out_path: str | Path | None = None,
                imgsz: int = 640, dynamic: bool = False,
                simplify: bool = False) -> Path:
    """导出 ONNX 并（可选）搬到 `out_path`，返回最终文件路径。

    修复记录（L7）：原实现给 `model.export()` 传了 `path=`，但 ultralytics
    的 `export()` **没有** 这个参数（与历史上踩过的 `hyp=` 同类坑），一调用
    就 TypeError。它只**返回**导出文件的路径，需要改位置得自己搬。

    `dynamic=False` 固定输入尺寸，部署更简单也更快；`simplify=True` 需要
    额外安装 onnxsim，未装时会报错，因此默认关闭。
    """
    exported = Path(model.export(format="onnx", imgsz=imgsz,
                                 dynamic=dynamic, simplify=simplify))
    if out_path is None:
        return exported

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out.resolve():
        shutil.copy2(exported, out)
    return out
