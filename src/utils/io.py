"""文件与路径工具。"""
from __future__ import annotations

import os
from pathlib import Path


def ensure_dir(path: str | os.PathLike) -> Path:
    """创建目录（含父级），幂等返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(dir_path: str | os.PathLike,
                exts: tuple[str, ...] = (".jpg", ".png", ".bmp", ".jpeg")) -> list[Path]:
    """递归列出目录下所有图片，按文件名排序。"""
    p = Path(dir_path)
    if not p.exists():
        return []
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
