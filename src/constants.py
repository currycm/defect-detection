"""项目级常量（唯一真源）。

类别表以前在 convert / api / ui / stream_demo / 多个 scripts 里各写一份，
改动时极易漏改（顺序即 class_id，写错会静默错标）。统一到这里。
"""
from __future__ import annotations

# 顺序即 class_id，勿随意调整
CLASS_NAMES: list[str] = [
    "crazing",          # 0 裂纹
    "inclusion",        # 1 夹杂
    "patches",          # 2 斑块
    "pitted_surface",   # 3 麻点
    "rolled-in_scale",  # 4 轧入氧化皮
    "scratches",        # 5 划痕
]

CLASS_NAMES_ZH: dict[str, str] = {
    "crazing": "裂纹",
    "inclusion": "夹杂",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled-in_scale": "轧入氧化皮",
    "scratches": "划痕",
}

NUM_CLASSES: int = len(CLASS_NAMES)
