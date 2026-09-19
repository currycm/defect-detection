"""深入检查 NEU-DET 目录结构与标注文件真实格式（Step 2 诊断脚本）。

用法：python scripts/verify_raw.py
"""
from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.utils import paths

BASE = paths.RAW_DIR / "NEU-DET"

print("=== NEU-DET 目录树（前 3 层）===")


def walk(d, depth=0):
    if depth > 2:
        return
    for c in sorted(d.iterdir()):
        print("  " * depth + c.name + ("/" if c.is_dir() else ""))
        if c.is_dir():
            walk(c, depth + 1)


if not BASE.exists():
    raise SystemExit(f"未找到原始数据集目录: {BASE}\n请先把 NEU-DET 放到 data/raw/ 下")

walk(BASE)

print("\n=== 各类型标注文件数量 ===")
for ext in ("*.txt", "*.xml", "*.json", "*.csv", "*.yaml", "*.yml"):
    print(f"  {ext}: {len(list(BASE.rglob(ext)))}")

print("\n=== 抽样查看标注内容 ===")
for ext in ("*.txt", "*.xml", "*.json"):
    fs = sorted(BASE.rglob(ext))
    if fs:
        print(f"\n--- 样本 {fs[0].relative_to(BASE)} 前 5 行 ---")
        for ln in fs[0].read_text(encoding="utf-8", errors="ignore").splitlines()[:5]:
            print("   ", repr(ln))
        break
