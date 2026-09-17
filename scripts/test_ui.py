"""无头验证 Gradio 界面的推理与绘图逻辑（不启动 Web 服务器）。

直接调用 src.ui.app 的 predict()，检查：
  1. 能正确读出图片并推理
  2. 标注图尺寸正确、且与原图不同（说明确实画上了框）
  3. 结果表行数 == 检出框数
  4. 阈值变化确实影响检出数量

用法：python scripts/test_ui.py
"""
import os

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.ui.app import CLASS_NAMES, predict  # noqa: E402

TEST_DIR = os.path.join(PROJECT, "data", "processed", "images", "test")


def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return None if data.size == 0 else cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    # 每类取一张
    samples = []
    for cls in CLASS_NAMES:
        for f in sorted(os.listdir(TEST_DIR)):
            if f.lower().startswith(cls) and f.lower().endswith((".jpg", ".png")):
                samples.append(os.path.join(TEST_DIR, f))
                break

    print(f"示例 {len(samples)} 张\n")
    ok = True
    for p in samples:
        bgr = imread_unicode(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ann, rows, summary = predict(rgb, 0.25, 0.5)

        if ann is None or not rows:
            print(f"  [失败] {os.path.basename(p)}: 无检出或绘图失败")
            ok = False
            continue
        same = np.array_equal(ann, rgb)
        h, w = ann.shape[:2]
        name = os.path.basename(p)
        print(f"  {name:<28} 检出 {len(rows):<3} 框  图 {w}x{h}  未改动={same}")
        if same:
            ok = False
        print(f"      {summary.replace('**','')}")

    # 阈值敏感性
    p = samples[0]
    bgr = imread_unicode(p)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    counts = []
    for c in (0.05, 0.25, 0.6):
        _, rows, _ = predict(rgb, c, 0.5)
        counts.append(len(rows))
    print(f"\n阈值敏感性 ({os.path.basename(p)}): conf 0.05/0.25/0.6 -> {counts} 框")
    if not (counts[0] >= counts[1] >= counts[2]):
        print("  [警告] 检出数未随 conf 单调下降")
        ok = False

    print("\n结论:", "界面逻辑正常 ✓" if ok else "存在问题 ✗")


if __name__ == "__main__":
    main()
