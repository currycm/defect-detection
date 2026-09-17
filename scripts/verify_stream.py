"""实时流式检测的无头自动化验证（需求⑤）。

不需要摄像头，用「合成帧源」循环喂图，对四条需求逐项断言：
  A. 限频生效：推理次数远小于读帧次数（间隔内的帧被复用，没有重复处理）
  B. 去抖生效：同一画面连续输入只回调 1 次；画面切换后才再次回调
  C. 回调只在状态变化时触发：回调次数 == 稳定状态变化次数
  D. 生命周期干净：重复 start 不产生多个线程；stop 后线程结束、可重复 stop、无线程残留

用法：python scripts/verify_stream.py
"""
import os
import sys
import threading
import time

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

import cv2  # noqa: E402

from src.constants import CLASS_NAMES  # noqa: E402
from src.inference.onnx_predictor import ONNXDefectPredictor  # noqa: E402
from src.inference.stream import StreamDetector, SyntheticSource  # noqa: E402
from src.utils import paths  # noqa: E402

ONNX_PATH = str(paths.ONNX_PATH)
TEST_DIR = str(paths.PROCESSED_DIR / "images" / "test")

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def simple_draw(frame, dets):
    """极简画框（不依赖 gradio），仅用于确认输出帧确实被加工过。"""
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out


def pick(cls: str) -> str:
    for f in sorted(os.listdir(TEST_DIR)):
        if f.lower().startswith(cls) and f.lower().endswith((".jpg", ".png")):
            return os.path.join(TEST_DIR, f)
    raise FileNotFoundError(cls)


def main():
    predictor = ONNXDefectPredictor(ONNX_PATH, CLASS_NAMES, imgsz=640)

    print("加载模型完成，开始验证\n")

    # ---------------- A. 限频：不重复处理 ----------------
    print("A. 限频（避免重复推理）")
    src = SyntheticSource([pick("crazing")], repeat=1_000_000)
    det = StreamDetector(predictor, src, target_fps=200.0,
                         detect_interval=0.25, stable_frames=1,
                         draw=simple_draw)
    det.start()
    time.sleep(2.0)
    st = det.stats()
    det.stop()
    expected_infer = 2.0 / 0.25 + 2   # 允许少量边界误差
    check("推理次数受 detect_interval 限制",
          st["frames_inferred"] <= expected_infer,
          f"2.0s 内推理 {st['frames_inferred']} 次（上限≈{expected_infer:.0f}）")
    check("大部分帧被复用而非重复推理",
          st["infer_ratio"] < 0.2,
          f"读帧 {st['frames_read']}，推理占比 {st['infer_ratio']:.1%}")
    check("循环帧率受 target_fps 约束",
          st["frames_read"] / 2.0 <= 200 * 1.3,
          f"实测约 {st['frames_read'] / 2.0:.0f} fps")

    # ---------------- B. 去抖：同一画面只回调一次 ----------------
    print("\nB. 去抖（同一状态不重复回调）")
    hits_b: list = []
    src = SyntheticSource([pick("crazing")], repeat=1_000_000)
    det = StreamDetector(predictor, src, target_fps=60.0, detect_interval=0.0,
                         stable_frames=3, on_change=lambda d, f: hits_b.append(len(d)),
                         draw=simple_draw)
    det.start()
    time.sleep(1.5)
    st = det.stats()
    det.stop()
    check("同一画面持续输入只回调 1 次",
          st["stabilizer_emits"] == 1,
          f"回调 {len(hits_b)} 次 / 推理 {st['frames_inferred']} 帧")

    # ---------------- C. 状态切换时才回调 ----------------
    print("\nC. 状态变化才回调")
    hits_c: list = []
    # 前 12 帧 crazing，随后 12 帧 inclusion（各能稳定 3 帧以上）
    imgs = [pick("crazing")] * 12 + [pick("inclusion")] * 12
    src = SyntheticSource(imgs, repeat=1)
    det = StreamDetector(predictor, src, target_fps=60.0, detect_interval=0.0,
                         stable_frames=3, on_change=lambda d, f: hits_c.append(len(d)),
                         draw=simple_draw)
    det.start()
    # 等合成源自然播完（线程会自动结束），而不是立刻 stop —— 否则帧还没被处理
    deadline = time.time() + 20
    while det.running and time.time() < deadline:
        time.sleep(0.1)
    det.stop()
    time.sleep(0.2)
    check("一次状态切换只多出一次回调",
          len(hits_c) == 2,
          f"回调 {len(hits_c)} 次（crazing -> inclusion 各 1 次）")

    # ---------------- D. 生命周期与资源释放 ----------------
    print("\nD. 生命周期与资源")
    before = sum(1 for t in threading.enumerate() if t.name == "stream-detector")
    src = SyntheticSource([pick("scratches")], repeat=1_000_000)
    det = StreamDetector(predictor, src, target_fps=30.0, detect_interval=0.1,
                         draw=simple_draw)
    det.start()
    det.start()   # 幂等：不应再开第二个线程
    time.sleep(0.6)
    during = sum(1 for t in threading.enumerate() if t.name == "stream-detector")
    check("重复 start 只存在一个工作线程",
          during - before == 1, f"新增线程 {during - before} 个")

    det.pause()
    time.sleep(0.3)
    paused_stats = det.stats()
    time.sleep(0.3)
    check("暂停后不再读取/推理新帧",
          det.stats()["frames_read"] - paused_stats["frames_read"] <= 1,
          f"暂停期间读帧增量 {det.stats()['frames_read'] - paused_stats['frames_read']}")
    det.resume()
    time.sleep(0.3)
    check("恢复后继续读取新帧",
          det.stats()["frames_read"] > paused_stats["frames_read"],
          f"恢复后累计读帧 {det.stats()['frames_read']}")

    det.stop()
    wait_deadline = time.time() + 3
    while det.running and time.time() < wait_deadline:
        time.sleep(0.05)
    after = sum(1 for t in threading.enumerate() if t.name == "stream-detector")
    check("stop 后线程已结束", not det.running)
    check("无线程残留", after == before, f"残留 {after - before} 个")
    check("帧源已释放", det.source is None)

    det.stop()   # 幂等
    check("重复 stop 不报错", True)

    # ---------------- 汇总 ----------------
    failed = [n for n, ok in results if not ok]
    print("\n" + "=" * 56)
    print(f"通过 {len(results) - len(failed)}/{len(results)} 项")
    if failed:
        print("失败项:")
        for n in failed:
            print("  -", n)
        sys.exit(1)
    print("实时流式检测逻辑验证全部通过 ✓")


if __name__ == "__main__":
    main()
