"""实时流式检测演示 / 实机运行入口。

用法：
    # 摄像头（0 通常是内置摄像头），按 q 退出
    python scripts/stream_demo.py 0

    # 视频文件
    python scripts/stream_demo.py D:/videos/steel.mp4

    # 无摄像头也能跑：用 test 集做合成流（默认）
    python scripts/stream_demo.py

    # 纯日志模式（无窗口，跑 10 秒自动结束，适合验证/服务器）
    python scripts/stream_demo.py --no-display --seconds 10

    # 调参：目标帧率 15、每 0.2s 才推理一次
    python scripts/stream_demo.py 0 --fps 15 --interval 0.2

观测要点：
  - 只有「稳定状态发生变化」时才会打印 [状态变化] 事件（去抖生效）
  - 每秒打印一次 读帧/推理/复用率，可直观看到没有重复推理
"""
import argparse
import os
import time

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

import cv2

from src.constants import CLASS_NAMES
from src.inference.onnx_predictor import ONNXDefectPredictor
from src.inference.stream import StreamDetector
from src.utils import paths
from src.utils.imageio import draw_detections

ONNX_PATH = str(paths.ONNX_PATH)
TEST_DIR = str(paths.PROCESSED_DIR / "images" / "test")
SNAP_DIR = str(paths.RUNS_DIR / "stream_snapshots")


def summary_of(dets: list[dict]) -> str:
    stat: dict[str, int] = {}
    for d in dets:
        stat[d["name"]] = stat.get(d["name"], 0) + 1
    if not stat:
        return "空"
    return "、".join(f"{k}×{v}" for k, v in sorted(stat.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=f"synthetic:{TEST_DIR}",
                    help="摄像头索引 / 视频文件 / 流地址 / synthetic:<路径>（默认 test 集）")
    ap.add_argument("--fps", type=float, default=15.0, help="目标处理帧率")
    ap.add_argument("--interval", type=float, default=0.1,
                    help="两次推理之间的最小间隔（秒），间隔内复用上次结果")
    ap.add_argument("--stable", type=int, default=3, help="连续多少帧相同才确认状态")
    ap.add_argument("--hold", type=int, default=15,
                    help="合成源每张图连续播多少帧（模拟静止场景；默认 15）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--no-display", action="store_true", help="不弹窗口，只打日志")
    ap.add_argument("--seconds", type=float, default=0.0, help="运行多少秒后自动停止（0=手动）")
    ap.add_argument("--save-snapshots", action="store_true", help="状态变化时存一张快照")
    args = ap.parse_args()

    # 数字字符串转成摄像头索引
    src = int(args.source) if args.source.isdigit() else args.source

    predictor = ONNXDefectPredictor(ONNX_PATH, CLASS_NAMES, imgsz=640)
    events: list[str] = []
    snap_i = {"n": 0}

    def on_change(dets, frame):
        ts = time.strftime("%H:%M:%S")
        msg = f"[状态变化 {ts}] {summary_of(dets)}"
        print(msg, flush=True)
        events.append(msg)
        if args.save_snapshots:
            os.makedirs(SNAP_DIR, exist_ok=True)
            snap_i["n"] += 1
            path = os.path.join(SNAP_DIR, f"snap_{snap_i['n']:03d}.jpg")
            ok, buf = cv2.imencode(".jpg", draw_detections(frame, dets))
            if ok:
                from pathlib import Path
                Path(path).write_bytes(buf.tobytes())

    det = StreamDetector(predictor, src,
                         target_fps=args.fps, detect_interval=args.interval,
                         stable_frames=args.stable, conf=args.conf, iou=args.iou,
                         loop=True, hold=args.hold,
                         on_change=on_change, draw=draw_detections)

    print(f"输入源: {src}")
    print(f"目标帧率 {args.fps} fps ｜ 推理间隔 {args.interval}s ｜ 去抖帧数 {args.stable} "
          f"｜ 合成源持帧 {args.hold}")
    print("启动中…（窗口模式按 q 退出；Ctrl+C 亦可）\n")

    det.start()
    t_start = time.time()
    last_report = t_start
    prev = det.stats()
    try:
        while det.running:
            time.sleep(0.1)

            if args.no_display:
                frame = None
            else:
                frame, _ = det.latest()
            if frame is not None:
                cv2.imshow("Defect Stream (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - last_report >= 1.0:
                st = det.stats()
                d_read = st["frames_read"] - prev["frames_read"]
                d_infer = st["frames_inferred"] - prev["frames_inferred"]
                print(f"  {now - t_start:5.1f}s  处理 {d_read:3d} 帧/s  "
                      f"推理 {d_infer:2d} 次/s  实时检出 {st['latest_detections']} 框  "
                      f"累计状态变化 {st['stabilizer_emits']}", flush=True)
                last_report, prev = now, st

            if args.seconds > 0 and now - t_start >= args.seconds:
                print("\n到达设定时长，自动停止。")
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，停止中…")
    finally:
        det.stop()                              # 停止线程 + 释放帧源
        if not args.no_display:
            cv2.destroyAllWindows()             # 释放 GUI 资源

    st = det.stats()
    print("\n===== 汇总 =====")
    print(f"读帧总数      : {st['frames_read']}")
    print(f"实际推理次数  : {st['frames_inferred']}")
    print(f"推理占比      : {st['infer_ratio']:.1%}  （越低说明复用越充分）")
    print(f"状态变化次数  : {st['stabilizer_emits']}")
    print(f"线程已结束    : {not det.running}")
    print(f"帧源已释放    : {det.source is None}")
    if events:
        print("状态变化记录:")
        for e in events[:10]:
            print("  " + e)
        if len(events) > 10:
            print(f"  …（共 {len(events)} 条）")


if __name__ == "__main__":
    main()
