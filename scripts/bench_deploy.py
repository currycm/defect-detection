"""产线部署就绪度 · 性能基准脚本（对应 docs/deployment-readiness-2026-09-17.md §维度1）。

把「端到端延迟未测」这个 P0 缺口变成可复现的数字：本脚本把推理链路
**按段拆开**逐段计时，并给出延迟分位数、吞吐上限与长稳漂移。

为什么不能只看单帧总耗时：
  设计/排产用的是「节拍」，而节拍 = 采集 + 传输 + 预处理 + 推理 + 后处理 + 决策。
  只报一个总毫秒数，就无法判断优化该往哪一段投。
  所以这里逐段分开测，并明确标出**哪一段是当前瓶颈**。

测的四段（对应线上链路）：
  1) preprocess : letterbox 缩放 + 补边 + BGR->RGB + 归一化 + HWC->CHW
  2) inference  : ONNXRuntime session.run（纯模型前向）
  3) postprocess: 解码 + 逐类 NMS + 坐标还原裁剪
  4) encode     : JPEG 编码（MJPEG 推流每帧都要做，常被忽略）
  另测 total    : 上述之和（不含网络往返与相机采集）

用法：
    # 默认 200 次，用 test 集真实图片
    D:/Anaconda/python.exe scripts/bench_deploy.py

    # 指定次数与图片来源
    D:/Anaconda/python.exe scripts/bench_deploy.py --n 500 --source data/processed/images/test

    # 指定图片尺寸（模拟产线高分辨率相机，验证缩放放大后的开销变化）
    D:/Anaconda/python.exe scripts/bench_deploy.py --resize 1920x1080

    # 长稳测试：连续跑 4 小时，每 10 分钟报一次分位漂移
    D:/Anaconda/python.exe scripts/bench_deploy.py --soak-minutes 240

    # 线程数扫描（在目标工控机上标定最优线程数）
    D:/Anaconda/python.exe scripts/bench_deploy.py --threads-sweep

注意：
  - 本脚本**只测推理链路**，不含相机采集、网络传输、PLC 通信。
    端到端节拍需在产线现场用打点或抓包方式另行测量。
  - 计时结果受同机其他负载影响极大，测量前请确认机器空闲
    （本机历史上出现过其他重负载干扰性能测量的情况）。
  - 涉 cv2/onnxruntime，Windows 下建议设 KMP_DUPLICATE_LIB_OK=TRUE。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

import cv2
import numpy as np

from src.constants import CLASS_NAMES
from src.inference.onnx_predictor import ONNXDefectPredictor
from src.utils import paths
from src.utils.imageio import encode_jpeg, load_image

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(source: str, limit: int = 40) -> list[Path]:
    """收集一批测试图片（目录或单文件）。"""
    p = Path(source)
    if not p.is_absolute():
        p = PROJECT / p
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)
    elif p.is_file():
        files = [p]
    else:
        raise SystemExit(f"图片来源不存在: {p}")
    if not files:
        raise SystemExit(f"目录下没有图片: {p}")
    return files[:limit]


def percentile(values: list[float], q: float) -> float:
    """线性插值分位数（不依赖 numpy 版本差异）。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def fmt_stats(name: str, xs: list[float]) -> dict:
    return {
        "段": name,
        "P50_ms": round(percentile(xs, 0.50), 2),
        "P95_ms": round(percentile(xs, 0.95), 2),
        "P99_ms": round(percentile(xs, 0.99), 2),
        "mean_ms": round(statistics.fmean(xs), 2),
        "max_ms": round(max(xs), 2),
    }


def sync_barrier() -> None:
    """让尽可能多的时间花在推理上、减少线程/调度抖动对单次计时的污染。

    说明：Python 没有跨库统一的 GPU/CPU 同步原语，onnxruntime 的 run() 本身
    是同步阻塞的，所以这里只需一次极短的 sleep 把控制权让给调度器，
    避免连续循环里计时被前一轮的后台线程干扰。
    """
    time.sleep(0)


def bench_once(
    predictor: ONNXDefectPredictor,
    img: np.ndarray,
    *,
    conf: float,
    iou: float,
    measure_encode: bool = True,
) -> dict:
    """对单帧完整走一遍链路，逐段计时。返回各段耗时（毫秒）。"""
    t0 = time.perf_counter()

    # --- 1) preprocess ---
    t = time.perf_counter()
    box = predictor._letterbox(img)
    x = predictor._preprocess(box)
    t_pre = (time.perf_counter() - t) * 1000

    # --- 2) inference ---
    sync_barrier()
    t = time.perf_counter()
    out = predictor.session.run(None, {predictor.input_name: x})[0]
    t_inf = (time.perf_counter() - t) * 1000

    # --- 3) postprocess ---
    h, w = img.shape[:2]
    t = time.perf_counter()
    dets = predictor._postprocess(out, box[1], box[2], conf, iou, w, h)
    t_post = (time.perf_counter() - t) * 1000

    # --- 4) encode（MJPEG 推流每帧都要做） ---
    t_enc = 0.0
    if measure_encode:
        t = time.perf_counter()
        encode_jpeg(img)
        t_enc = (time.perf_counter() - t) * 1000

    t_total = (time.perf_counter() - t0) * 1000
    return {"pre": t_pre, "inf": t_inf, "post": t_post, "enc": t_enc,
            "total": t_total, "n_dets": len(dets)}


def run_bench(predictor, imgs, n, conf, iou, label=""):
    """跑 n 次基准，返回各段耗时列表。"""
    acc = {k: [] for k in ("pre", "inf", "post", "enc", "total")}
    n_dets = 0
    warmup = min(10, max(3, n // 20))
    for i in range(n + warmup):
        img = imgs[i % len(imgs)]
        r = bench_once(predictor, img, conf=conf, iou=iou)
        if i < warmup:
            continue
        for k, series in acc.items():
            series.append(r[k])
        n_dets += r["n_dets"]
    if label:
        print(f"  [{label}] 完成 {n} 次")
    return acc, n_dets / max(n, 1)


def report(acc: dict, n_dets_avg: float, img_desc: str, threads: int) -> dict:
    rows = [
        fmt_stats("1.预处理(letterbox+归一化)", acc["pre"]),
        fmt_stats("2.推理(session.run)", acc["inf"]),
        fmt_stats("3.后处理(解码+NMS)", acc["post"]),
        fmt_stats("4.JPEG编码(推流用)", acc["enc"]),
        fmt_stats("合计(不含采集/网络)", acc["total"]),
    ]
    print(f"\n{'段':<28}{'P50':>9}{'P95':>9}{'P99':>9}{'mean':>9}{'max':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['段']:<28}{r['P50_ms']:>9.2f}{r['P95_ms']:>9.2f}"
              f"{r['P99_ms']:>9.2f}{r['mean_ms']:>9.2f}{r['max_ms']:>9.2f}")

    p50_total = percentile(acc["total"], 0.50)
    p99_total = percentile(acc["total"], 0.99)
    fps = 1000.0 / p50_total if p50_total > 0 else 0.0
    infer_share = percentile(acc["inf"], 0.50) / p50_total if p50_total > 0 else 0.0

    print(f"\n图片: {img_desc}  ｜  intra_op_num_threads={threads}  ｜  平均检出 {n_dets_avg:.2f} 框/帧")
    print(f"单帧总耗时 P50 {p50_total:.2f} ms ｜ P99 {p99_total:.2f} ms")
    print(f"推理占比 {infer_share:.1%}（其余为前后处理与编码）")
    print(f"单线程串行吞吐上限 ≈ {fps:.1f} FPS")

    # ---- 对照报告里的验收门槛 ----
    print("\n对照产线验收门槛（高速连续产线建议值）:")
    g1 = p99_total < 50.0
    print(f"  [{'通过' if g1 else '未通过'}] 单帧 P99 < 50 ms        （实测 {p99_total:.2f} ms）")
    print("  [--] 端到端 P99 < 50 ms        （本脚本不含采集/网络，需现场实测）")
    g2 = infer_share <= 0.5
    print(f"  [{'通过' if g2 else '未通过'}] 推理占比 ≤ 50%          （实测 {infer_share:.1%}）")

    return {
        "p50_total_ms": round(p50_total, 2),
        "p99_total_ms": round(p99_total, 2),
        "serial_fps": round(fps, 1),
        "infer_share": round(infer_share, 3),
        "detail": rows,
    }


def soak(predictor, imgs, minutes: float, conf, iou, report_every: float = 10.0):
    """长稳测试：连续跑指定时长，按窗口报分位漂移（对应验收项 P2-#19）。"""
    print(f"\n=== 长稳测试 {minutes:.0f} 分钟 ===")
    t_end = time.time() + minutes * 60
    first_window = None
    rss0 = _rss_mb()
    windows = 0
    while time.time() < t_end:
        w_end = min(time.time() + report_every * 60, t_end)
        acc = {k: [] for k in ("pre", "inf", "post", "enc", "total")}
        while time.time() < w_end:
            img = imgs[len(acc["total"]) % len(imgs)]
            r = bench_once(predictor, img, conf=conf, iou=iou, measure_encode=False)
            for k, series in acc.items():
                series.append(r[k])
        p95 = percentile(acc["total"], 0.95)
        rss = _rss_mb()
        windows += 1
        if first_window is None:
            first_window = p95
            print(f"  窗口1  P95 {p95:.2f} ms  RSS {rss:.0f} MB  (基准)")
        else:
            drift = (p95 - first_window) / first_window * 100 if first_window else 0.0
            print(f"  窗口{windows}  P95 {p95:.2f} ms  RSS {rss:.0f} MB  "
                  f"漂移 {drift:+.1f}%")
            if abs(drift) > 10.0:
                print(f"    ⚠ 延迟漂移超过 10% 门槛（首窗 P95 {first_window:.2f} ms）")
    rss1 = _rss_mb()
    growth = (rss1 - rss0) / rss0 * 100 if rss0 else 0.0
    print(f"\n长稳结束：RSS {rss0:.0f} -> {rss1:.0f} MB（增长 {growth:+.1f}%，门槛 ≤ 5%）")
    print(f"线程数 {_thread_count()}（应保持稳定，不随时间增长）")


def _rss_mb() -> float:
    """当前进程 RSS（MB）。psutil 不可用时返回 0（长稳内存检查会自动跳过）。"""
    try:
        import psutil
    except ImportError:
        return 0.0
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024
    except (OSError, AttributeError):
        return 0.0


def _thread_count() -> int:
    """当前进程线程数。psutil 不可用时回退到 threading.active_count()。"""
    try:
        import psutil
    except ImportError:
        import threading
        return threading.active_count()
    try:
        return psutil.Process().num_threads()
    except (OSError, AttributeError):
        import threading
        return threading.active_count()


def main() -> int:
    ap = argparse.ArgumentParser(description="产线部署性能基准（逐段计时）")
    ap.add_argument("--source", default=str(paths.PROCESSED_DIR / "images" / "test"),
                    help="图片目录或单文件（默认 test 集）")
    ap.add_argument("--onnx", default=str(paths.ONNX_PATH))
    ap.add_argument("--n", type=int, default=200, help="正式计时次数（不含预热）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU intra_op_num_threads（默认 4，开发机标定值）")
    ap.add_argument("--resize", default="", help="把测试图缩放到 WxH，如 1920x1080")
    ap.add_argument("--soak-minutes", type=float, default=0.0,
                    help="长稳测试时长（分钟），如 240 = 4 小时")
    ap.add_argument("--threads-sweep", action="store_true",
                    help="扫描线程数 1/2/4/物理核，找出目标机器最优值")
    ap.add_argument("--json-out", default="", help="把结果写入 JSON 文件")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = PROJECT / onnx_path
    if not onnx_path.exists():
        raise SystemExit(f"未找到 ONNX 模型: {onnx_path}\n请先运行 python scripts/export.py")

    files = collect_images(args.source)
    imgs = []
    for f in files:
        im = load_image(f)          # 中文路径安全
        if im is None:
            continue
        imgs.append(im)
    if not imgs:
        raise SystemExit("所有图片读取失败")

    if args.resize:
        try:
            w, h = (int(v) for v in args.resize.lower().split("x"))
        except ValueError as e:
            raise SystemExit("--resize 格式应为 WxH，例如 1920x1080") from e
        imgs = [cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR) for im in imgs]
        img_desc = f"{len(imgs)} 张 @ {w}x{h}（已缩放）"
    else:
        h0, w0 = imgs[0].shape[:2]
        img_desc = f"{len(imgs)} 张 @ {w0}x{h0}（原始）"

    print("=" * 74)
    print("产线部署性能基准 · 逐段计时")
    print("=" * 74)
    print(f"模型    : {onnx_path.name}  (imgsz={args.imgsz})")
    print(f"图片    : {img_desc}")
    print(f"配置    : conf={args.conf} iou={args.iou} n={args.n}")
    print("说明    : 仅测推理链路，不含相机采集/网络传输/PLC 通信")
    print()

    results: dict = {"model": onnx_path.name, "images": img_desc}

    if args.threads_sweep:
        import os
        physical = os.cpu_count() or 4
        cands = sorted({1, 2, 4, physical})
        print(f"=== 线程数扫描（物理核 {physical}）===")
        sweep = []
        for th in cands:
            pred = ONNXDefectPredictor(str(onnx_path), CLASS_NAMES, imgsz=args.imgsz,
                                       conf=args.conf, iou=args.iou,
                                       intra_op_num_threads=th)
            acc, nd = run_bench(pred, imgs, args.n, args.conf, args.iou,
                                label=f"threads={th}")
            p50 = percentile(acc["total"], 0.50)
            p99 = percentile(acc["total"], 0.99)
            sweep.append({"threads": th, "p50_ms": round(p50, 2),
                          "p99_ms": round(p99, 2)})
            print(f"    threads={th:<3} P50 {p50:7.2f} ms   P99 {p99:7.2f} ms")
        best = min(sweep, key=lambda r: r["p50_ms"])
        print(f"\n  → 最优线程数: {best['threads']}（P50 {best['p50_ms']} ms）")
        print("  → 请把该值写入产线部署配置（DEFECT_ONNX_THREADS）")
        results["threads_sweep"] = sweep
        results["best_threads"] = best["threads"]

    predictor = ONNXDefectPredictor(str(onnx_path), CLASS_NAMES, imgsz=args.imgsz,
                                    conf=args.conf, iou=args.iou,
                                    intra_op_num_threads=args.threads)
    print(f"providers: {predictor.providers}\n")
    acc, nd = run_bench(predictor, imgs, args.n, args.conf, args.iou)
    results["bench"] = report(acc, nd, img_desc, args.threads)

    if args.soak_minutes > 0:
        soak(predictor, imgs, args.soak_minutes, args.conf, args.iou)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = PROJECT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n结果已写入: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
