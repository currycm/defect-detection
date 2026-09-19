"""断点续跑工具。

训练被中断（内存压力 / 进程被杀 / 手工停止）后，从 last.pt 恢复，
继续剩下的 epoch，optimizer 状态与学习率调度一并恢复。

用法：
    python scripts/resume.py exp_aug640
    python scripts/resume.py exp_aug640 2      # 指定 workers（省内存）

背景：本机在这一 step 出现过训练在第 30 轮被静默杀死的情况，根因是
① imgsz=640 + workers=8 的数据加载内存峰值很高，而系统可用内存不足；
② Anaconda 与 torch 各带一份 libiomp5md.dll，OpenMP 重复初始化可能原生崩溃。
本脚本允许用更小的 workers 续跑，降低再次被杀的概率。
"""
import os
import sys

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from ultralytics import YOLO


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/resume.py <run_name> [workers]")
        sys.exit(1)

    run_name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    run_dir = os.path.join(PROJECT, "runs", "detect", "runs", run_name)
    last_pt = os.path.join(run_dir, "weights", "last.pt")
    if not os.path.exists(last_pt):
        raise FileNotFoundError(f"没有找到断点文件: {last_pt}")

    # 已完成的轮数
    results_csv = os.path.join(run_dir, "results.csv")
    done = 0
    if os.path.exists(results_csv):
        with open(results_csv, encoding="utf-8", errors="ignore") as f:
            done = max(len(f.read().splitlines()) - 1, 0)
    print(f"从 {run_name} 续跑：已完成 {done} 轮，workers={workers}")

    model = YOLO(last_pt)
    model.train(resume=True, workers=workers)


if __name__ == "__main__":
    main()
