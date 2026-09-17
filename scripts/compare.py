"""Step5 对比脚本：baseline(exp, imgsz=640) vs 弱类增强版(exp_aug-2, imgsz=320)。

在同一验证集(val)上跑两个 best.pt，输出整体与逐类 P/R/mAP50/mAP50-95，
用于判断「弱类定向增强 + 降分辨率」到底是赚是亏。

注意：
- 必须以真实脚本文件运行（不能用 stdin 管道），否则 dataloader 的
  multiprocessing spawn 会尝试 re-import <stdin> 而报 OSError(22)。
- workers=0 关闭多进程，规避 Windows spawn 问题。
"""
import os

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from ultralytics import YOLO  # noqa: E402

from src.utils import paths  # noqa: E402

# 经 paths 绝对化：ultralytics 解析相对 `path` 时以 cwd 为准，不可依赖
DATA_YAML = str(paths.runtime_data_yaml())

# (显示名, 权重相对路径, 评估 imgsz)
# 消融矩阵：
#   baseline exp    : imgsz=640, 原始数据 1439 张
#   exp_aug-2       : imgsz=320, 增强数据 2877 张
#   exp_aug640      : imgsz=640, 增强数据 2877 张  <- 对照，用于分离「增强」与「分辨率」
RUNS = [
    ("baseline exp", os.path.join("runs", "detect", "runs", "exp", "weights", "best.pt"), 640),
    ("aug320 exp_aug-2", os.path.join("runs", "detect", "runs", "exp_aug-2", "weights", "best.pt"), 320),
    ("aug640 exp_aug640", os.path.join("runs", "detect", "runs", "exp_aug640", "weights", "best.pt"), 640),
]


def evaluate(weight_rel: str, imgsz: int):
    """在 val 集上评估一个权重，返回 (metrics, names)。"""
    weight = os.path.join(PROJECT, weight_rel)
    if not os.path.exists(weight):
        raise FileNotFoundError(f"权重不存在: {weight}")
    model = YOLO(weight)
    res = model.val(
        data=DATA_YAML,
        imgsz=imgsz,
        split="val",
        workers=0,      # 关闭多进程，规避 Windows spawn
        verbose=False,
        plots=False,
    )
    return res


def main():
    results = {}
    for label, wrel, imgsz in RUNS:
        r = evaluate(wrel, imgsz)
        results[label] = r
        print(f"\n=== {label}  (imgsz={imgsz}) ===")
        print(f"{'class':<18}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
        for i, name in r.names.items():
            print(
                f"{name:<18}{r.box.p[i]:>8.3f}{r.box.r[i]:>8.3f}"
                f"{r.box.ap50[i]:>8.3f}{r.box.ap[i]:>10.3f}"
            )
        print(
            f"{'ALL':<18}{r.box.mp:>8.3f}{r.box.mr:>8.3f}"
            f"{r.box.map50:>8.3f}{r.box.map:>10.3f}"
        )

    # 差异对比：以第一个（baseline）为基准，其余各实验分别求差
    items = list(results.items())
    if len(items) >= 2:
        a_name, ra = items[0]
        for b_name, rb in items[1:]:
            print(f"\n=== 差异：{b_name} - {a_name} ===")
            print(f"{'class':<18}{'dP':>9}{'dR':>9}{'dmAP50':>9}{'dmAP50-95':>11}")
            for i, name in ra.names.items():
                print(
                    f"{name:<18}{rb.box.p[i] - ra.box.p[i]:>+9.3f}"
                    f"{rb.box.r[i] - ra.box.r[i]:>+9.3f}"
                    f"{rb.box.ap50[i] - ra.box.ap50[i]:>+9.3f}"
                    f"{rb.box.ap[i] - ra.box.ap[i]:>+11.3f}"
                )
            print(
                f"{'ALL':<18}{rb.box.mp - ra.box.mp:>+9.3f}"
                f"{rb.box.mr - ra.box.mr:>+9.3f}"
                f"{rb.box.map50 - ra.box.map50:>+9.3f}"
                f"{rb.box.map - ra.box.map:>+11.3f}"
            )


if __name__ == "__main__":
    main()
