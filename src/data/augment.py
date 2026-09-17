"""离线弱类定向增强（Step 5）。

针对 baseline（Step 4）中精度最低的 3 个类
crazing / rolled-in_scale / scratches，在 *train* 划分上生成额外增强样本，
直接写入 data/processed/images/train 与 data/processed/labels/train。
val/test 保持不动，确保评估干净、可比。

在线增强（mosaic / hsv / flip 等）仍由 configs/hyp.yaml 在训练时施加，
本模块是“离线扩样”，两者互补：在线负责每 epoch 的随机性，
离线负责把难例类的样本频率整体抬高（弱类过采样），针对性缓解精度塌陷。
"""
from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import cv2

from ..utils.imageio import load_image, save_image
from ..utils.logger import get_logger

LOG = get_logger("augment")

# baseline（Step 4）精度最低的 3 类：P 分别为 0.426 / 0.48 / 0.5
WEAK_CLASSES = ["crazing", "rolled-in_scale", "scratches"]

# 每类每图生成的增强副本数 -> 弱类整体约 2× 过采样
COPIES_PER_IMAGE = 2

# 增强文件名标记，用于可重入（跳过已生成的 _aug 文件，避免重复增强爆炸）
_AUG_TAG = "_aug"


def build_weak_augmenter() -> A.Compose:
    """弱类专用增强：只含“框安全”变换，不丢失/位移目标框。

    选型说明：
    - 翻转 / 90° 旋转：几何上只镜像或循环移位，YOLO 框可精确变换，不裁剪目标。
    - 亮度/对比度/HSV/噪声/运动模糊：纯像素级，框坐标完全不变。
    - 刻意不用任意角度 degrees 旋转或大 scale/translate，
      这类会裁掉框或把框推出画面，对小图反而伤标签质量。
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5
            ),
            A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


def _read_label(txt_path: Path) -> tuple[list[int], list[list[float]]]:
    """读取 YOLO 标签，返回 (class_ids, [[cx,cy,w,h], ...]) 归一化。

    容错：标签行必须是 5 段（class + 4 个归一化坐标）。历史上直接取
    parts[0] / parts[1..4]，标签被截断时会抛 IndexError 让整个增强中断。
    """
    classes: list[int] = []
    boxes: list[list[float]] = []
    if not txt_path.exists():
        return classes, boxes
    for lineno, line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            LOG.warning("%s 第 %d 行字段数=%d（应为 5），已跳过",
                        txt_path.name, lineno, len(parts))
            continue
        try:
            cls = int(parts[0])
            box = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
        except ValueError:
            LOG.warning("%s 第 %d 行无法解析为数字，已跳过", txt_path.name, lineno)
            continue
        classes.append(cls)
        boxes.append(box)
    return classes, boxes


def _write_label(txt_path: Path, classes: list[int], boxes: list[list[float]]) -> None:
    """写出 YOLO 标签。"""
    lines = [
        f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
        for c, b in zip(classes, boxes)
    ]
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def augment_weak_classes(processed_dir: str,
                         copies_per_image: int = COPIES_PER_IMAGE,
                         seed: int = 42) -> dict[str, int]:
    """对弱类 train 样本离线增强，写回 train 目录。

    参数:
        processed_dir: data/processed 根目录（内含 images/train, labels/train）
        copies_per_image: 每类每图生成多少增强副本
        seed: 随机种子（用于文件名去重，保证可复现）
    返回:
        每类新增样本计数。
    """
    rng = random.Random(seed)
    aug = build_weak_augmenter()

    img_dir = Path(processed_dir) / "images" / "train"
    lbl_dir = Path(processed_dir) / "labels" / "train"
    if not img_dir.exists():
        LOG.error("未找到 %s，请先完成 Step 3 转换", img_dir)
        return {}

    added: dict[str, int] = {c: 0 for c in WEAK_CLASSES}

    # 按文件名前缀把训练图归类到弱类（NEU-DET 图名形如 crazing_001.jpg）
    weak_imgs: dict[str, list[Path]] = {c: [] for c in WEAK_CLASSES}
    for img in sorted(img_dir.glob("*.jpg")):
        if _AUG_TAG in img.stem:
            continue  # 跳过已增强文件，保证可重入
        for c in WEAK_CLASSES:
            if img.stem.startswith(c):
                weak_imgs[c].append(img)
                break

    for c, imgs in weak_imgs.items():
        if not imgs:
            LOG.warning("%s 在 train 中未找到样本，跳过", c)
            continue
        for img in imgs:
            image = load_image(img)  # 中文路径安全读图（唯一实现见 utils.imageio）
            if image is None:
                LOG.warning("读不到 %s，跳过", img.name)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            classes, boxes = _read_label(lbl_dir / (img.stem + ".txt"))
            if not boxes:
                continue
            for k in range(copies_per_image):
                out = aug(image=image, bboxes=boxes, class_labels=classes)
                aug_img = out["image"]
                aug_boxes = out["bboxes"]
                aug_classes = out["class_labels"]
                if not aug_boxes:
                    continue  # 理论上框安全变换不会清空，保险起见跳过孤儿图
                new_stem = f"{img.stem}{_AUG_TAG}{k+1}_{rng.randint(0, 9999)}"
                save_image(
                    img_dir / (new_stem + ".jpg"),
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                )
                _write_label(lbl_dir / (new_stem + ".txt"), aug_classes, aug_boxes)
                added[c] += 1
        LOG.info("%s: 原有 %d 张，新增 %d 张增强样本", c, len(imgs), added[c])
    return added
