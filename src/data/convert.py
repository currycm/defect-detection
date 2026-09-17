"""NEU-DET Pascal VOC XML 标注 -> YOLO 格式转换。

原始数据集（Kaggle: kaustubhdikshit/neu-surface-defect-database）结构：
    data/raw/NEU-DET/{train,validation}/{images/<class>/*.jpg, annotations/*.xml}

每个 .xml 为 Pascal VOC 格式，含 <size> 与一个或多个 <object><bndbox>。
本模块将其转为 YOLO 归一化标签（`class_id cx cy w h`，绝对坐标 / 图片尺寸），
并保留 train/validation 划分（validation 在输出中记为 val）。
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from ..constants import CLASS_NAMES
from ..utils.io import ensure_dir
from ..utils.logger import get_logger

LOG = get_logger("convert")

# 保留旧名字给外部/测试引用；真源是 src.constants.CLASS_NAMES，
# 避免「同一个类别表在多个文件各写一份、改一处漏一处」。
NEU_CLASSES = list(CLASS_NAMES)
NEUDET_SUBDIR = "NEU-DET"
SOURCE_SPLITS = ("train", "validation")
_IMG_EXTS = (".jpg", ".png", ".jpeg", ".bmp")


def _parse_voc(xml_path: Path):
    """解析 Pascal VOC XML。

    返回 (W, H, [(class_name, xmin, ymin, xmax, ymax), ...])。
    W/H 取自 <size>；boxes 取所有 <object> 的 <bndbox>。
    """
    root = ET.parse(str(xml_path)).getroot()
    # 容错：部分第三方标注缺失 <size> 节点，直接 size.findtext 会抛
    # AttributeError 而非走下面的「按图片实际尺寸回退」逻辑（M4）。
    size = root.find("size")
    if size is not None:
        W = int(size.findtext("width", "0") or 0)
        H = int(size.findtext("height", "0") or 0)
    else:
        W = H = 0

    boxes: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            LOG.warning("跳过 %s 中无法解析的 bndbox", xml_path.name)
            continue
        boxes.append((name, xmin, ymin, xmax, ymax))
    return W, H, boxes


def convert_one(xml_path: Path, img_path: Path, class_map: dict[str, int]) -> str:
    """转换单个 XML 为 YOLO 文本标签。"""
    W, H, boxes = _parse_voc(xml_path)

    # 尺寸回退：XML 缺失 <size> 时用图片实际尺寸
    # 注意用 load_image 而非 cv2.imread —— 本机项目路径含中文（「机器视觉」），
    # cv2.imread 在中文路径下会静默返回 None（M8/L4）。
    if W <= 0 or H <= 0:
        from ..utils.imageio import load_image
        arr = load_image(img_path)
        if arr is None:
            LOG.warning("读不到 %s 尺寸，按 200x200 处理", img_path.name)
            W = H = 200
        else:
            H, W = arr.shape[:2]

    if not boxes:
        LOG.warning("%s 无任何有效目标框", xml_path.name)

    lines: list[str] = []
    for name, xmin, ymin, xmax, ymax in boxes:
        if name not in class_map:
            LOG.warning("未知类别 %s（%s），跳过", name, xml_path.name)
            continue
        # 容错：部分标注可能 xmin>xmax
        xmin, xmax = sorted((xmin, xmax))
        ymin, ymax = sorted((ymin, ymax))
        cx = (xmin + xmax) / 2 / W
        cy = (ymin + ymax) / 2 / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H
        # 裁剪到 [0,1]，防止越界坐标污染训练
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        lines.append(f"{class_map[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def convert_all(raw_dir: str, out_dir: str,
                class_names: Sequence[str] | None = None) -> dict[str, int]:
    """转换 data/raw/NEU-DET 下 train/validation 的全部 XML 标注。

    输出布局（与 configs/data.yaml 对齐）：
        data/processed/images/{train,val}/
        data/processed/labels/{train,val}/

    返回值：每个 split 的样本计数。

    说明：构建跨 train/validation 的全局图片索引，以兼容个别
    “图片在 train、标注在 validation”的错位文件（如 crazing_240）。
    """
    raw = Path(raw_dir) / NEUDET_SUBDIR
    if not raw.exists():
        LOG.error("未找到 %s，请先把 NEU-DET 下载到 data/raw/NEU-DET", raw)
        return {}

    class_names = list(class_names or NEU_CLASSES)
    class_map = {n: i for i, n in enumerate(class_names)}
    out = Path(out_dir)

    # 全局图片索引（按 stem），覆盖两个 split
    img_index: dict[str, Path] = {}
    for split in SOURCE_SPLITS:
        for p in (raw / split / "images").rglob("*"):
            if p.suffix.lower() in _IMG_EXTS:
                img_index.setdefault(p.stem, p)

    counts: dict[str, int] = {}
    for split in SOURCE_SPLITS:
        ann_dir = raw / split / "annotations"
        target_split = "val" if split == "validation" else split
        od_img = ensure_dir(out / "images" / target_split)
        od_lbl = ensure_dir(out / "labels" / target_split)
        xmls = sorted(ann_dir.glob("*.xml"))
        n = 0
        for xml in xmls:
            img = img_index.get(xml.stem)
            if img is None:
                LOG.warning("找不到 %s 对应图片，跳过", xml.stem)
                continue
            yolo = convert_one(xml, img, class_map)
            (od_lbl / (xml.stem + ".txt")).write_text(yolo, encoding="utf-8")
            shutil.copy(img, od_img / img.name)
            n += 1
        counts[target_split] = n
        LOG.info("%s: 转换 %d 个样本 -> %s", split, n, out / "images" / target_split)
    return counts
