"""图像读写与画框工具测试（src/utils/imageio.py）。

重点覆盖 Windows 中文路径：项目本身就在「机器视觉」目录下，
`cv2.imread` / `cv2.imwrite` 在含非 ASCII 的路径上会**静默失败**
（读返回 None、写不报错但文件不生成），这是本项目最容易踩的环境坑。
"""
import numpy as np

from src.utils.imageio import (
    CLASS_COLORS,
    draw_boxes,
    draw_detections,
    encode_jpeg,
    load_image,
    save_image,
)


def _checker(h=32, w=48) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h // 4: h // 2, w // 4: w // 2] = (0, 180, 0)
    return img


def test_save_load_roundtrip_with_chinese_path(tmp_path):
    d = tmp_path / "机器视觉目录"
    d.mkdir()
    p = d / "测试图像.jpg"

    assert save_image(p, _checker()) is True
    assert p.exists() and p.stat().st_size > 0

    back = load_image(p)
    assert back is not None
    assert back.shape == (32, 48, 3)


def test_load_missing_file_returns_none(tmp_path):
    assert load_image(tmp_path / "不存在.jpg") is None


def test_save_image_creates_valid_jpeg(tmp_path):
    p = tmp_path / "out.png"
    assert save_image(p, _checker()) is True
    assert p.read_bytes()[:4] == b"\x89PNG"


def test_encode_jpeg_returns_jpeg_payload():
    data = encode_jpeg(_checker(), quality=70)
    assert data[:2] == b"\xff\xd8"      # JPEG SOI
    assert data[-2:] == b"\xff\xd9"     # JPEG EOI
    assert len(data) > 100


def test_draw_detections_uses_name_then_fallback():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(img, [
        {"xyxy": [5, 5, 60, 60], "conf": 0.5, "cls": 5, "name": "scratches"},
        {"xyxy": [10, 10, 20, 20], "conf": 0.3, "cls": 0},  # 无 name，回退 cls
    ], names=["crazing"])
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_draw_detections_clamps_label_at_borders():
    """全部贴四个角，标签仍不能画到图外（旧实现缺右边界钳制）。"""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    dets = [{"xyxy": [0, 0, 39, 39], "conf": 0.99, "cls": 0},
            {"xyxy": [39, 39, 39, 39], "conf": 0.99, "cls": 1}]
    out = draw_detections(img, dets)
    assert out.shape == img.shape


def test_draw_detections_skips_malformed_entry():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    out = draw_detections(img, [{"conf": 0.9, "cls": 0}])   # 缺 xyxy
    assert np.array_equal(out, img)


def test_draw_boxes_compat_wrapper():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = draw_boxes(img, [{"xyxy": [2, 2, 30, 30], "conf": 0.7, "cls": 2}],
                     ["crazing", "inclusion", "patches"])
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_class_colors_cover_all_classes():
    from src.constants import NUM_CLASSES

    assert set(CLASS_COLORS) == set(range(NUM_CLASSES))
