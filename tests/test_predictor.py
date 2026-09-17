"""纯函数式工具测试（不依赖模型权重）。"""
import numpy as np

from src.utils.imageio import draw_boxes, draw_detections


def test_draw_boxes_runs():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_boxes(
        img,
        [{"xyxy": [10, 10, 50, 50], "conf": 0.9, "cls": 0}],
        ["crazing"],
    )
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_draw_detections_uses_name_field():
    """det 自带 name 时优先用它，不需要外部 names 列表。"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_detections(img, [{"xyxy": [5, 5, 60, 60], "conf": 0.5,
                                 "cls": 5, "name": "scratches"}])
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_draw_detections_clamps_box_at_border():
    """贴边/越界的框不应抛异常（旧实现会因标签画到图外而裁切）。"""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    out = draw_detections(img, [{"xyxy": [0, 0, 39, 39], "conf": 0.99, "cls": 0}])
    assert out.shape == img.shape


def test_draw_detections_skips_malformed_entry():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    out = draw_detections(img, [{"conf": 0.9, "cls": 0}])  # 缺 xyxy
    assert np.array_equal(out, img)


def test_class_colors_cover_all_classes():
    """颜色表必须覆盖全部 6 类，否则某些类会退化成默认绿色。"""
    from src.constants import NUM_CLASSES
    from src.utils.imageio import CLASS_COLORS

    assert set(CLASS_COLORS) == set(range(NUM_CLASSES))
