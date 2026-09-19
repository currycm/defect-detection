"""ONNX 后处理与 NMS 单元测试（不加载真实 ONNX 模型）。

NMS 是检测器里最容易写错的算法：索引错位、面积零除、order 被改写后
还用旧索引……这里覆盖空输入、单框、完全重叠、IoU 恰好等于阈值、
多类并存等典型用例，锁死 `_nms` 与 `_postprocess` 的行为。

构造测试用 predictor 时用 `ONNXDefectPredictor.__new__` 跳过 __init__
（__init__ 会去加载真实 ONNX session），只把 `class_names` 设上即可调用
`_postprocess`（它只依赖 `class_names` 与静态方法 `_nms`）。
"""
from __future__ import annotations

import numpy as np

from src.inference.onnx_predictor import ONNXDefectPredictor


# --------------------------------------------------------------------------
# _nms
# --------------------------------------------------------------------------
def test_nms_empty_input():
    """空输入应安全返回空列表，不应抛 IndexError/ZeroDivisionError。"""
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    assert ONNXDefectPredictor._nms(boxes, scores, 0.5) == []


def test_nms_single_box_kept():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    assert ONNXDefectPredictor._nms(boxes, scores, 0.5) == [0]


def test_nms_complete_overlap_drops_lower_score():
    """完全重叠的两框应保留得分高的那个。"""
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert ONNXDefectPredictor._nms(boxes, scores, 0.5) == [0]


def test_nms_far_apart_keeps_both():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert sorted(ONNXDefectPredictor._nms(boxes, scores, 0.5)) == [0, 1]


def test_nms_iou_strictly_above_threshold_is_suppressed():
    """IoU 严格大于阈值才被抑制；等于阈值时保留。

    实现里 `order = rest[ious <= iou_thr]`，即 IoU <= 阈值保留、> 阈值才抑制。
    两个完全相同的 10x10 框 IoU=1.0：
      - 阈值 1.0：1.0 <= 1.0 True -> 第二个保留
      - 阈值 0.99：1.0 <= 0.99 False -> 第二个被抑制
    """
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    # 等于阈值 -> 不抑制
    assert ONNXDefectPredictor._nms(boxes, scores, 1.0) == [0, 1]
    # 严格大于阈值 -> 抑制
    assert ONNXDefectPredictor._nms(boxes, scores, 0.99) == [0]


def test_nms_zero_area_box_does_not_crash():
    """退化为线/点的框（w 或 h = 0）不该让 NMS 抛错；结果取决于排序顺序。"""
    boxes = np.array([[5, 5, 5, 5], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    # 零面积框面积为 0，与任何框的 inter 都为 0，IoU 为 0/union=0 -> 0（被 1e-6 兜底），
    # 因此两个都会被保留
    assert sorted(ONNXDefectPredictor._nms(boxes, scores, 0.5)) == [0, 1]


def test_nms_keeps_highest_score_first_in_order():
    """三个重叠框，保留顺序应是得分从高到低。"""
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], dtype=np.float32)
    scores = np.array([0.7, 0.9, 0.8], dtype=np.float32)
    keep = ONNXDefectPredictor._nms(boxes, scores, 0.3)
    # 第一保留必是得分最高的索引 1；后续两个因 IoU>=0.3 被压掉
    assert keep == [1]


# --------------------------------------------------------------------------
# _postprocess（用 __new__ 跳过 ONNX 加载）
# --------------------------------------------------------------------------
def _make_predictor(class_names=("crazing", "inclusion")) -> ONNXDefectPredictor:
    """构造一个**不加载 ONNX** 的实例，仅用于测 _postprocess。"""
    p = ONNXDefectPredictor.__new__(ONNXDefectPredictor)
    p.class_names = list(class_names)
    return p


def test_postprocess_empty_predictions():
    """8400 个 anchor 全 0 分数时应返回空列表，不抛异常。"""
    p = _make_predictor()
    # shape=(1, 4+nc=6, 8400)，全 0
    out = np.zeros((1, 6, 8400), dtype=np.float32)
    res = p._postprocess(out, r=1.0, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    assert res == []


def test_postprocess_single_box_layout_4_plus_nc():
    """(1, 4+nc, N) 布局：单框 + 类别 0 高分应被解码并保留。"""
    p = _make_predictor()
    nc = 2
    # 构造一个 8400-anchor 的输出，只在第 0 个 anchor 放一个真实框
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    out[0, 0, 0] = 50      # cx
    out[0, 1, 0] = 50      # cy
    out[0, 2, 0] = 20      # w
    out[0, 3, 0] = 20      # h
    out[0, 4, 0] = 0.9     # class 0 score
    out[0, 5, 0] = 0.1     # class 1 score

    res = p._postprocess(out, r=1.0, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    assert len(res) == 1
    d = res[0]
    assert d["cls"] == 0
    assert d["name"] == "crazing"
    # cx,cy,w,h=50,50,20,20 -> xyxy=(40,40,60,60)
    assert d["xyxy"] == [40.0, 40.0, 60.0, 60.0]
    assert abs(d["conf"] - 0.9) < 1e-3


def test_postprocess_layout_N_plus_4_plus_nc():
    """(1, N, 4+nc) 布局也应被正确识别与解码。"""
    p = _make_predictor()
    nc = 2
    out = np.zeros((1, 8400, 4 + nc), dtype=np.float32)
    out[0, 0, 0] = 50
    out[0, 0, 1] = 50
    out[0, 0, 2] = 20
    out[0, 0, 3] = 20
    out[0, 0, 4] = 0.9
    out[0, 0, 5] = 0.1

    res = p._postprocess(out, r=1.0, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    assert len(res) == 1
    assert res[0]["xyxy"] == [40.0, 40.0, 60.0, 60.0]


def test_postprocess_letterbox_inverse():
    """letterbox 还原：先减 pad，再除以缩放比。"""
    p = _make_predictor()
    nc = 1
    # 在 640x640 输入空间下的框 (300, 300, 320, 320)
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    out[0, 0, 0] = 300
    out[0, 1, 0] = 300
    out[0, 2, 0] = 20
    out[0, 3, 0] = 20
    out[0, 4, 0] = 0.9

    # 假设原图 200x200，r=3.2（640/200），pad=(0,0)
    r = 640 / 200
    res = p._postprocess(out, r=r, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    assert len(res) == 1
    # (300-0)/3.2 = 93.75，框宽 20/3.2 = 6.25
    # xyxy = (93.75-10/3.2, 93.75-10/3.2, 93.75+10/3.2, 93.75+10/3.2)
    x1 = round((300 - 10) / r, 2)
    x2 = round((300 + 10) / r, 2)
    assert res[0]["xyxy"] == [x1, x1, x2, x2]


def test_postprocess_clamps_to_image_bounds():
    """还原后越界的框应被裁剪到图像范围内，且退化零面积框被丢弃。"""
    p = _make_predictor()
    nc = 1
    # 框中心在 (-10, -10)，半径 20，letterbox r=1 pad=(0,0)，
    # 还原后部分会落在图像外，应被裁到 [0, 200]
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    out[0, 0, 0] = -10
    out[0, 1, 0] = -10
    out[0, 2, 0] = 40
    out[0, 3, 0] = 40
    out[0, 4, 0] = 0.9

    res = p._postprocess(out, r=1.0, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    assert len(res) == 1
    x1, y1, x2, y2 = res[0]["xyxy"]
    assert x1 == 0.0 and y1 == 0.0
    assert x2 <= 200 and y2 <= 200
    assert x2 > 0 and y2 > 0          # 不应该退化成零面积


def test_postprocess_drops_zero_area_after_clamp():
    """裁剪后退化成零面积的框应被丢弃。"""
    p = _make_predictor()
    nc = 1
    # 完全在图像左上外侧的框：cx=-100, cy=-100, w=10, h=10
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    out[0, 0, 0] = -100
    out[0, 1, 0] = -100
    out[0, 2, 0] = 10
    out[0, 3, 0] = 10
    out[0, 4, 0] = 0.9

    res = p._postprocess(out, r=1.0, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    assert res == []


def test_postprocess_results_sorted_by_conf_desc():
    """返回结果应按置信度从高到低排序。"""
    p = _make_predictor()
    nc = 1
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    # 三个不同位置、不同分数的框
    for i, (cx, cy, conf) in enumerate([(50, 50, 0.6), (100, 100, 0.9), (150, 150, 0.3)]):
        out[0, 0, i] = cx
        out[0, 1, i] = cy
        out[0, 2, i] = 10
        out[0, 3, i] = 10
        out[0, 4, i] = conf

    res = p._postprocess(out, r=1.0, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    confs = [d["conf"] for d in res]
    assert confs == sorted(confs, reverse=True)


def test_postprocess_sigmoid_applied_when_needed():
    """cls_scores 超过 1 时应自动做一次 sigmoid。"""
    p = _make_predictor()
    nc = 1
    # 直接放 logits=2.2（sigmoid 后约 0.9），conf 阈值 0.25
    out = np.zeros((1, 4 + nc, 8400), dtype=np.float32)
    out[0, 0, 0] = 50
    out[0, 1, 0] = 50
    out[0, 2, 0] = 20
    out[0, 3, 0] = 20
    out[0, 4, 0] = 2.2      # > 1，触发 sigmoid 分支

    res = p._postprocess(out, r=1.0, pad=(0, 0), conf=0.25, iou=0.5,
                         img_w=200, img_h=200)
    assert len(res) == 1
    expected = 1.0 / (1.0 + np.exp(-2.2))
    assert abs(res[0]["conf"] - expected) < 1e-3
