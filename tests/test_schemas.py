"""API 数据契约测试（不启动服务、不加载模型）。

背景（H3）：两个 predictor 的返回字典不一致（ONNX 版带 `name`，ultralytics
回退版不带），torch 路径下 `Detection(**d)` 会抛 pydantic ValidationError，
表现为 POST /detect 必然 500。所有 producer 的输出现在都在 `to_detection()`
收口，这里锁死该契约。
"""
from src.constants import CLASS_NAMES
from src.inference.schemas import Detection, DetectResponse, to_detection


def test_response_schema():
    r = DetectResponse(
        detections=[Detection(xyxy=[1, 2, 3, 4], conf=0.9, cls=0, name="crazing")],
        count=1,
    )
    assert r.count == 1
    assert r.detections[0].name == "crazing"


def test_detection_name_is_optional():
    """缺 name 也只能降级，不能让整个接口 500。"""
    d = Detection(xyxy=[0, 0, 1, 1], conf=0.5, cls=3)
    assert d.name == ""


def test_to_detection_fills_name_from_class_names():
    d = to_detection({"cls": 5, "conf": 0.8123, "xyxy": [1, 2, 3, 4]},
                     class_names=CLASS_NAMES)
    assert d.name == "scratches"
    assert d.cls == 5


def test_to_detection_keeps_explicit_name():
    d = to_detection({"cls": 0, "conf": 0.5, "xyxy": [0, 0, 1, 1],
                      "name": "crazing_v2"}, class_names=CLASS_NAMES)
    assert d.name == "crazing_v2"


def test_to_detection_falls_back_to_class_id_without_names():
    d = to_detection({"cls": 4, "conf": 0.5, "xyxy": [0, 0, 1, 1]})
    assert d.name == "4"


def test_to_detection_casts_numpy_style_scalars():
    """numpy 标量不可 JSON 序列化，必须转成原生 float。"""

    class FakeFloat:
        def __float__(self):
            return 0.25

    d = to_detection({"cls": 1, "conf": FakeFloat(), "xyxy": [FakeFloat()] * 4},
                     class_names=CLASS_NAMES)
    assert isinstance(d.conf, float)
    assert all(isinstance(v, float) for v in d.xyxy)


def test_to_detection_out_of_range_cls_does_not_crash():
    d = to_detection({"cls": 99, "conf": 0.5, "xyxy": [0, 0, 1, 1]},
                     class_names=CLASS_NAMES)
    assert d.name == "99"
