"""convert 模块单元测试（Pascal VOC XML 解析）。"""
from pathlib import Path

from src.data.convert import NEU_CLASSES, _parse_voc, convert_one


def _make_xml(tmp_path: Path, w: int = 200, h: int = 200) -> Path:
    xml = tmp_path / "crazing_1.xml"
    xml.write_text(
        '<?xml version="1.0"?>\n'
        '<annotation><size>'
        f'<width>{w}</width><height>{h}</height>'
        '</size><object><name>crazing</name><bndbox>'
        '<xmin>2</xmin><ymin>2</ymin><xmax>193</xmax><ymax>194</ymax>'
        '</bndbox></object></annotation>',
        encoding="utf-8",
    )
    return xml


def test_parse_voc_basic(tmp_path: Path):
    xml = _make_xml(tmp_path)
    W, H, boxes = _parse_voc(xml)
    assert (W, H) == (200, 200)
    assert boxes == [("crazing", 2.0, 2.0, 193.0, 194.0)]


def test_convert_one_normalizes(tmp_path: Path):
    xml = _make_xml(tmp_path)
    img = tmp_path / "crazing_1.jpg"
    img.write_bytes(b"")
    class_map = {n: i for i, n in enumerate(NEU_CLASSES)}
    yolo = convert_one(xml, img, class_map)
    cid, cx, cy, bw, bh = yolo.split()
    assert int(cid) == 0  # crazing 是第 0 类
    assert abs(float(cx) - (195 / 2) / 200) < 1e-3
    assert abs(float(cy) - (196 / 2) / 200) < 1e-3
    assert abs(float(bw) - 191 / 200) < 1e-3
    assert abs(float(bh) - 192 / 200) < 1e-3


def test_class_order():
    assert NEU_CLASSES.index("scratches") == 5
