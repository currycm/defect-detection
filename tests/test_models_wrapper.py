"""ONNX 导出与搬运的单元测试（不需要 torch / 真模型）。

回归点（L7）：`model.export()` 不接受 `path=` 参数，历史实现传了它，
一调用即 TypeError。这里用假模型锁死「传参集合」。
"""
from pathlib import Path

from src.models.wrapper import export_onnx


class FakeModel:
    """只记录 export 的调用参数，不真的导出。"""

    def __init__(self, produced: Path):
        self.produced = produced
        self.kwargs: dict = {}

    def export(self, **kwargs):
        self.kwargs = kwargs
        self.produced.write_bytes(b"fake-onnx")
        return str(self.produced)


def test_export_onnx_does_not_pass_unsupported_kwargs(tmp_path):
    src = tmp_path / "produced.onnx"
    m = FakeModel(src)

    export_onnx(m, tmp_path / "out" / "best.onnx", imgsz=320)

    assert "path" not in m.kwargs, "ultralytics 的 export() 不支持 path="
    assert m.kwargs["format"] == "onnx"
    assert m.kwargs["imgsz"] == 320
    assert m.kwargs["dynamic"] is False


def test_export_onnx_moves_file_to_target(tmp_path):
    src = tmp_path / "produced.onnx"
    m = FakeModel(src)
    out = tmp_path / "nested" / "best.onnx"

    result = export_onnx(m, out)

    assert Path(result) == out
    assert out.exists()
    assert out.read_bytes() == b"fake-onnx"


def test_export_onnx_without_target_returns_produced_path(tmp_path):
    src = tmp_path / "produced.onnx"
    m = FakeModel(src)

    result = export_onnx(m)

    assert Path(result) == src
