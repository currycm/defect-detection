"""HTTP 接口回归测试（需要 fastapi + 权重）。

覆盖三条曾经的真实故障：
  H1 `/stream?source=` 曾能读取并回传项目外的任意本机文件 -> 现在越界即 400
  H3 torch 回退路径的 /detect 因返回字典缺 `name` 而 pydantic 500 -> 现在收口
  M2 `detect_interval=-1` 等非法参数曾让限频失效 -> 现在由 Query 边界挡在 422

用 TestClient 启动应用会触发 lifespan（加载 ONNX），所以整文件在缺权重时跳过。
"""
import numpy as np
import pytest

pytest.importorskip("fastapi", reason="需要 fastapi")
pytest.importorskip("cv2", reason="需要 opencv")
pytest.importorskip("httpx", reason="TestClient 需要 httpx")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.constants import CLASS_NAMES  # noqa: E402
from src.inference import api as api_mod  # noqa: E402
from src.utils import paths  # noqa: E402

pytestmark = pytest.mark.skipif(
    not paths.ONNX_PATH.exists(), reason="需要 weights/best.onnx（先跑 scripts/export.py）"
)


# ------------------------------------------------------------ 源码解析（不启服务）
def test_resolve_source_defaults_to_test_split():
    assert api_mod._resolve_source("").startswith("synthetic:")


def test_resolve_source_accepts_camera_index():
    assert api_mod._resolve_source("0") == 0


def test_resolve_source_accepts_project_path():
    target = paths.PROCESSED_DIR / "images" / "test"
    assert api_mod._resolve_source(str(target)) == str(target)


def test_resolve_source_rejects_outside_path(tmp_path):
    with pytest.raises(HTTPException) as ei:
        api_mod._resolve_source(str(tmp_path))
    assert ei.value.status_code == 400


def test_resolve_source_rejects_outside_synthetic(tmp_path):
    with pytest.raises(HTTPException) as ei:
        api_mod._resolve_source(f"synthetic:{tmp_path}")
    assert ei.value.status_code == 400


# ------------------------------------------------------------ 端到端
@pytest.fixture(scope="module")
def client():
    with TestClient(api_mod.app) as c:
        yield c


def test_health_reports_model_and_classes(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["classes"] == CLASS_NAMES
    assert body["backend"] == "onnxruntime"   # 部署链路就是 ONNXRuntime
    assert body["providers"]


def test_detect_returns_contract_compliant_payload(client):
    """H3 回归：返回体必须能被 DetectResponse 校验，且每条都带 name。"""
    import cv2

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 200
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    r = client.post("/detect?conf=0.25&iou=0.5",
                    files={"file": ("t.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == len(body["detections"])
    for d in body["detections"]:
        assert isinstance(d["name"], str) and d["name"]
        assert len(d["xyxy"]) == 4


def test_detect_rejects_non_image(client):
    r = client.post("/detect", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_stream_rejects_out_of_range_params(client):
    """M2 回归：非法参数应由框架直接挡成 422，而不是进入流循环。"""
    assert client.get("/stream?detect_interval=-1").status_code == 422
    assert client.get("/stream?target_fps=0").status_code == 422
    assert client.get("/stream?conf=2").status_code == 422


def test_stream_rejects_outside_source(client):
    """H1 回归：项目外路径必须 400（而不是开始推流）。"""
    r = client.get("/stream?source=C:/Windows")
    assert r.status_code == 400
