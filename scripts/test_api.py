"""端到端测试 FastAPI 服务：/health 与 /detect。

用法（需先启动服务）：
    python scripts/serve.py
    python scripts/test_api.py
"""
import json
import os
import time
import urllib.request

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

BASE = "http://127.0.0.1:8000"
TEST_DIR = os.path.join(PROJECT, "data", "processed", "images", "test")


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post_detect(img_path: str, conf: float = 0.25, iou: float = 0.5):
    """手工构造 multipart/form-data 上传图片（不依赖 requests）。"""
    boundary = "----defectBoundary7MA4YWxk"
    fname = os.path.basename(img_path)
    with open(img_path, "rb") as f:
        content = f.read()

    body = b""
    body += f"--{boundary}\r\n".encode()
    body += (f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n').encode()
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    url = f"{BASE}/detect?conf={conf}&iou={iou}"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(timeout: int = 60) -> bool:
    for _ in range(timeout):
        try:
            get("/health")
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    if not wait_ready():
        raise SystemExit("服务未就绪，请先运行: python scripts/serve.py")

    print("=== GET /health ===")
    h = get("/health")
    print(json.dumps(h, ensure_ascii=False, indent=2))

    imgs = sorted([f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith((".jpg", ".png"))])[:2]

    print("\n=== POST /detect （默认 conf=0.25, iou=0.5）===")
    for name in imgs:
        res = post_detect(os.path.join(TEST_DIR, name))
        print(f"\n  {name}: 检出 {res['count']} 个缺陷")
        for d in res["detections"][:5]:
            print(f"    {d['name']:<16} conf={d['conf']:.3f}  box={d['xyxy']}")

    print("\n=== POST /detect （低 conf=0.05，看召回变化）===")
    res = post_detect(os.path.join(TEST_DIR, imgs[0]), conf=0.05)
    print(f"  {imgs[0]}: conf=0.05 时检出 {res['count']} 个（对比默认档）")

    print("\n=== POST /detect （高 conf=0.6，看精度档）===")
    res = post_detect(os.path.join(TEST_DIR, imgs[0]), conf=0.6)
    print(f"  {imgs[0]}: conf=0.6 时检出 {res['count']} 个")


if __name__ == "__main__":
    main()
