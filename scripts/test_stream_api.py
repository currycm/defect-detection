"""实时推流接口（/stream 系列）自动化验证。

关键：不能用 FastAPI TestClient —— 它的传输层会缓冲整个响应体，
对「无限 MJPEG 流」永远等不到结束。这里在进程内起一个真实 uvicorn
（127.0.0.1:8123）+ 真实 socket，用 httpx 流式读取，最贴近实际使用。

  T1 /health          后端为 onnxruntime
  T2 /detect          普通图片检测仍正常（确认加流式接口没破坏原有功能）
  T3 /stream          MJPEG 多段响应结构正确、能持续产出可解码的 JPEG 帧
  T4 /stream/state    轮询接口能反映活跃状态
  T5 断开即释放        客户端断开后流自动停止、线程无残留、状态归位
  T6 /stream/stop     无活跃流时幂等返回，不报错
"""
import sys
import threading
import time
from pathlib import Path

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import httpx  # noqa: E402
import uvicorn  # noqa: E402

from src.inference import api as api_mod  # noqa: E402

PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"
RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def imdecode_bytes(buf: bytes):
    arr = np.frombuffer(buf, np.uint8)
    return None if arr.size == 0 else cv2.imdecode(arr, cv2.IMREAD_COLOR)


def wait_server(timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"{BASE}/health", timeout=2.0).status_code == 200:
                return True
        except Exception:
            time.sleep(0.3)
    return False


def split_jpegs(buf: bytes):
    """从累积字节流里切出完整 JPEG（SOI=FFD8 ... EOI=FFD9），返回 (帧列表, 剩余缓冲)。"""
    out = []
    while True:
        s = buf.find(b"\xff\xd8")
        if s < 0:
            return out, b""
        e = buf.find(b"\xff\xd9", s + 2)
        if e < 0:
            return out, buf[s:]
        out.append(buf[s:e + 2])
        buf = buf[e + 2:]


def main():
    server = uvicorn.Server(uvicorn.Config(api_mod.app, host="127.0.0.1", port=PORT,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, name="uvicorn-test", daemon=True)
    th.start()
    if not wait_server():
        print("[FATAL] 测试服务未能启动")
        return 1
    print(f"测试服务已就绪 {BASE}\n")

    try:
        print("T1. /health")
        r = httpx.get(f"{BASE}/health", timeout=10.0)
        body = r.json()
        check("/health 返回 200", r.status_code == 200, f"status_code={r.status_code}")
        check("后端为 onnxruntime", body.get("backend") == "onnxruntime", str(body.get("backend")))
        check("类别数为 6", len(body.get("classes", [])) == 6)

        print("T2. /detect（确认原有接口未被破坏）")
        test_dir = Path(PROJECT) / "data" / "processed" / "images" / "test"
        det_counts = []
        for p in sorted(test_dir.glob("*.jpg"))[:3]:
            r = httpx.post(f"{BASE}/detect", params={"conf": 0.25, "iou": 0.5},
                           files={"file": ("a.jpg", p.read_bytes(), "image/jpeg")}, timeout=30.0)
            if r.status_code == 200:
                det_counts.append(r.json()["count"])
        check("/detect 对 3 张图均返回 200", len(det_counts) == 3, f"检出 {det_counts}")

        print("T3. /stream（MJPEG 流，真实 socket）")
        thread_before = {t.ident for t in threading.enumerate()}
        frames, t0 = [], time.time()
        # hold 取小值 + 按墙钟时间观察，避免机器被他进程抢占时窗口内只播一张图而误判
        with httpx.stream("GET", f"{BASE}/stream",
                          params={"target_fps": 10, "detect_interval": 0.2, "hold": 2},
                          timeout=httpx.Timeout(40.0, read=40.0)) as resp:
            check("/stream 返回 200", resp.status_code == 200, f"status_code={resp.status_code}")
            ct = resp.headers.get("content-type", "")
            check("Content-Type 为 multipart/x-mixed-replace",
                  "multipart/x-mixed-replace" in ct, ct)
            buf = b""
            for chunk in resp.iter_bytes():
                buf += chunk
                got, buf = split_jpegs(buf)
                frames.extend(got)
                el = time.time() - t0
                if (el >= 6.0 and len(frames) >= 15) or len(frames) >= 150 or el > 25:
                    break
        check("收到 >=3 帧 JPEG", len(frames) >= 3, f"收到 {len(frames)} 帧")
        if frames:
            ok_decode = all(imdecode_bytes(j) is not None for j in frames[:3])
            check("帧为合法 JPEG（可解码）", ok_decode)
            sizes = [len(f) for f in frames]
            check("每帧大小合理（>1KB）", min(sizes) > 1024,
                  f"最小 {min(sizes)}B 最大 {max(sizes)}B")
            shp = [imdecode_bytes(j).shape for j in frames[:1]]
            check("帧尺寸与源图一致（200x200）", all(s[0] == 200 and s[1] == 200 for s in shp), str(shp))
            # 活跃性：画面必须真的在推进，而不是把同一帧反复吐出来
            import hashlib
            uniq = len({hashlib.md5(f).hexdigest() for f in frames})
            check("画面持续更新（存在不同帧）", uniq >= 2,
                  f"{len(frames)} 帧中有 {uniq} 帧不同")

        print("T4. 断开后自动释放（生命周期）")
        active = True
        deadline = time.time() + 12
        while time.time() < deadline:
            if httpx.get(f"{BASE}/stream/state", timeout=10.0).json().get("active") is False:
                active = False
                break
            time.sleep(0.3)
        check("断开后 /stream/state 报 active=False", active is False)
        check("断开后无残留活动流对象", api_mod._active_stream is None)
        thread_after = {t.ident for t in threading.enumerate()}
        leaked = thread_after - thread_before
        check("无 stream-detector 线程残留",
              not any(t.name == "stream-detector" for t in threading.enumerate()),
              f"新增线程 {len(leaked)} 个")

        print("T5. /stream/stop 幂等")
        r = httpx.post(f"{BASE}/stream/stop", timeout=10.0)
        check("无活跃流时 /stream/stop 正常返回", r.status_code == 200, str(r.json()))
    finally:
        server.should_exit = True
        th.join(timeout=10)

    print()
    print("=" * 56)
    passed = sum(1 for x in RESULTS if x)
    print(f"通过 {passed}/{len(RESULTS)} 项")
    if passed == len(RESULTS):
        print("实时推流接口验证全部通过 ✓")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
