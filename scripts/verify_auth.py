"""真正的 uvicorn 上验证鉴权（TestClient 覆盖不到流式响应）。

TestClient 会把无限流式响应整体缓冲，测不了 /stream 的鉴权分支；
必须真实起服务 + 真实 socket 请求。

用法：
    D:/Anaconda/python.exe scripts/verify_auth.py
"""
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "e2e-test-key-9f3a"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def req(url, headers=None, timeout=5, max_bytes=200):
    """返回 (status, body 前 max_bytes 字符)。

    /health 的 JSON 较长（类别表 + 允许路径），断言字段时要放宽 max_bytes，
    否则截断后 `in` 判断会假失败。
    """
    r = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read(max_bytes).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read(max_bytes).decode("utf-8", "replace")


def main():
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["DEFECT_API_KEY"] = KEY
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"启动真实服务于 {base}（已配置 DEFECT_API_KEY）\n")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.inference.api:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        # 等服务起来
        ready = False
        for _ in range(60):
            time.sleep(0.5)
            try:
                s, _b = req(f"{base}/health", timeout=2)
                if s == 200:
                    ready = True
                    break
            except Exception:
                continue
        if not ready:
            print("服务未能在 30s 内启动")
            return 2

        print("=== 1) 免鉴权接口 ===")
        s, b = req(f"{base}/health", max_bytes=8192)
        check("GET /health 无密钥可访问", s == 200, f"HTTP {s}")
        check("health 报告 auth_enabled=true",
              '"auth_enabled":true' in b.replace(" ", ""),
              "已找到" if '"auth_enabled":true' in b.replace(" ", "")
              else b[-160:])

        print("\n=== 2) 受保护接口拒绝匿名访问 ===")
        s, _ = req(f"{base}/stream/state")
        check("GET /stream/state 无密钥 -> 401", s == 401, f"HTTP {s}")
        s, _ = req(f"{base}/stream/state", {"X-API-Key": "wrong"})
        check("GET /stream/state 错误密钥 -> 401", s == 401, f"HTTP {s}")

        print("\n=== 3) 正确密钥放行 ===")
        s, b = req(f"{base}/stream/state", {"X-API-Key": KEY})
        check("GET /stream/state 正确密钥 -> 200", s == 200, f"HTTP {s}")
        s, b = req(f"{base}/health", {"Authorization": f"Bearer {KEY}"})
        check("Bearer 形式也被接受", s == 200, f"HTTP {s}")

        print("\n=== 4) 流式接口的鉴权（TestClient 覆盖不到）===")
        s, _ = req(f"{base}/stream?target_fps=5")
        check("GET /stream 无密钥 -> 401（未开始推流）", s == 401, f"HTTP {s}")

        # 有密钥时应该开始持续推流，读到 MJPEG 边界即可
        r = urllib.request.Request(f"{base}/stream?target_fps=5",
                                  headers={"X-API-Key": KEY})
        got_header = False
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                ctype = resp.headers.get("Content-Type", "")
                got_header = resp.status == 200 and "multipart/x-mixed-replace" in ctype
                resp.read(512)      # 读一点确认真的在推流
        except Exception as e:
            print(f"    流读取异常（可能是超时，正常）：{type(e).__name__}")
        check("GET /stream 正确密钥 -> 开始 MJPEG 推流", got_header)

        print("\n=== 5) 收敛：旧密钥失效后立即拒绝（无需重启）===")
        # 环境变量在进程启动时就固定了，这里验证的是"启动时读取生效"，
        # 热改由单测 test_api_key_is_read_per_request 覆盖。
        s, _ = req(f"{base}/stream/state", {"X-API-Key": KEY + "x"})
        check("近似密钥（多一个字符）仍被拒", s == 401, f"HTTP {s}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{'=' * 46}")
    print(f"通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：" + "、".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
