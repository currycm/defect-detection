"""启动 FastAPI 推理服务。

用法：
    python scripts/serve.py                       # 默认 127.0.0.1:8000
    python scripts/serve.py --host 0.0.0.0        # 对外监听（需自行加鉴权/反代）
    python scripts/serve.py --port 8001

关于绑定地址（H1）：默认只绑 **127.0.0.1**。此前默认 0.0.0.0，
配合当时未做白名单的 /stream?source= 参数，局域网内任意主机都能读取
并回传本机任意文件；现在即便对外暴露，路径白名单也会兜底拒绝项目外路径，
但仍建议对外时前置鉴权。
"""
from __future__ import annotations

import argparse

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

import uvicorn


def main() -> int:
    ap = argparse.ArgumentParser(description="缺陷检测推理服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"启动服务: http://{args.host}:{args.port}")
    print("  健康检查: GET /health")
    print("  检测接口: POST /detect  (form-data: file=@图片; 可选 conf / iou)")
    print("  实时推流: GET /stream   (浏览器 <img src=...> 直接看)")
    if args.host not in ("127.0.0.1", "localhost"):
        print("  ⚠ 已对外监听，请确保前置鉴权或仅在内网使用")
    uvicorn.run("src.inference.api:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
