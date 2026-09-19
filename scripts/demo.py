"""启动 Gradio 演示界面（Step 8）。

用法：
    python scripts/demo.py                 # 默认 127.0.0.1:7860
    python scripts/demo.py --port 7861
"""
from __future__ import annotations

import argparse

from _bootstrap import ensure_project_root

PROJECT = ensure_project_root()

from src.ui.app import build_ui


def main() -> int:
    ap = argparse.ArgumentParser(description="Gradio 缺陷检测演示")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true",
                    help="生成 Gradio 公网临时链接（默认关闭）")
    args = ap.parse_args()

    print(f"启动 Gradio 演示: http://{args.host}:{args.port}")
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
