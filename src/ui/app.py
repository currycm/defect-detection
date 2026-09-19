"""Step 8：Gradio 可视化演示界面（面试可直接现场演示）。

复用 Step 7 的 ONNXRuntime 推理器，保证「演示效果 == 线上部署效果」，
不会出现 demo 好看、上线跑偏的情况。

功能：
  - 上传钢材表面图片，实时看到缺陷框 + 类别 + 置信度
  - conf / NMS iou 两个滑块实时调节，直观展示工作点权衡
  - 右侧表格给出每个框的坐标与置信度

修复记录（L1）：原实现在模块顶层做 `os.chdir(PROJECT)` + 硬编码绝对路径 +
导入即加载 ONNX 模型，导致
  1) import 本模块会改变调用方的当前工作目录（污染副作用）；
  2) 换机器/换目录必须改源码；
  3) 任何 import 都会加载 ~12MB 模型，无法在测试里导入。
现在改为：项目根由 src.utils.paths 统一解析（可被环境变量 DEFECT_PROJECT_ROOT
覆盖），模型改为首次调用时惰性加载。

运行：
    python -m src.ui.app       # 推荐（项目根需在 sys.path）
    python src/ui/app.py       # 也支持，内部会自动定位项目根
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# 引导：让「以脚本方式运行」也能 import src.*
# 仅此处保留一份最小候选链（与 src/utils/paths.py 同策略）：
# 环境变量 -> 本文件位置推导，**每个候选都必须含 configs/data.yaml 才被采纳**。
# 之所以不复用 paths.py，是因为该模块本身还需要先有 sys.path 才能导入
# （先有鸡还是先有蛋）。这里**不写任何本机绝对路径**，否则会随代码一起公开。
# ---------------------------------------------------------------------------
def _ensure_project_root_on_path() -> None:
    marker = Path("configs") / "data.yaml"
    candidates: list[Path] = []
    env = os.environ.get("DEFECT_PROJECT_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    with contextlib.suppress(OSError):
        candidates.append(Path(__file__).resolve().parents[2])

    for cand in candidates:
        try:
            if (cand / marker).exists():
                p = str(cand.resolve())
                if p not in sys.path:
                    sys.path.insert(0, p)
                # 注意：**不**做 os.chdir，避免 import 产生全局副作用
                return
        except OSError:
            continue


_ensure_project_root_on_path()

import cv2
import gradio as gr
import numpy as np

from src.constants import CLASS_NAMES, CLASS_NAMES_ZH
from src.inference.onnx_predictor import ONNXDefectPredictor
from src.utils.imageio import CLASS_COLORS, load_image
from src.utils.paths import ONNX_PATH, PROCESSED_DIR

CLASS_DESC = CLASS_NAMES_ZH
TEST_DIR = PROCESSED_DIR / "images" / "test"

_PREDICTOR: ONNXDefectPredictor | None = None


def get_predictor() -> ONNXDefectPredictor:
    """惰性加载 ONNX 推理器（首次调用时才读模型文件）。"""
    global _PREDICTOR
    if _PREDICTOR is None:
        if not Path(ONNX_PATH).exists():
            raise FileNotFoundError(
                f"未找到 ONNX 模型: {ONNX_PATH}\n请先运行 python scripts/export.py"
            )
        _PREDICTOR = ONNXDefectPredictor(
            str(ONNX_PATH), CLASS_NAMES, imgsz=640, conf=0.25, iou=0.5
        )
    return _PREDICTOR


def _to_bgr(image) -> np.ndarray | None:
    """Gradio 传入 RGB numpy 或文件路径，统一转成 BGR 供 cv2 使用。"""
    if image is None:
        return None
    if isinstance(image, (str, os.PathLike)):
        return load_image(image)  # 中文路径安全
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _load_font(size: int = 14):
    """加载中文字体，用于在图上写中文类别名；找不到返回 None。"""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except OSError:
                continue
    return None


def _draw(img_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    """画框并返回 RGB 图（Gradio 显示用）。中文标签用 PIL 绘制。"""
    out = img_bgr.copy()
    font = _load_font(14)

    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        color = CLASS_COLORS.get(d["cls"], (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        name = d["name"]
        text = f"{name} {CLASS_DESC.get(name, '')} {d['conf']:.2f}"
        if font is not None:
            try:
                from PIL import Image, ImageDraw
                pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                # 用字体实际bbox测量宽高，避免中英文混排时文字溢出底色；
                # 并做边界钳制，防止靠右/靠上的框标签被裁掉。
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                tw, th = (right - left) + 8, (bottom - top) + 6
                tx = min(max(x1, 0), max(pil.size[0] - tw, 0))
                ty = max(y1 - th - 2, 0)
                draw.rectangle([tx, ty, tx + tw, ty + th], fill=color)
                draw.text((tx + 4 - left, ty + 3 - top), text,
                          fill=(255, 255, 255), font=font)
                out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                continue
            except Exception:
                pass
        # 无中文字体时退化为英文标签（cv2.putText 画不了中文）
        cv2.putText(out, f"{name} {d['conf']:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def predict(image, conf: float, iou: float):
    """Gradio 回调：返回 (标注图 RGB, 结果表, 摘要 Markdown)。"""
    if image is None:
        return None, [], "请先上传一张钢材表面图片"

    img_bgr = _to_bgr(image)
    if img_bgr is None:
        return None, [], "图片解码失败，请换一张 jpg/png"

    try:
        preds = get_predictor().predict_image(img_bgr, conf=conf, iou=iou)
    except FileNotFoundError as e:
        return None, [], str(e)

    # 按置信度降序，保证摘要里的「最高置信度」确实是最高
    dets = sorted(preds, key=lambda d: d["conf"], reverse=True)
    annotated = _draw(img_bgr, dets)

    rows = [[d["name"], CLASS_DESC.get(d["name"], ""), round(d["conf"], 4),
             round(d["xyxy"][0], 1), round(d["xyxy"][1], 1),
             round(d["xyxy"][2], 1), round(d["xyxy"][3], 1)] for d in dets]

    if dets:
        stat: dict[str, int] = {}
        for d in dets:
            stat[d["name"]] = stat.get(d["name"], 0) + 1
        summary = (f"共检出 **{len(dets)}** 个缺陷　｜　"
                   f"最高置信度 {dets[0]['conf']:.3f}　｜　" +
                   "、".join(f"{k}×{v}" for k, v in sorted(stat.items())))
    else:
        summary = "未检出缺陷（可把置信度阈值调低试试）"

    return annotated, rows, summary


def build_examples() -> list[list[str]]:
    """从 test 集每类挑一张作为示例。"""
    paths: list[list[str]] = []
    if not TEST_DIR.is_dir():
        return paths
    names = sorted(p.name for p in TEST_DIR.iterdir() if p.is_file())
    for cls in CLASS_NAMES:
        for f in names:
            if f.lower().startswith(cls) and f.lower().endswith((".jpg", ".png")):
                paths.append([str(TEST_DIR / f)])
                break
    return paths


def build_ui() -> gr.Blocks:
    title = "钢材表面缺陷检测 · YOLOv8 + ONNXRuntime"
    desc = (
        "上传钢材表面图片，实时检测 6 类缺陷。"
        "推理链路为 **ONNXRuntime**（与线上部署一致，不依赖 PyTorch）。\n\n"
        "模型：YOLOv8n · NEU-DET · test 集 **mAP@0.5 = 0.787**、mAP@0.5:0.95 = 0.410"
    )

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{desc}")

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="上传图片", type="numpy", height=320)
                gr.Markdown(
                    "上传区自带的**摄像头按钮**走的是浏览器 `getUserMedia`，需要网页摄像头授权："
                    "请在独立浏览器（Edge/Chrome）里打开本页并在弹窗点「允许」；"
                    "若页面嵌在应用内预览中，通常拿不到该权限，点了会没有反应。\n\n"
                    "**要在预览里也能用摄像头**，请用推理服务页的「摄像头」按钮 "
                    "http://127.0.0.1:8000/ —— 它由服务端 OpenCV 直接取流，不需要任何浏览器授权。"
                )
                conf = gr.Slider(0.01, 0.9, value=0.25, step=0.01,
                                 label="置信度阈值 conf（越高越保守：框更少更准）")
                iou = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                label="NMS IoU（调优结论：0.5 优于默认 0.7）")
                btn = gr.Button("开始检测", variant="primary")
            with gr.Column(scale=1):
                out_img = gr.Image(label="检测结果", height=320)
                out_md = gr.Markdown("等待检测…")
                out_tbl = gr.Dataframe(
                    headers=["类别", "中文", "置信度", "x1", "y1", "x2", "y2"],
                    label="检测明细", wrap=True,
                )

        btn.click(predict, inputs=[inp, conf, iou],
                  outputs=[out_img, out_tbl, out_md])
        inp.change(predict, inputs=[inp, conf, iou],
                   outputs=[out_img, out_tbl, out_md])

        examples = build_examples()
        if examples:
            gr.Markdown("### 示例图片（点击直接体验）")
            gr.Examples(examples=examples, inputs=[inp],
                        outputs=[out_img, out_tbl, out_md],
                        fn=predict, cache_examples=False)

        gr.Markdown(
            "---\n"
            "**阈值怎么选？** 追求召回（宁可多框不可漏检）→ conf 调到 0.05 左右；"
            "不能容忍误报 → conf 调到 0.6 以上（test 集 P=0.858 / R=0.592）。"
        )
    return demo


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    build_ui().launch(server_name="127.0.0.1", server_port=port, share=False)
