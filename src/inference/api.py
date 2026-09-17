"""FastAPI 推理服务（Step 7 + Step 8b 实时流式）。

优先用 ONNXRuntime 加载 weights/best.onnx（不依赖 torch，启动快、镜像小）；
若 ONNX 不存在则回退到 ultralytics 的 .pt。

对外接口：
  GET  /              极简可视化页（<img src="/stream"> 直接看实时画面）
  GET  /health        服务与模型状态（后端 provider、输入尺寸、类别表）
  POST /detect        上传图片做检测，可用 query 参数覆盖 conf / iou
  GET  /stream        MJPEG 实时推流（客户端连接=启动，断开=自动释放）
  GET  /stream/state  轮询最新稳定状态
  POST /stream/stop   显式停止当前推流

阈值说明（来自 Step 6b 调优）：
  - NMS iou 默认 0.5（调优结论：比默认 0.7 在 test 上 mAP@0.5 高 +0.019）
  - conf 默认 0.25：服务返回给用户看的实用档；
    需要极限召回可传更低的 conf（代价是返回更多低置信度框）。

安全说明：
  `/stream?source=` 的本地路径会被限制在 `utils.paths.ALLOWED_SOURCE_ROOTS`（默认为项目目录）
  之内，避免未授权读取本机任意文件。需要放行别的目录时设环境变量
  `DEFECT_ALLOWED_SOURCE_ROOTS`（多个用 os.pathsep 分隔）。
  网络流地址（rtsp/http/rtmp）默认放行，因为摄像头接入是既定用法；
  若部署在不可信网络，请自行收紧或在前置反代上加鉴权。
"""
from __future__ import annotations

import asyncio
import contextlib
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from ..constants import CLASS_NAMES
from ..utils.imageio import draw_detections, encode_jpeg
from ..utils.logger import get_logger
from ..utils.paths import (
    ALLOWED_SOURCE_ROOTS,
    BEST_PT,
    ONNX_PATH,
    PROCESSED_DIR,
    is_path_allowed,
)
from .schemas import DetectResponse, to_detection
from .stream import StreamDetector

LOG = get_logger("api")

try:  # ONNX 路径（首选）
    from .onnx_predictor import ONNXDefectPredictor
except Exception:  # pragma: no cover
    LOG.exception("ONNX 推理器不可用，将回退到 ultralytics")
    ONNXDefectPredictor = None  # type: ignore

try:  # 回退路径
    from .predictor import DefectPredictor
except Exception:  # pragma: no cover
    LOG.exception("ultralytics 回退推理器也不可用")
    DefectPredictor = None  # type: ignore

DEFAULT_TEST_DIR = PROCESSED_DIR / "images" / "test"

# ---------------------------------------------------------------- 全局状态
_backend = None            # 推理器实例
_backend_info: dict = {}   # /health 用的元信息

# 单例：同一时刻只允许一个流，避免多个客户端各自开线程造成资源竞争
_active_stream: StreamDetector | None = None
_stream_lock = threading.Lock()


def _init_backend() -> None:
    """启动时加载一次模型，优先 ONNX。"""
    global _backend, _backend_info

    if ONNXDefectPredictor is not None and ONNX_PATH.exists():
        _backend = ONNXDefectPredictor(
            str(ONNX_PATH), CLASS_NAMES, imgsz=640, conf=0.25, iou=0.5
        )
        _backend_info = {
            "backend": "onnxruntime",
            "model": ONNX_PATH.name,
            "providers": _backend.providers,
            "imgsz": _backend.imgsz,
            "default_conf": _backend.conf,
            "default_iou": _backend.iou,
        }
    elif DefectPredictor is not None and BEST_PT.exists():
        _backend = DefectPredictor(str(BEST_PT), CLASS_NAMES, conf=0.25, iou=0.5)
        _backend_info = {
            "backend": "ultralytics(pytorch)",
            "model": BEST_PT.name,
            "providers": ["cpu/cuda via torch"],
            "imgsz": 640,
            "default_conf": _backend.conf,
            "default_iou": _backend.iou,
        }
    else:
        _backend = None
        _backend_info = {"backend": None, "error": "未找到 best.onnx 或 best.pt"}


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时加载模型；关闭时确保流线程与帧源被释放。"""
    _init_backend()
    try:
        yield
    finally:
        with _stream_lock:
            det = _active_stream
            globals()["_active_stream"] = None
        if det is not None:
            det.stop()


app = FastAPI(title="Defect Detection API", version="1.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if _backend is not None else "model_missing",
        **_backend_info,
        "classes": CLASS_NAMES,
        "allowed_source_roots": [str(p) for p in ALLOWED_SOURCE_ROOTS],
    }


# 极简可视化页：打开 http://127.0.0.1:8000/ 即可看到实时带框画面
INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>钢材表面缺陷 · 实时检测</title>
<style>
 body{margin:0;background:#f5f6f8;font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#222}
 .wrap{max-width:760px;margin:28px auto;padding:0 16px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px}
 p.sub{margin:0 0 16px;color:#666;font-size:13px}
 .card{background:#fff;border:1px solid #e3e5e9;border-radius:10px;padding:12px}
 img{width:100%;display:block;background:#ddd;border-radius:6px;min-height:240px}
 .bar{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
 button{border:1px solid #d0d3d8;background:#fff;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:13px}
 button:hover{background:#f0f1f3}
 .hint{color:#888;font-size:12px;width:100%}
 .err{color:#c0392b}
</style></head><body><div class="wrap">
<h1>钢材表面缺陷 · 实时检测</h1>
<p class="sub">6 类：crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches　·　MJPEG 流，浏览器直接显示</p>
<div class="card">
  <img id="v" alt="点击“模拟流”或“摄像头”加载实时流" onerror="failed()">
  <div class="bar">
    <button onclick="start('')">模拟流</button>
    <button onclick="start('0')">摄像头</button>
    <button onclick="pause()">停止</button>
    <span class="hint" id="st">“模拟流”= test 集循环。“摄像头”由服务端经 OpenCV 直接读取本机摄像头，不需要浏览器授权（所以嵌在应用内预览里也能用）。</span>
  </div>
</div>
</div>
<script>
 var cur = '';
 function set(msg, err){
   var el = document.getElementById('st');
   el.textContent = msg;
   el.className = err ? 'hint err' : 'hint';
 }
 function start(source){
   cur = source;
   var q = '/stream?target_fps=15&detect_interval=0.1&t=' + Date.now();
   if (source) q += '&source=' + encodeURIComponent(source);
   set('已连接：' + (source ? '摄像头 ' + source : '模拟流（test 集）') + '　—　点“停止”释放。');
   document.getElementById('v').src = q;
 }
 async function pause(){
   cur = '';
   await fetch('/stream/stop', {method:'POST'});
   document.getElementById('v').src = '';
   set('已停止，资源已释放。');
 }
 function failed(){
   if (!cur) return;            // 主动停止时 img 也会触发 error，不该报错
   set('加载失败：摄像头可能被其它程序占用（例如网页版摄像头已授权），可先试“模拟流”。', true);
   cur = '';
 }
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...),
                 conf: float = Query(0.25, ge=0.0, le=1.0),
                 iou: float = Query(0.5, ge=0.0, le=1.0)):
    if _backend is None:
        raise HTTPException(status_code=503, detail="模型未加载，请先导出权重")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片，请上传 jpg/png")

    # conf/iou 作为调用参数传入，不再写回后端实例（旧实现改共享状态，
    # 并发请求会互相篡改阈值；且回退路径还会静默忽略 iou）。
    dets = _backend.predict_image(img, conf=conf, iou=iou)

    return DetectResponse(
        detections=[to_detection(d, CLASS_NAMES) for d in dets],
        count=len(dets),
    )


# ---------------------------------------------------------------- 实时流式检测
_REMOTE_SCHEMES = ("rtsp://", "http://", "https://", "rtmp://")


def _resolve_source(source: str):
    """把 query 参数解析成 StreamDetector 能识别的输入源。

    - ""                -> 默认用 test 集做合成流（无摄像头也能演示/验证）
    - "0" / "1" 等数字   -> 摄像头索引
    - "synthetic:<路径>" -> 合成帧源（图片或目录）
    - rtsp/http/rtmp URL -> 网络流
    - 其它               -> 视频文件路径

    本地路径一律要求落在 ALLOWED_SOURCE_ROOTS 内，否则 400 —— 这是为了防止
    `/stream?source=` 被用来读取并回传本机任意文件（实测未加限制时可成功推流项目外目录）。
    """
    if not source:
        return f"synthetic:{DEFAULT_TEST_DIR}"
    s = source.strip()
    if s.isdigit():
        return int(s)
    if s.startswith("synthetic:"):
        target = s.split(":", 1)[1]
        if not is_path_allowed(target):
            raise HTTPException(status_code=400, detail=(
                "路径不在允许的根目录内；默认仅允许项目目录，"
                "如需放行请设置环境变量 DEFECT_ALLOWED_SOURCE_ROOTS"))
        return s
    if s.lower().startswith(_REMOTE_SCHEMES):
        LOG.warning("接受网络流输入: %s（部署在不可信网络时请自行收紧）", s)
        return s
    if not is_path_allowed(s):
        raise HTTPException(status_code=400, detail=(
            "文件路径不在允许的根目录内；默认仅允许项目目录，"
            "如需放行请设置环境变量 DEFECT_ALLOWED_SOURCE_ROOTS"))
    return s


def _new_detector(src, *, conf, iou, target_fps, detect_interval, hold) -> StreamDetector:
    return StreamDetector(
        _backend, src,
        target_fps=target_fps,
        detect_interval=detect_interval,
        stable_frames=3,
        conf=conf, iou=iou,
        loop=True, hold=hold,
        allowed_roots=ALLOWED_SOURCE_ROOTS,
        draw=draw_detections,
    )


def _swap_active(new: StreamDetector | None) -> StreamDetector | None:
    """在锁内完成单例指针交换，返回被替换下来的旧流（由调用方在锁外停止）。

    注意不要在持锁时调用 stop()：stop() 会 join 线程（最长 5s），
    持锁做慢操作会让其他请求排队抖动。
    """
    global _active_stream
    with _stream_lock:
        old = _active_stream
        _active_stream = new
    return old


def _stop_quietly(det: StreamDetector | None) -> None:
    if det is None:
        return
    try:
        det.stop()
    except Exception:
        LOG.exception("停止流时出错（已忽略）")


@app.get("/stream")
async def stream(request: Request,
                 source: str = "",
                 conf: float = Query(0.25, ge=0.0, le=1.0),
                 iou: float = Query(0.5, ge=0.0, le=1.0),
                 target_fps: float = Query(15.0, ge=1.0, le=120.0),
                 detect_interval: float = Query(0.1, gt=0.0, le=60.0),
                 hold: int = Query(15, ge=1, le=10_000)):
    """MJPEG 实时推流：浏览器 <img src=".../stream"> 即可持续看到带框画面。

    启停由请求生命周期管理：客户端连接 = 启动，断开 = 自动停止并释放资源。
    hold 仅作用于默认的 synthetic 演示源（每张图连续播 N 帧，模拟静止场景）。

    断开检测用 request.is_disconnected() 显式轮询：只依赖框架内部的
    listen_for_disconnect 不够可靠（实测断开后要十几秒才回收），
    显式轮询可把回收延迟压到一次循环内。
    """
    if _backend is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    src = _resolve_source(source)

    async def gen():
        det = _new_detector(src, conf=conf, iou=iou, target_fps=target_fps,
                            detect_interval=detect_interval, hold=hold)
        _stop_quietly(_swap_active(det))
        det.start()
        interval = 1.0 / target_fps
        try:
            while det.running:
                if await request.is_disconnected():
                    break
                frame, _ = det.latest()
                if frame is None:
                    await asyncio.sleep(0.02)
                    continue
                # JPEG 编码是 CPU 活，放线程池执行，别阻塞事件循环
                payload = await asyncio.to_thread(encode_jpeg, frame)
                if payload:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
                           + payload + b"\r\n")
                await asyncio.sleep(interval)
        finally:
            # 客户端断开 / 异常都会走到这里：确保线程结束、帧源释放
            _stop_quietly(det)
            with _stream_lock:
                if _active_stream is det:
                    globals()["_active_stream"] = None

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream/state")
def stream_state():
    """轮询接口：返回最新的稳定检测状态（供不方便用 MJPEG 的客户端）。

    取引用与判空都在锁内完成 —— 旧实现在判空之后无锁解引用，
    与 /stream/stop 并发时会 AttributeError -> 500（实测 1147 次请求中出现 1 次）。
    """
    with _stream_lock:
        det = _active_stream
    if det is None:
        return {"active": False, "detections": [], "count": 0}
    _, dets = det.latest()
    return {
        "active": det.running,
        "paused": det.paused,
        "count": len(dets),
        "detections": [to_detection(d, CLASS_NAMES).model_dump() for d in dets],
        "stable_signature_size": len(det.stabilizer.stable_signature),
        "stats": det.stats(),
    }


@app.post("/stream/stop")
def stop_stream():
    """手动停止当前推流（客户端主动断开之外的显式停止方式）。"""
    old = _swap_active(None)
    if old is None:
        return {"stopped": False, "detail": "当前没有活跃的流"}
    _stop_quietly(old)          # 锁外停止，避免持锁 join 线程
    return {"stopped": True}
