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
  1) `/stream?source=` 的本地路径会被限制在 `utils.paths.ALLOWED_SOURCE_ROOTS`（默认为项目目录）
     之内，避免未授权读取本机任意文件。需要放行别的目录时设环境变量
     `DEFECT_ALLOWED_SOURCE_ROOTS`（多个用 os.pathsep 分隔）。
     网络流地址（rtsp/http/rtmp）默认放行，因为摄像头接入是既定用法。
  2) **接口鉴权**（P0-7）：原实现无任何鉴权，内网任意一台机器都能调用 /detect
     与 /stream，甚至可以 `source=rtsp://...` 让服务去连任意外部地址。
     现在通过环境变量 `DEFECT_API_KEY` 启用：
       - 未设置：不鉴权，但 /health 会暴露 auth_enabled=false，便于运维发现
         「忘了配密钥」这个高危疏漏；
       - 已设置：除 `/`、`/health` 外的所有接口都要求请求头 `X-API-Key`
         （或 `Authorization: Bearer <key>`），不匹配即 401。
     用 `secrets.compare_digest` 做常数时间比较，避免时序侧信道。
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import threading
import time

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, StreamingResponse

from ..constants import CLASS_NAMES
from ..utils.imageio import draw_detections, encode_jpeg
from ..utils.logger import get_logger, prune_old_logs, record_decision
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

# /detect 上传上限：NEU-DET 图都是 200×200，但用户上传的真实产线图分辨率更大；
# 10MB 已足够 12MP 的 JPEG。设太低会拒绝合法输入，设太高会被恶意大文件耗内存。
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------- 全局状态
_backend = None            # 推理器实例
_backend_info: dict = {}   # /health 用的元信息

# 单例：同一时刻只允许一个流，避免多个客户端各自开线程造成资源竞争
_active_stream: StreamDetector | None = None
_stream_lock = threading.Lock()

# 模型加载失败的重试状态（对应部署报告 P0「异常自恢复」）
# 原实现的失效模式：启动时模型缺失 -> _backend 永久为 None -> 所有请求 503，
# 且没有任何重试 —— 7x24 无人值守下这意味着永久停机，只能人工重启。
_reload_lock = threading.Lock()
_last_reload_attempt: float = 0.0
_reload_failures: int = 0
_next_reload_at: float = 0.0

# 重试间隔（秒）：失败后从 INITIAL 指数退避到 MAX，避免磁盘故障时疯狂重试
RELOAD_INITIAL_DELAY = 5.0
RELOAD_MAX_DELAY = 300.0
RELOAD_MIN_INTERVAL = 5.0    # 两次尝试之间的硬下限，防请求打爆


# ---------------------------------------------------------------- 鉴权（P0-7）
def _api_key() -> str:
    """期望的 API key；空串表示未启用鉴权。每次请求都读环境变量，便于热改。"""
    return os.environ.get("DEFECT_API_KEY", "").strip()


def auth_enabled() -> bool:
    return bool(_api_key())


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """鉴权依赖：未配置 DEFECT_API_KEY 时放行，否则校验请求头。

    接受两种传法，覆盖常见客户端：
      - `X-API-Key: <key>`（推荐，语义最清晰）
      - `Authorization: Bearer <key>`（兼容标准 OAuth 风格的调用方）

    `/` 与 `/health` **刻意不挂这个依赖**：
      - `/health` 必须无鉴权，否则监控系统探活也得配密钥，且它在降级时
        是最重要的信息来源（此时更要能读到）；
      - `/` 只是内嵌页面的壳，真正的数据仍在受保护的 /stream 上。

    用 compare_digest 做常数时间比较：普通 `==` 会随前缀匹配长度提前返回，
    理论上可被时序分析逐字节猜出密钥。
    """
    expected = _api_key()
    if not expected:
        return
    token = (x_api_key or "").strip()
    if not token and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            token = value.strip()
    if not token or not secrets.compare_digest(token, expected):
        LOG.warning("鉴权失败：%s %s", request.method, request.url.path)
        raise HTTPException(
            status_code=401,
            detail="缺少或错误的 API key（请设置 X-API-Key 请求头）",
        )


def _load_backend_once() -> tuple[object | None, dict, str]:
    """尝试加载一次模型。返回 (backend, info, error_message)。

    与旧实现的关键差异：**任何异常都被捕获并转成错误信息**，而不是让
    服务启动失败。加载不来就进降级态，由后台线程继续重试。
    """
    intra_threads = os.environ.get("DEFECT_ONNX_THREADS")
    try:
        intra_threads = int(intra_threads) if intra_threads else 4
    except ValueError:
        LOG.warning("DEFECT_ONNX_THREADS=%r 不是整数，回退到 4", intra_threads)
        intra_threads = 4

    if ONNXDefectPredictor is not None and ONNX_PATH.exists():
        backend = ONNXDefectPredictor(
            str(ONNX_PATH), CLASS_NAMES, imgsz=640, conf=0.25, iou=0.5,
            intra_op_num_threads=intra_threads,
        )
        return backend, {
            "backend": "onnxruntime",
            "model": ONNX_PATH.name,
            "providers": backend.providers,
            "imgsz": backend.imgsz,
            "default_conf": backend.conf,
            "default_iou": backend.iou,
            "intra_op_num_threads": intra_threads,
        }, ""

    if DefectPredictor is not None and BEST_PT.exists():
        backend = DefectPredictor(str(BEST_PT), CLASS_NAMES, conf=0.25, iou=0.5)
        return backend, {
            "backend": "ultralytics(pytorch)",
            "model": BEST_PT.name,
            "providers": ["cpu/cuda via torch"],
            "imgsz": 640,
            "default_conf": backend.conf,
            "default_iou": backend.iou,
        }, ""

    return None, {}, f"未找到模型文件（{ONNX_PATH.name} 或 {BEST_PT.name}）"


def _init_backend() -> bool:
    """加载模型。成功返回 True；失败不抛异常，置降级态并安排重试。

    注意：**失败也要让服务起来**。服务活着才能通过 /health 报告问题、
    才能在模型文件恢复后自动恢复 —— 让进程直接退出反而是最差的选择。
    """
    global _backend, _backend_info, _reload_failures, _next_reload_at

    try:
        backend, info, err = _load_backend_once()
    except Exception as e:                      # 模型损坏/格式不对/依赖缺失
        LOG.exception("模型加载失败")
        backend, info, err = None, {}, f"{type(e).__name__}: {e}"

    with _reload_lock:
        if backend is not None:
            _backend, _backend_info = backend, info
            if _reload_failures:
                LOG.info("模型加载成功（此前失败 %d 次后已恢复）", _reload_failures)
            _reload_failures = 0
            _next_reload_at = 0.0
            return True
        _backend = None
        _backend_info = {"backend": None, "error": err}
        _reload_failures += 1
        delay = min(RELOAD_INITIAL_DELAY * (2 ** (_reload_failures - 1)),
                    RELOAD_MAX_DELAY)
        _next_reload_at = time.time() + delay
        LOG.error("模型加载失败（第 %d 次），%.0fs 后重试：%s",
                  _reload_failures, delay, err)
        return False


def _ensure_backend() -> None:
    """请求路径的兜底：模型不可用时触发一次（限频的）重试加载。

    为什么请求路径也要重试：后台线程有间隔，而产线希望"文件一恢复就能服务"。
    用 RELOAD_MIN_INTERVAL 限频，避免高频请求把重试变成 DoS。
    """
    if _backend is not None:
        return
    now = time.time()
    if now - _last_reload_attempt < RELOAD_MIN_INTERVAL:
        return
    globals()["_last_reload_attempt"] = now
    _init_backend()


def _reload_probe_loop(stop_evt: threading.Event) -> None:
    """后台线程：定期探测模型是否已可加载，成功后自动恢复正常服务。"""
    while not stop_evt.wait(1.0):
        if _backend is not None:
            continue
        if time.time() < _next_reload_at:
            continue
        globals()["_last_reload_attempt"] = time.time()
        _init_backend()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时加载模型；关闭时确保流线程与帧源被释放。"""
    _init_backend()
    # 启动时清理过期日志（保留期 30 天，对应部署报告「保留期 >= 30 天」）。
    # 放在启动而非请求路径：避免每次请求做目录扫描。
    pruned = prune_old_logs()
    LOG.info("服务启动：backend=%s model=%s 清理过期日志 %d 个",
             (_backend_info or {}).get("backend"),
             (_backend_info or {}).get("model"), pruned)

    # 后台探测线程：模型加载失败时定期重试，成功后自动恢复正常服务。
    # 这是「异常自恢复」的核心 —— 没有它，一次磁盘抖动就等于永久停机。
    stop_evt = threading.Event()
    probe = threading.Thread(target=_reload_probe_loop, args=(stop_evt,),
                             name="model-reload-probe", daemon=True)
    probe.start()
    try:
        yield
    finally:
        stop_evt.set()
        with _stream_lock:
            det = _active_stream
            globals()["_active_stream"] = None
        if det is not None:
            det.stop()
        LOG.info("服务关闭：流已释放，探测线程已停止")


app = FastAPI(title="Defect Detection API", version="1.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    # /stream 当前是单例设计：同一时间只允许一个活跃推流。
    # 暴露 active_stream 字段方便监控/排障：若 N 个客户端轮流访问 /stream，
    # 应看到这个值在 0/1 之间跳变；若长期为 1 而无人看，说明上次没正常断开。
    with _stream_lock:
        stream_active = _active_stream is not None
    with _reload_lock:
        reload_failures = _reload_failures
        next_reload_in = max(0.0, round(_next_reload_at - time.time(), 1))
    # degraded = "服务活着，但检测能力不可用"——这是最需要告警的状态。
    # 单独给一个布尔字段，方便监控系统不必解析 status 字符串。
    degraded = _backend is None
    return {
        "status": "ok" if _backend is not None else "model_missing",
        **_backend_info,
        "classes": CLASS_NAMES,
        "allowed_source_roots": [str(p) for p in ALLOWED_SOURCE_ROOTS],
        "stream_active": stream_active,
        "stream_mode": "single_instance",
        # ---- 自恢复相关（P0 可观测性）----
        # 模型加载失败时，monitoring 应看 reload_failures>0 或 status=model_missing；
        # 恢复后这两个字段自动归零，无需人工干预。
        "reload_failures": reload_failures,
        "next_reload_in_s": next_reload_in,
        "degraded": degraded,
        # 鉴权是否启用。**特意暴露成字段而不是静默**：产线部署最容易犯的
        # 疏漏就是忘配 DEFECT_API_KEY，那样接口对内网完全敞开。监控可以直接
        # 对该字段告警（auth_enabled=false 视为配置不合格）。
        "auth_enabled": auth_enabled(),
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


@app.post("/detect", response_model=DetectResponse,
          dependencies=[Depends(require_api_key)])
async def detect(file: UploadFile = File(...),
                 conf: float = Query(0.25, ge=0.0, le=1.0),
                 iou: float = Query(0.5, ge=0.0, le=1.0)):
    # 模型不可用时先触发一次（限频的）重试：文件若已恢复就无需人工干预。
    _ensure_backend()
    if _backend is None:
        raise HTTPException(
            status_code=503,
            detail=f"模型未加载（{(_backend_info or {}).get('error', '未知原因')}），"
                   f"服务会在后台自动重试",
        )

    # 大文件先看 Content-Length 提前拒绝，避免读到一半才发现超限
    # （没有 Content-Length 时 chunked 上传会落到下面读完后判断）
    declared = file.size or 0
    if declared and declared > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"文件过大，上限 {MAX_UPLOAD_MB}MB")

    # MIME 白名单：浏览器/工具按规范会带正确 content-type，但 curl -F 也可能
    # 发 application/octet-stream，所以仅做轻校验 + 真正以解码结果为准。
    ctype = (file.content_type or "").lower()
    if ctype and not ctype.startswith(("image/", "application/octet-stream")):
        raise HTTPException(status_code=415, detail=f"仅支持图片，收到 {ctype}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"文件过大，上限 {MAX_UPLOAD_MB}MB")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片，请上传 jpg/png")

    # conf/iou 作为调用参数传入，不再写回后端实例（旧实现改共享状态，
    # 并发请求会互相篡改阈值；且回退路径还会静默忽略 iou）。
    t0 = time.perf_counter()
    dets = _backend.predict_image(img, conf=conf, iou=iou)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 追溯记录（旁路，失败不影响返回）：现场需要回答
    # "某时刻这张图为什么判 NG"，没有这条记录就无法复盘。
    record_decision(
        source=f"upload:{file.filename or 'unnamed'}",
        detections=dets,
        elapsed_ms=elapsed_ms,
        conf=conf, iou=iou,
        model=(_backend_info or {}).get("model"),
    )

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


def _make_on_change(src, conf: float, iou: float):
    """构造流式「稳定状态变化」回调：落一条判定追溯记录。

    只在状态真正变化时触发（StreamDetector 内部已去抖），因此不会每帧写盘。
    """
    def _on_change(dets: list[dict], _frame) -> None:
        try:
            record_decision(
                source=f"stream:{src}",
                detections=dets,
                conf=conf, iou=iou,
                model=(_backend_info or {}).get("model"),
            )
        except Exception:  # pragma: no cover - 追溯是旁路，绝不打断推流
            LOG.exception("流式判定追溯写入失败（已忽略）")
    return _on_change


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
        on_change=_make_on_change(src, conf, iou),
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


@app.get("/stream", dependencies=[Depends(require_api_key)])
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

    **并发限制**：当前是单例流——同一时间只允许一个活跃 /stream。
    新连接会先停掉旧流（_swap_active 在锁内交换指针、锁外 stop）。
    若需要多客户端并发，建议在前置反代（nginx/haproxy）上做：
      - 要么多 worker 部署 + 反代轮询；
      - 要么反代把 N 个客户端 fan-out 到一个上游流（避免每客户端起一份推理）。
    项目定位为面试/演示场景，单例设计是有意为之，避免引入 stream pool 的复杂度。
    """
    _ensure_backend()
    if _backend is None:
        raise HTTPException(
            status_code=503,
            detail=f"模型未加载（{(_backend_info or {}).get('error', '未知原因')}），"
                   f"服务会在后台自动重试",
        )

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


@app.get("/stream/state", dependencies=[Depends(require_api_key)])
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


@app.post("/stream/stop", dependencies=[Depends(require_api_key)])
def stop_stream():
    """手动停止当前推流（客户端主动断开之外的显式停止方式）。"""
    old = _swap_active(None)
    if old is None:
        return {"stopped": False, "detail": "当前没有活跃的流"}
    _stop_quietly(old)          # 锁外停止，避免持锁 join 线程
    return {"stopped": True}
