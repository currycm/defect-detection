"""统一日志 + 判定追溯。

设计目标（对应产线部署 P0 项）：
  1) **日志必须落盘**：原实现只有 StreamHandler，容器重启/进程崩溃后日志全丢，
     现场出问题（"昨天 14:23 那张图为什么判 NG"）永远答不了。
  2) **按天滚动 + 保留期**：7x24 运行不能无限写盘，超出保留期的自动清理。
  3) **判定结果结构化追溯**：每条判定以 JSON Lines 落一份记录，含时间戳、
     输入源、模型版本、阈值、耗时、逐框结果——这是定位现场误判的唯一证据链。

几个刻意的设计取舍：
  - **惰性初始化**：import 本模块不建文件、不建目录。否则任何 import
    （包括单测、文档工具）都会产生副作用和垃圾文件。
  - **测试环境自动降级**：检测到 pytest 在跑（`PYTEST_CURRENT_TEST`）或显式设置
    `DEFECT_LOG_DISABLE=1` 时，只挂 StreamHandler，不写盘。避免测试污染仓库。
  - **日志目录可配**：`DEFECT_LOG_DIR`，默认 `<项目根>/logs`。容器里挂卷即可持久化。
  - **失败不致命**：日志系统自身出问题（磁盘满、无权限）绝不能拖垮推理服务，
    因此所有落盘操作都吞异常并退回控制台输出。
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
import time
from pathlib import Path
from typing import Any

# 日志文件与保留策略
LOG_FILENAME = "defect.log"
TRACE_FILENAME = "decisions.jsonl"
DEFAULT_BACKUP_DAYS = 30
DEFAULT_MAX_BYTES = 32 * 1024 * 1024   # 单文件最大 32MB，超过即滚动
DEFAULT_BACKUP_COUNT = 10

_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

# 记录哪些 logger 已经配过 FileHandler，避免重复添加（每次调用都加会重复输出）
_configured: set[str] = set()
_config_lock = threading.Lock()

# 追溯写入用的锁：多线程（流式推理线程 + 请求线程）会并发写同一文件
_trace_lock = threading.Lock()


def log_dir() -> Path:
    """日志目录。环境变量 `DEFECT_LOG_DIR` 优先，否则用项目根的 logs/。

    import 期不调用本函数，只在真正要写文件时才解析（惰性）。
    """
    env = os.environ.get("DEFECT_LOG_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    try:
        # 延迟导入：避免 logger -> paths -> logger 的循环依赖
        from .paths import PROJECT_ROOT
        return PROJECT_ROOT / "logs"
    except Exception:  # pragma: no cover - 极端情况下退回 cwd
        return Path.cwd() / "logs"


def _should_write_files() -> bool:
    """是否需要落盘。

    关闭条件（任一）：
      - `DEFECT_LOG_DISABLE=1`：显式关闭（容器/测试用）
      - 正在跑 pytest：避免单测在仓库里产生日志垃圾

    注意（实测踩过的坑）：pytest 会在**每个测试阶段开始时重新写入**
    `PYTEST_CURRENT_TEST`（值形如 "tests/x.py::test_a (call)"）。因此在 fixture
    里 `delenv` 或 `setenv("")` 都挡不住——测试体真正执行时它已被重新塞回。
    测试若需要落盘，请 monkeypatch 本函数本身（见 tests/test_logger_and_recovery.py），
    不要试图操纵环境变量。
    """
    if os.environ.get("DEFECT_LOG_DISABLE", "").strip() in ("1", "true", "True"):
        return False
    # 跑 pytest 时不落盘，避免单测在仓库里产生日志垃圾
    return not os.environ.get("PYTEST_CURRENT_TEST")


def _build_file_handler() -> logging.Handler | None:
    """构建按大小滚动的文件处理器；任何异常都返回 None 而不是抛出。"""
    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            d / LOG_FILENAME,
            maxBytes=DEFAULT_MAX_BYTES,
            backupCount=DEFAULT_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        return handler
    except Exception:  # pragma: no cover - 磁盘满/无权限时不应拖垮服务
        return None


def get_logger(name: str = "defect") -> logging.Logger:
    """取 logger。控制台 + （可选的）落盘双通道。

    幂等：同名 logger 只配置一次，重复调用不会叠加 handler。
    """
    logger = logging.getLogger(name)
    with _config_lock:
        if name not in _configured:
            _configured.add(name)
            if not logger.handlers:
                stream = logging.StreamHandler()
                stream.setFormatter(logging.Formatter(_LOG_FORMAT))
                logger.addHandler(stream)
            logger.setLevel(logging.INFO)
            if _should_write_files():
                fh = _build_file_handler()
                if fh is not None:
                    logger.addHandler(fh)
            # 避免日志同时冒泡到 root 造成重复输出
            logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 判定追溯（JSON Lines）
# ---------------------------------------------------------------------------
def _model_version_hint() -> str:
    """模型版本标识：优先环境变量，否则空串（由调用方填具体模型名）。"""
    return os.environ.get("DEFECT_MODEL_VERSION", "").strip()


def record_decision(
    *,
    source: str,
    detections: list[dict],
    elapsed_ms: float | None = None,
    conf: float | None = None,
    iou: float | None = None,
    model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """记录一条判定结果到 `decisions.jsonl`（每行一个 JSON 对象）。

    为什么单独一个文件而不是混在普通日志里：
      普通日志是给"人看"的（排障），追溯记录是给"机器查"的（统计/回溯/验收）。
      混在一起会让双方都难用，且文本日志滚动后格式不可靠。

    **绝不抛异常**：追溯是旁路能力，写失败不能影响检测主流程。
    """
    if not _should_write_files():
        return
    rec: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "ts_unix": round(time.time(), 3),
        "source": str(source),
        "model": model or _model_version_hint(),
        "count": len(detections),
        # 只留关键字段，避免把整帧坐标数组无节制写盘
        "detections": [
            {
                "cls": int(d.get("cls", -1)),
                "name": str(d.get("name", "")),
                "conf": round(float(d.get("conf", 0.0)), 4),
            }
            for d in detections
        ],
    }
    if elapsed_ms is not None:
        rec["elapsed_ms"] = round(float(elapsed_ms), 2)
    if conf is not None:
        rec["conf_thr"] = round(float(conf), 4)
    if iou is not None:
        rec["iou_thr"] = round(float(iou), 4)
    if extra:
        rec["extra"] = extra

    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False)
        with _trace_lock, open(d / TRACE_FILENAME, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # pragma: no cover - 追溯失败不影响检测
        pass


def prune_old_logs(keep_days: int = DEFAULT_BACKUP_DAYS) -> int:
    """清理超过保留期的日志文件，返回删除数量。

    供服务启动时调用一次（不要放进请求路径）。保留期默认 30 天，
    对应部署报告里「保留期 >= 30 天」的验收要求。
    """
    if not _should_write_files():
        return 0
    deleted = 0
    try:
        d = log_dir()
        if not d.is_dir():
            return 0
        cutoff = time.time() - keep_days * 86400
        for p in d.iterdir():
            if not p.is_file():
                continue
            # 只清理本模块管理的文件，避免误删用户放在 logs/ 里的其他东西
            if not (p.name.startswith(LOG_FILENAME)
                    or p.name.startswith(TRACE_FILENAME)):
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    deleted += 1
            except OSError:
                continue
    except Exception:  # pragma: no cover
        return deleted
    return deleted


def reset_for_tests() -> None:
    """测试辅助：清掉 _configured 缓存，让下一次 get_logger 重新配置。

    不放进 __all__，也不在正常运行路径调用。
    """
    with _config_lock:
        _configured.clear()


__all__ = [
    "DEFAULT_BACKUP_DAYS",
    "LOG_FILENAME",
    "TRACE_FILENAME",
    "get_logger",
    "log_dir",
    "prune_old_logs",
    "record_decision",
]
