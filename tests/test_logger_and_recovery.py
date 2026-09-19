"""P0 可运维性回归：日志落盘 / 判定追溯 / 模型加载异常自恢复。

对应的三条真实缺陷（产线部署评估 P0）：
  P0-6 原实现只有 StreamHandler，容器重启后 7x24 的历史判定全丢，
       出现"昨天 14:23 那张图为什么判 NG"永远无法回答。
  P0-6 判定结果没有结构化记录，误判无法回溯统计。
  P0-7 启动时模型缺失 -> `_backend` 永久为 None -> 所有请求 503 且**永不重试**，
       无人值守场景等于永久停机，只能人工重启。现在应为：服务照常启动、
       明确报告降级、后台自动重试、文件恢复后无需人工干预即恢复。

这些测试**不需要真实权重**：自恢复用 monkeypatch 替换加载函数，
日志用 tmp_path + DEFECT_LOG_DIR 隔离。
"""
import json
import logging
import sys
import time
import types

import pytest

from src.utils import logger as logmod

# ---------------------------------------------------------------- 夹具


@pytest.fixture
def log_home(tmp_path, monkeypatch):
    """把日志目录指向 tmp_path，并强制开启落盘。

    为什么要 monkeypatch `_should_write_files`（而不是操纵环境变量）：
    `_should_write_files()` 在跑 pytest 时默认返回 False（防止单测在仓库里
    堆日志垃圾），这是**正确的生产设计**。但 pytest 会在**每个测试阶段开始时
    重新写入** `PYTEST_CURRENT_TEST`，所以在 fixture 里 `delenv` / `setenv("")`
    全部无效 —— 测试体执行时它已被重新塞回。（这一点已用探针脚本实测确认）
    因此直接替换判定函数，是唯一稳定的做法。
    """
    monkeypatch.setenv("DEFECT_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("DEFECT_LOG_DISABLE", raising=False)
    monkeypatch.setattr(logmod, "_should_write_files", lambda: True)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_logger_state():
    """每个用例前后重置 logger 配置缓存，避免用例间相互污染。"""
    logmod.reset_for_tests()
    yield
    logmod.reset_for_tests()


# ---------------------------------------------------------------- 日志落盘
def test_log_dir_env_override_wins(log_home):
    assert logmod.log_dir() == log_home


def test_log_dir_defaults_to_project_logs(monkeypatch, tmp_path):
    """没有 DEFECT_LOG_DIR 时应落在项目根的 logs/，而不是 cwd。"""
    monkeypatch.delenv("DEFECT_LOG_DIR", raising=False)
    from src.utils.paths import PROJECT_ROOT
    assert logmod.log_dir() == PROJECT_ROOT / "logs"


def test_files_disabled_under_pytest(monkeypatch):
    """跑 pytest 时默认不落盘 —— 否则单测会在仓库里堆日志垃圾。"""
    monkeypatch.delenv("DEFECT_LOG_DISABLE", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_a")
    assert logmod._should_write_files() is False


def test_files_disabled_by_env(monkeypatch):
    monkeypatch.setenv("DEFECT_LOG_DISABLE", "1")
    assert logmod._should_write_files() is False


def test_get_logger_writes_to_disk(log_home):
    """P0-6 核心：日志必须真的落盘，而不是只打到控制台。"""
    lg = logmod.get_logger("defect_test_disk")
    lg.info("落盘验证 %s", "ok")
    for h in lg.handlers:
        h.flush()

    f = log_home / logmod.LOG_FILENAME
    assert f.exists(), "日志文件未创建 —— 说明只有 StreamHandler"
    assert "落盘验证 ok" in f.read_text(encoding="utf-8")


def test_get_logger_is_idempotent(log_home):
    """重复取同一 logger 不能叠加 handler（否则日志行会成倍重复）。"""
    lg1 = logmod.get_logger("defect_test_idem")
    n1 = len(lg1.handlers)
    lg2 = logmod.get_logger("defect_test_idem")
    assert lg1 is lg2
    assert len(lg2.handlers) == n1


def test_logger_does_not_propagate(log_home):
    """不冒泡到 root，否则与 root handler 叠加造成重复输出。"""
    lg = logmod.get_logger("defect_test_prop")
    assert lg.propagate is False


def test_get_logger_survives_unwritable_dir(monkeypatch, tmp_path):
    """磁盘满/无权限时不能抛异常 —— 日志系统故障不该拖垮推理服务。"""
    # 用一个"父路径是文件"的目录，mkdir 必然失败
    bad = tmp_path / "afile"
    bad.write_text("x", encoding="utf-8")
    monkeypatch.setenv("DEFECT_LOG_DIR", str(bad / "sub"))
    monkeypatch.setattr(logmod, "_should_write_files", lambda: True)
    lg = logmod.get_logger("defect_test_baddir")
    assert lg is not None
    assert any(isinstance(h, logging.StreamHandler) for h in lg.handlers)


# ---------------------------------------------------------------- 判定追溯
def test_record_decision_writes_jsonl(log_home):
    logmod.record_decision(
        source="upload:steel.jpg",
        detections=[
            {"cls": 5, "name": "scratches", "conf": 0.91234},
            {"cls": 3, "name": "pitted_surface", "conf": 0.5},
        ],
        elapsed_ms=28.3456, conf=0.25, iou=0.5, model="best.onnx",
    )
    f = log_home / logmod.TRACE_FILENAME
    assert f.exists()
    rec = json.loads(f.read_text(encoding="utf-8").strip())

    assert rec["source"] == "upload:steel.jpg"
    assert rec["count"] == 2
    assert rec["model"] == "best.onnx"
    assert rec["conf_thr"] == 0.25
    assert rec["iou_thr"] == 0.5
    assert rec["elapsed_ms"] == 28.35          # 四舍五入到 2 位
    assert rec["detections"][0] == {"cls": 5, "name": "scratches", "conf": 0.9123}
    assert "ts_unix" in rec and "ts" in rec


def test_record_decision_appends_not_overwrites(log_home):
    """必须是追加（JSONL），滚动覆盖就没法回溯了。"""
    for i in range(3):
        logmod.record_decision(source=f"s{i}", detections=[])
    lines = (log_home / logmod.TRACE_FILENAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(x)["source"] for x in lines] == ["s0", "s1", "s2"]


def test_record_decision_omits_optional_fields(log_home):
    """未传的阈值不应写成 null 混进记录，保持 JSONL 干净可统计。"""
    logmod.record_decision(source="s", detections=[])
    rec = json.loads(
        (log_home / logmod.TRACE_FILENAME).read_text(encoding="utf-8").strip())
    assert "conf_thr" not in rec
    assert "iou_thr" not in rec
    assert "elapsed_ms" not in rec


def test_record_decision_never_raises(tmp_path, monkeypatch):
    """追溯是旁路能力：写失败绝不能影响检测主流程。"""
    bad = tmp_path / "afile"
    bad.write_text("x", encoding="utf-8")
    monkeypatch.setenv("DEFECT_LOG_DIR", str(bad / "sub"))
    monkeypatch.setattr(logmod, "_should_write_files", lambda: True)
    logmod.record_decision(source="s", detections=[{"cls": 0}])   # 不应抛


def test_record_decision_survives_malformed_detections(log_home):
    """上游给的字段缺失/类型异常时按默认值降级，而不是炸掉追溯。"""
    logmod.record_decision(source="s", detections=[{"cls": 2}, {"name": "x"}])
    rec = json.loads(
        (log_home / logmod.TRACE_FILENAME).read_text(encoding="utf-8").strip())
    assert rec["count"] == 2
    assert rec["detections"][0]["cls"] == 2
    assert rec["detections"][1]["name"] == "x"


def test_record_decision_serializes_concurrently(log_home):
    """流式推理线程与请求线程会并发写同一文件，行的完整性必须保住。"""
    import threading
    def worker(n):
        for i in range(20):
            logmod.record_decision(source=f"t{n}-{i}", detections=[])
    ts = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lines = (log_home / logmod.TRACE_FILENAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 80
    for ln in lines:                        # 每行都必须是合法 JSON（没交错撕裂）
        json.loads(ln)


# ---------------------------------------------------------------- 日志清理
def test_prune_removes_old_keeps_fresh(log_home):
    """>30 天的删掉，新的保留 —— 对应部署报告「保留期 >= 30 天」。"""
    old = log_home / logmod.LOG_FILENAME
    old.write_text("old", encoding="utf-8")
    stale = time.time() - 40 * 86400
    import os as _os
    _os.utime(old, (stale, stale))

    fresh = log_home / "decisions.jsonl.1"
    fresh.write_text("fresh", encoding="utf-8")

    assert logmod.prune_old_logs(keep_days=30) == 1
    assert not old.exists()
    assert fresh.exists()


def test_prune_ignores_unrelated_files(log_home):
    """logs/ 里用户自己放的东西不能被误删。"""
    other = log_home / "keep-me.txt"
    other.write_text("user data", encoding="utf-8")
    stale = time.time() - 99 * 86400
    import os as _os
    _os.utime(other, (stale, stale))

    assert logmod.prune_old_logs(keep_days=1) == 0
    assert other.exists()


def test_prune_missing_dir_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFECT_LOG_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(logmod, "_should_write_files", lambda: True)
    assert logmod.prune_old_logs() == 0


# ---------------------------------------------------------------- 异常自恢复
@pytest.fixture
def api_mod(monkeypatch):
    """取 api 模块并把自恢复状态复位成一个干净的起点。"""
    pytest.importorskip("fastapi")
    from src.inference import api
    monkeypatch.setattr(api, "_backend", None, raising=False)
    monkeypatch.setattr(api, "_backend_info", {}, raising=False)
    monkeypatch.setattr(api, "_reload_failures", 0, raising=False)
    monkeypatch.setattr(api, "_next_reload_at", 0.0, raising=False)
    monkeypatch.setattr(api, "_last_reload_attempt", 0.0, raising=False)
    return api


def _fake_backend():
    return types.SimpleNamespace(
        providers=["CPUExecutionProvider"], imgsz=640, conf=0.25, iou=0.5)


def test_init_backend_failure_does_not_raise(api_mod, monkeypatch):
    """P0-7 核心：加载失败必须让服务活着，而不是启动即崩。"""
    monkeypatch.setattr(api_mod, "_load_backend_once",
                        lambda: (None, {}, "未找到模型文件（best.onnx 或 best.pt）"))
    assert api_mod._init_backend() is False
    assert api_mod._backend is None
    assert api_mod._backend_info["error"]


def test_init_backend_survives_exception(api_mod, monkeypatch):
    """模型文件损坏抛异常时同样不能崩，异常要转成错误信息。"""
    def boom():
        raise RuntimeError("invalid protobuf")
    monkeypatch.setattr(api_mod, "_load_backend_once", boom)
    assert api_mod._init_backend() is False
    assert "invalid protobuf" in api_mod._backend_info["error"]


def test_failures_accumulate_and_backoff_grows(api_mod, monkeypatch):
    """失败次数递增，且下次重试时间按指数退避推后（防磁盘故障时疯狂重试）。"""
    monkeypatch.setattr(api_mod, "_load_backend_once", lambda: (None, {}, "missing"))
    monkeypatch.setattr(api_mod, "RELOAD_INITIAL_DELAY", 5.0)

    api_mod._init_backend()
    assert api_mod._reload_failures == 1
    d1 = api_mod._next_reload_at - time.time()

    api_mod._init_backend()
    assert api_mod._reload_failures == 2
    d2 = api_mod._next_reload_at - time.time()

    assert d2 > d1, "退避没有增长"


def test_backoff_is_capped(api_mod, monkeypatch):
    """退避必须有上限，否则几十次失败后会退避到几天之后。"""
    monkeypatch.setattr(api_mod, "_load_backend_once", lambda: (None, {}, "missing"))
    monkeypatch.setattr(api_mod, "RELOAD_MAX_DELAY", 1.0)
    for _ in range(8):
        api_mod._init_backend()
    assert api_mod._next_reload_at - time.time() <= 1.05


def test_recovery_resets_failure_counter(api_mod, monkeypatch):
    """文件恢复后：加载成功、计数归零、清除下次重试时间。"""
    monkeypatch.setattr(api_mod, "_load_backend_once", lambda: (None, {}, "missing"))
    api_mod._init_backend()
    api_mod._init_backend()
    assert api_mod._reload_failures == 2

    backend = _fake_backend()
    monkeypatch.setattr(api_mod, "_load_backend_once",
                        lambda: (backend, {"backend": "onnxruntime"}, ""))
    assert api_mod._init_backend() is True
    assert api_mod._backend is backend
    assert api_mod._reload_failures == 0
    assert api_mod._next_reload_at == 0.0


def test_ensure_backend_noop_when_healthy(api_mod, monkeypatch):
    """模型可用时请求路径不应触发任何加载尝试（零开销）。"""
    calls = []
    api_mod._backend = _fake_backend()
    monkeypatch.setattr(api_mod, "_init_backend", lambda: calls.append(1) or True)
    api_mod._ensure_backend()
    assert calls == []


def test_ensure_backend_is_rate_limited(api_mod, monkeypatch):
    """降级态下高频请求不能把重试变成 DoS —— 受 RELOAD_MIN_INTERVAL 限频。"""
    calls = []
    monkeypatch.setattr(api_mod, "RELOAD_MIN_INTERVAL", 60.0)
    monkeypatch.setattr(api_mod, "_init_backend", lambda: calls.append(1) or False)
    api_mod._last_reload_attempt = time.time()

    for _ in range(50):          # 模拟 50 个并发请求
        api_mod._ensure_backend()
    assert len(calls) <= 1, f"限频失效，触发了 {len(calls)} 次重试"


def test_probe_loop_recovers_without_manual_intervention(api_mod, monkeypatch, ):
    """P0-7 核心验收：后台探测线程在文件恢复后自动把服务拉回正常。

    这是整条自恢复链路的端到端验证（不依赖 TestClient，纯线程状态机）。
    """
    import threading

    monkeypatch.setattr(api_mod, "RELOAD_MAX_DELAY", 0.05)
    state = {"ok": False}
    monkeypatch.setattr(
        api_mod, "_load_backend_once",
        lambda: ((_fake_backend(), {"backend": "onnxruntime"}, "")
                 if state["ok"] else (None, {}, "missing")),
    )
    api_mod._init_backend()
    assert api_mod._backend is None

    stop = threading.Event()
    t = threading.Thread(target=api_mod._reload_probe_loop, args=(stop,), daemon=True)
    t.start()
    try:
        assert api_mod._backend is None, "模型还不该可用"
        state["ok"] = True                       # 相当于运维把文件放回去了
        # 注意：探测线程先设 _backend 再清 _reload_failures，两者不同步完成，
        # 所以不能只等 _backend —— 必须等状态**整体收敛**，否则会读到中间态。
        deadline = time.time() + 5.0
        while time.time() < deadline and (api_mod._backend is None
                                         or api_mod._reload_failures != 0):
            time.sleep(0.05)
        assert api_mod._backend is not None, "探测线程未自动恢复"
        assert api_mod._reload_failures == 0
    finally:
        stop.set()
        t.join(timeout=3.0)
    assert not t.is_alive(), "探测线程未随 stop 事件退出（会泄漏线程）"


def test_probe_loop_exits_promptly(api_mod):
    """stop 事件置位后线程要立即退出，否则服务关闭会挂住。"""
    import threading
    stop = threading.Event()
    t = threading.Thread(target=api_mod._reload_probe_loop, args=(stop,), daemon=True)
    t.start()
    stop.set()
    t.join(timeout=3.0)
    assert not t.is_alive()


# ---------------------------------------------------------------- 健康检查可观测性
def test_health_exposes_degraded_fields(monkeypatch):
    """/health 必须能直接被监控消费：degraded / reload_failures / next_reload_in_s。"""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.inference import api

    monkeypatch.setattr(api, "_backend", None, raising=False)
    monkeypatch.setattr(api, "_backend_info",
                        {"backend": None, "error": "missing"}, raising=False)
    monkeypatch.setattr(api, "_reload_failures", 3, raising=False)
    monkeypatch.setattr(api, "_next_reload_at", time.time() + 7, raising=False)
    # 不让 lifespan 真的去加载模型 / 起线程
    monkeypatch.setattr(api, "_init_backend", lambda: False)
    monkeypatch.setattr(api, "prune_old_logs", lambda *a, **k: 0)

    with TestClient(api.app) as c:
        b = c.get("/health").json()

    assert b["degraded"] is True
    assert b["status"] == "model_missing"
    assert b["reload_failures"] == 3
    assert b["next_reload_in_s"] > 0
    assert "error" in b


def test_health_ok_state_is_not_degraded(monkeypatch):
    """恢复后 degraded 必须回到 False，否则告警永远挂着（狼来了）。"""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.inference import api

    monkeypatch.setattr(api, "_backend", _fake_backend(), raising=False)
    monkeypatch.setattr(api, "_backend_info", {"backend": "onnxruntime"}, raising=False)
    monkeypatch.setattr(api, "_reload_failures", 0, raising=False)
    monkeypatch.setattr(api, "_next_reload_at", 0.0, raising=False)

    with TestClient(api.app) as c:
        b = c.get("/health").json()

    assert b["degraded"] is False
    assert b["status"] == "ok"
    assert b["reload_failures"] == 0


def test_degraded_detect_returns_503_with_reason(monkeypatch):
    """降级态下 /detect 应是 503（服务可用、模型不可用），并说明会自动重试。"""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.inference import api

    monkeypatch.setattr(api, "_backend", None, raising=False)
    monkeypatch.setattr(api, "_backend_info",
                        {"backend": None, "error": "未找到模型文件"}, raising=False)
    monkeypatch.setattr(api, "_init_backend", lambda: False)
    monkeypatch.setattr(api, "prune_old_logs", lambda *a, **k: 0)

    with TestClient(api.app) as c:
        r = c.post("/detect", files={"file": ("t.jpg", b"\xff\xd8\xff", "image/jpeg")})

    assert r.status_code == 503, f"期望 503，实际 {r.status_code}"
    assert "自动重试" in r.json()["detail"]


# ---------------------------------------------------------------- 接口鉴权
def _client(monkeypatch):
    """造一个 TestClient（lifespan 不真的加载模型、不清日志）。

    这里刻意**不返回 api 模块**：多数用例只要客户端即可，
    返回一个用不到的模块变量反而会被 lint 记为未使用。
    需要 api 模块的用例自行 `from src.inference import api`。
    """
    from fastapi.testclient import TestClient

    from src.inference import api
    monkeypatch.setattr(api, "_init_backend", lambda: False)
    monkeypatch.setattr(api, "prune_old_logs", lambda *a, **k: 0)
    return TestClient(api.app)


def test_endpoints_open_when_key_not_configured(monkeypatch):
    """未配密钥时保持向后兼容（本地开发/演示不该被卡住）。"""
    monkeypatch.delenv("DEFECT_API_KEY", raising=False)
    with _client(monkeypatch) as c:
        assert c.get("/stream/state").status_code == 200
        assert c.post("/stream/stop").status_code == 200


def test_health_is_reachable_without_key(monkeypatch):
    """/health 必须免鉴权：监控探活与降级诊断都依赖它。"""
    monkeypatch.setenv("DEFECT_API_KEY", "s3cret")
    with _client(monkeypatch) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["auth_enabled"] is True


def test_health_reports_auth_disabled(monkeypatch):
    """忘配密钥要能被监控发现（否则接口对内网完全敞开却无人知晓）。"""
    monkeypatch.delenv("DEFECT_API_KEY", raising=False)
    with _client(monkeypatch) as c:
        assert c.get("/health").json()["auth_enabled"] is False


def test_protected_endpoints_reject_missing_key(monkeypatch):
    monkeypatch.setenv("DEFECT_API_KEY", "s3cret")
    with _client(monkeypatch) as c:
        assert c.get("/stream/state").status_code == 401
        assert c.post("/stream/stop").status_code == 401
        r = c.post("/detect", files={"file": ("t.jpg", b"\xff\xd8\xff", "image/jpeg")})
        assert r.status_code == 401, r.text


def test_protected_endpoints_reject_wrong_key(monkeypatch):
    monkeypatch.setenv("DEFECT_API_KEY", "s3cret")
    with _client(monkeypatch) as c:
        r = c.get("/stream/state", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_protected_endpoints_accept_correct_key(monkeypatch):
    monkeypatch.setenv("DEFECT_API_KEY", "s3cret")
    with _client(monkeypatch) as c:
        assert c.get("/stream/state", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_bearer_token_is_accepted(monkeypatch):
    """兼容 Authorization: Bearer <key> —— 标准 OAuth 风格的调用方会这么发。"""
    monkeypatch.setenv("DEFECT_API_KEY", "s3cret")
    with _client(monkeypatch) as c:
        r = c.get("/stream/state", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_bearer_with_wrong_scheme_is_rejected(monkeypatch):
    monkeypatch.setenv("DEFECT_API_KEY", "s3cret")
    with _client(monkeypatch) as c:
        r = c.get("/stream/state", headers={"Authorization": "Basic s3cret"})
    assert r.status_code == 401


def test_api_key_is_read_per_request(monkeypatch):
    """密钥从环境变量每次请求读取 —— 改了不用重启服务。"""
    monkeypatch.setenv("DEFECT_API_KEY", "old")
    with _client(monkeypatch) as c:
        assert c.get("/stream/state", headers={"X-API-Key": "old"}).status_code == 200
        monkeypatch.setenv("DEFECT_API_KEY", "new")
        assert c.get("/stream/state", headers={"X-API-Key": "old"}).status_code == 401
        assert c.get("/stream/state", headers={"X-API-Key": "new"}).status_code == 200


# ---------------------------------------------------------------- 模块可导入性
def test_logger_import_has_no_side_effects(tmp_path, monkeypatch):
    """惰性初始化：import 本模块不能建目录/建文件。

    否则任何 import（单测、文档工具、打包）都会在仓库里留下垃圾。
    """
    monkeypatch.setenv("DEFECT_LOG_DIR", str(tmp_path / "lazytest"))
    monkeypatch.setattr(logmod, "_should_write_files", lambda: True)
    for m in [k for k in list(sys.modules) if k.startswith("src.utils.logger")]:
        del sys.modules[m]
    import importlib
    importlib.import_module("src.utils.logger")
    assert not (tmp_path / "lazytest").exists(), "import 期就建了日志目录"
