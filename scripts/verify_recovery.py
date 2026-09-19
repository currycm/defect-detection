"""异常自恢复的故障演练（P0 验收）。

做的事：真实走一遍「模型缺失 -> 服务降级 -> 文件恢复 -> 自动恢复」，
验证服务在整条路径上都不崩、不停机、不需要人工重启。

**关键设计：必须在隔离目录里演练。**
第一版直接把 weights/best.onnx 移走，结果服务回退到了 weights/best.pt
（项目有 PyTorch 兜底路径），根本没进入降级态 —— 演练本身失效了。
所以这里用一个临时项目根，里面只有 configs/ 与一个空 weights/，
通过 DEFECT_PROJECT_ROOT 指过去，从而确保**没有任何模型可用**。

用 TestClient 而不是真实 uvicorn：只验证状态机与重试逻辑，
不涉及无限流式响应（那个必须用真实 socket，见 scripts/test_stream_api.py）。

用法：
    D:/Anaconda/python.exe scripts/verify_recovery.py
"""
import os
import sys

# 真实项目根必须在 `import src.*` 之前进 sys.path。
# 注意：**不能用 _bootstrap.ensure_project_root()** —— 它会读 DEFECT_PROJECT_ROOT
# 并 chdir 到隔离根，那样这里就 import 不到 src 了（演练测试的是 src，不是隔离根）。
_REAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REAL_ROOT not in sys.path:
    sys.path.insert(0, _REAL_ROOT)

import shutil
import tempfile
import time
from pathlib import Path

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def build_isolated_root(src_root: Path, tmp: Path) -> Path:
    """造一个只有 configs/、没有任何模型文件的"假项目根"。"""
    root = tmp / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "weights").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    # data.yaml 是 paths._pick_root 的判定标记，必须存在
    shutil.copy2(src_root / "configs" / "data.yaml", root / "configs" / "data.yaml")
    return root


def main() -> int:
    src_root = Path(__file__).resolve().parent.parent
    real_onnx = src_root / "weights" / "best.onnx"
    if not real_onnx.exists():
        print(f"需要 {real_onnx} 才能演练（稍后要把它复制进来），请先 python scripts/export.py")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="defect_recovery_"))
    fake_root = build_isolated_root(src_root, tmp)
    os.environ["DEFECT_PROJECT_ROOT"] = str(fake_root)
    # 日志别写进真实项目
    os.environ["DEFECT_LOG_DIR"] = str(fake_root / "logs")

    print(f"隔离项目根: {fake_root}")
    print("（其中 weights/ 为空，因此 ONNX 与 .pt 兜底都不可用）\n")

    try:
        from fastapi.testclient import TestClient

        from src.inference import api

        # 加快重试节奏（默认退避是给生产用的，演练没必要等）
        api.RELOAD_INITIAL_DELAY = 0.3
        api.RELOAD_MAX_DELAY = 1.0
        api.RELOAD_MIN_INTERVAL = 0.2

        print("=== 1) 模型完全缺失时，服务仍能启动 ===")
        with TestClient(api.app) as client:
            r = client.get("/health")
            check("服务可响应 /health（未崩溃）", r.status_code == 200, f"HTTP {r.status_code}")
            body = r.json()
            check("status = model_missing", body.get("status") == "model_missing",
                  f"status={body.get('status')}")
            check("degraded = True（明确标记检测能力不可用）",
                  body.get("degraded") is True)
            check("reload_failures > 0（失败被记录）",
                  body.get("reload_failures", 0) > 0,
                  f"failures={body.get('reload_failures')}")
            check("error 字段说明原因", bool(body.get("error")), str(body.get("error")))
            check("next_reload_in_s 暴露了下次重试时间",
                  "next_reload_in_s" in body, f"{body.get('next_reload_in_s')}s")

            print("\n=== 2) 降级态下 /detect 返回 503（而非 500/崩溃）===")
            # 用真实图片，确保不走"解码失败"的 400 分支
            img = src_root / "data" / "processed" / "images" / "test"
            img = sorted(img.glob("*.jpg"))[0] if img.is_dir() else None
            if img and img.exists():
                with open(img, "rb") as f:
                    r = client.post(
                        "/detect",
                        files={"file": (img.name, f, "image/jpeg")},
                    )
                check("返回 503（服务可用、模型不可用）", r.status_code == 503,
                      f"HTTP {r.status_code}")
                detail = str(r.json().get("detail", ""))
                check("detail 提到会自动重试", "自动重试" in detail, detail[:70])
            else:
                print("  [跳过] 找不到 test 图片")

            print("\n=== 3) 放入模型文件，观察是否自动恢复 ===")
            shutil.copy2(real_onnx, fake_root / "weights" / "best.onnx")
            print("  已放入 best.onnx，等待后台探测线程接管（最多 10s）…")

            recovered = False
            t0 = time.time()
            for _ in range(50):
                time.sleep(0.2)
                b = client.get("/health").json()
                if b.get("status") == "ok":
                    recovered = True
                    print(f"  第 {time.time() - t0:.1f}s 自动恢复，无需人工干预")
                    break
            check("无需人工干预即自动恢复", recovered)
            if recovered:
                b = client.get("/health").json()
                check("reload_failures 归零", b.get("reload_failures") == 0,
                      f"failures={b.get('reload_failures')}")
                check("degraded 回到 False", b.get("degraded") is False)
                check("backend = onnxruntime", b.get("backend") == "onnxruntime",
                      f"backend={b.get('backend')}")

                print("\n=== 4) 恢复后 /detect 正常出结果 ===")
                if img and img.exists():
                    with open(img, "rb") as f:
                        r = client.post("/detect",
                                        files={"file": (img.name, f, "image/jpeg")})
                    check("/detect 返回 200", r.status_code == 200, f"HTTP {r.status_code}")
                    if r.status_code == 200:
                        check("检出结果非空（模型真的在工作）",
                              r.json().get("count", 0) > 0,
                              f"count={r.json().get('count')}")

            print("\n=== 5) 已加载模型被删除后的行为（边界）===")
            (fake_root / "weights" / "best.onnx").unlink()
            time.sleep(1.0)
            b = client.get("/health").json()
            # 设计决策：内存里已加载的模型仍然可用，**刻意不主动降级** ——
            # 因为模型文件被删/被换时，继续用已加载的模型服务，比立刻中断
            # 产线检测更安全（检测能力不该因为磁盘问题凭空消失）。
            # 这里验证的是"服务没崩"，而不是断言某个具体状态值。
            check("删除模型文件后服务仍存活",
                  b.get("status") in ("ok", "model_missing"), f"status={b.get('status')}")
            check("已加载的模型继续可用（不主动降级，符合设计）",
                  b.get("status") == "ok",
                  "继续用内存中的模型" if b.get("status") == "ok"
                  else f"已降级为 {b.get('status')}")
            # 仍能正常推理
            if img and img.exists():
                with open(img, "rb") as f:
                    r = client.post("/detect",
                                    files={"file": (img.name, f, "image/jpeg")})
                check("删除文件后仍能正常检测（服务未受影响）",
                      r.status_code == 200 and r.json().get("count", 0) > 0,
                      f"HTTP {r.status_code}, count={r.json().get('count') if r.status_code == 200 else '-'}")

    finally:
        os.environ.pop("DEFECT_PROJECT_ROOT", None)
        os.environ.pop("DEFECT_LOG_DIR", None)
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 46}")
    print(f"通过 {len(PASS)} / 失败 {len(FAIL)}")
    if FAIL:
        print("失败项：" + "、".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
