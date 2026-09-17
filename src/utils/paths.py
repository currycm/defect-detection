"""项目路径解析（全项目唯一真源）。

优先级：环境变量 `DEFECT_PROJECT_ROOT` > 由本文件位置推导 > 本机已知绝对路径。

为什么要做「多候选 + 校验」而不是直接用 `__file__`：
本机项目位于中文目录「机器视觉」下，中文路径经 shell（Git Bash）的 argv/cwd 传递
可能被破坏；历史上还踩过 ultralytics 因路径解析失败退回默认数据集目录（E:\\Mind\\...）
的坑。因此每个候选目录都必须**实际含有 configs/data.yaml** 才会被采纳，
否则继续下一个候选，最后兜底到已知绝对路径，保证「解析失败」不会静默发生。

迁移到别的机器时：设 `DEFECT_PROJECT_ROOT` 即可，无需改代码。
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DEFECT_PROJECT_ROOT"

# 用于判定候选目录是否是本项目根
_MARKER = Path("configs") / "data.yaml"

# 本机已知路径（最后兜底；换机时用环境变量覆盖即可）
_KNOWN_ROOT = Path(r"C:\Users\24830\Desktop\机器视觉\defect-detection")


def _candidates():
    env = os.environ.get(ENV_VAR)
    if env:
        yield Path(env).expanduser()
    try:
        # src/utils/paths.py -> src/utils -> src -> 项目根
        yield Path(__file__).resolve().parents[2]
    except Exception:  # pragma: no cover - 仅防御 __file__ 异常
        pass
    yield _KNOWN_ROOT


def _pick_root() -> Path:
    for cand in _candidates():
        try:
            if (cand / _MARKER).exists():
                return cand.resolve()
        except OSError:
            continue
    # 全部候选都不带标记文件：退回已知路径，并通过环境变量提示排查
    return _KNOWN_ROOT


PROJECT_ROOT: Path = _pick_root()

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
RUNS_DIR: Path = PROJECT_ROOT / "runs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

DATA_YAML: Path = CONFIGS_DIR / "data.yaml"
HYP_YAML: Path = CONFIGS_DIR / "hyp.yaml"
ONNX_PATH: Path = WEIGHTS_DIR / "best.onnx"
PRETRAINED_PT: Path = PROJECT_ROOT / "yolov8n.pt"

# 训练产物权重（Step 5 消融最优）
BEST_PT: Path = RUNS_DIR / "detect" / "runs" / "exp_aug640" / "weights" / "best.pt"

# 允许通过 /stream?source= 访问的本地路径根目录（安全边界，见 api._resolve_source）
ALLOWED_SOURCE_ROOTS: tuple[Path, ...] = (PROJECT_ROOT,)
_extra_roots = os.environ.get("DEFECT_ALLOWED_SOURCE_ROOTS", "").strip()
if _extra_roots:
    ALLOWED_SOURCE_ROOTS += tuple(
        Path(p).expanduser() for p in _extra_roots.split(os.pathsep) if p.strip()
    )


def is_path_allowed(path: str | os.PathLike) -> bool:
    """判断某个本地路径是否落在允许的根目录内（用于挡住任意文件读取）。

    注意：空串 / 纯空白必须**直接拒绝**。`Path("")` 会被 resolve 成「当前工作
    目录」，而 cwd 通常正是项目根，于是空路径会被误判成「允许」——等于把整个
    cwd 放行（此漏洞由 tests/test_paths.py 的用例实测发现）。
    """
    if path is None:
        return False
    try:
        if not str(path).strip():
            return False
        target = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return False
    for root in ALLOWED_SOURCE_ROOTS:
        try:
            if target == root.resolve() or target.is_relative_to(root.resolve()):
                return True
        except OSError:
            continue
    return False
