"""项目路径解析（全项目唯一真源）。

优先级：环境变量 `DEFECT_PROJECT_ROOT` > 由本文件位置推导。

为什么要做「多候选 + 校验」而不是直接用 `__file__`：
本机项目位于中文目录「机器视觉」下，中文路径经 shell（Git Bash）的 argv/cwd 传递
可能被破坏；历史上还踩过 ultralytics 因路径解析失败退回默认数据集目录（E:\\Mind\\...）
的坑。因此每个候选目录都必须**实际含有 configs/data.yaml** 才会被采纳，
全部不匹配时**直接报错**，保证「解析失败」不会静默发生。

这里刻意不内置本机绝对路径兜底：写进去就会随代码一起公开（用户名/目录结构泄露），
换机后也只是把错误藏得更深。迁移到别的机器时设 `DEFECT_PROJECT_ROOT` 即可。
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DEFECT_PROJECT_ROOT"

# 用于判定候选目录是否是本项目根
_MARKER = Path("configs") / "data.yaml"


def _candidates():
    env = os.environ.get(ENV_VAR)
    if env:
        yield Path(env).expanduser()
    # 不用 contextlib.suppress：这里是生成器，yield 位于 try 块内，
    # 换成 with 会改变生成器的求值时机语义，收益不足以承担该风险。
    # （ruff.toml 对本文件放行 SIM105，理由同此。）
    try:
        # src/utils/paths.py -> src/utils -> src -> 项目根
        yield Path(__file__).resolve().parents[2]
    except Exception:  # pragma: no cover - 仅防御 __file__ 异常
        pass


def _pick_root() -> Path:
    cands = list(_candidates())
    for cand in cands:
        try:
            if (cand / _MARKER).exists():
                return cand.resolve()
        except OSError:
            continue
    # 所有候选都不带标记文件：**不要**静默返回一个「看起来像」的目录 ——
    # 这正是当初「解析到错目录却不报错」的成因。直接抛错，并给出排查方式。
    tried = "\n".join(f"  - {c}" for c in cands)
    raise FileNotFoundError(
        f"未能定位项目根：以下候选目录均不含 {_MARKER}\n{tried}\n"
        f"请设置环境变量 {ENV_VAR} 指向项目根目录。"
    )


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


def runtime_data_yaml(data_yaml: str | os.PathLike | None = None) -> Path:
    """返回可直接交给 ultralytics 的 data.yaml：其中 `path` 一定是绝对路径。

    为什么需要这一步（不是多余的一层）：ultralytics 解析数据集根的代码是

        path = Path(data.get("path") or ...)
        if not path.exists() and not path.is_absolute():   # ← exists() 按 cwd 判定
            path = (DATASETS_DIR / path).resolve()

    也就是说 `path` 写相对路径时，解析结果**取决于调用时的 cwd**；cwd 一旦不是
    项目根，`Path("data/processed").exists()` 为假，就会静默改去全局 DATASETS_DIR
    （本项目历史上正是这样把训练数据读到了 E:\\Mind\\Mind+\\datasets）。

    所以仓库里 `configs/data.yaml` 保持可移植（相对路径、不含本机信息），
    真正喂给 ultralytics 的这份在这里把 `path` 绝对化，消除对 cwd 的隐式依赖。

    生成的副本为 `configs/data.local.yaml`（不入库，内容不变则不重写）。
    """
    src = Path(data_yaml) if data_yaml is not None else DATA_YAML
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(f"数据集配置不存在: {src}")

    import yaml  # 惰性导入：paths 被大量模块导入，不进 import 期依赖

    with open(src, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"数据集配置格式异常（应为映射）: {src}")

    raw = str(cfg.get("path") or "").strip()
    p = Path(raw).expanduser() if raw else PROCESSED_DIR
    cfg["path"] = (p if p.is_absolute() else PROJECT_ROOT / p).resolve().as_posix()

    out = CONFIGS_DIR / "data.local.yaml"
    text = (
        "# 自动生成（paths.runtime_data_yaml），请勿手改，也不入库。\n"
        "# 作用：把 path 绝对化，消除 ultralytics 按 cwd 解析相对路径的不确定性。\n"
        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    )
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.write_text(text, encoding="utf-8")
    return out


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
