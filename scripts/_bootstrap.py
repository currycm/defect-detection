"""scripts/ 下所有入口脚本共用的运行环境引导（唯一实现）。

为什么需要它
------------
本机项目位于中文目录「机器视觉」下。中文路径经 Git Bash 的 argv/cwd 传递
可能被破坏，历史上 ultralytics 曾因路径解析失败**静默**退回默认数据集目录
（E:\\Mind\\...），排查代价很高。因此 CLI 入口必须自己确定项目根，
不能依赖调用方的 cwd 或相对路径。

以前这段逻辑（`PROJECT` 字面量 + `os.chdir` + `sys.path.insert`）在十几个
脚本里各复制一份，改一处要改十几处，且硬编码路径无法换机器。
现收敛到此模块。

解析顺序（每个候选目录都必须**实际含 configs/data.yaml** 才被采纳，
避免「解析到了错误的目录却不报错」这种最难查的情况）：
    1. 环境变量 DEFECT_PROJECT_ROOT     —— 迁移机器时只需设这个
    2. 由本文件位置推导（scripts/ 的上一级）

注意这里**不内置任何本机绝对路径**作为兜底：一旦写进去就会随代码公开
（泄露用户名/目录结构），换机后也只会把错误隐藏得更深。候选都不匹配时
直接抛 SystemExit 并提示设置 DEFECT_PROJECT_ROOT。

用法
----
    from _bootstrap import ensure_project_root

    PROJECT = ensure_project_root()          # 定位 + 入 sys.path + chdir
    from src.training.train import train     # noqa: E402 须在引导之后导入
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_VAR = "DEFECT_PROJECT_ROOT"
_MARKER = Path("configs") / "data.yaml"

_cached: Path | None = None


def _candidates():
    env = os.environ.get(ENV_VAR)
    if env:
        yield Path(env).expanduser()
    try:
        # scripts/_bootstrap.py -> scripts/ -> 项目根
        yield Path(__file__).resolve().parent.parent
    except OSError:  # pragma: no cover - 防御 __file__ 异常
        pass


def ensure_project_root(chdir: bool = True) -> Path:
    """定位项目根：加入 sys.path，默认同时把 cwd 切过去，返回项目根（Path）。

    chdir 的目的：ultralytics 把 `runs/`、权重等产物写到**当前工作目录**，
    切到项目根才能保证产物落在本项目内，而不是调用方所在目录。

    定位失败时直接抛 SystemExit（带排查提示），不做静默兜底 —— 静默兜底
    正是当初「训练跑到别的数据集目录」的成因。
    """
    global _cached
    if _cached is not None:
        return _cached

    for cand in _candidates():
        try:
            if (cand / _MARKER).exists():
                root = cand.resolve()
                break
        except OSError:
            continue
    else:
        tried = "\n".join(f"  - {c}" for c in _candidates())
        raise SystemExit(
            f"无法定位项目根：以下候选目录均不含 {_MARKER}\n{tried}\n"
            f"请设置环境变量 {ENV_VAR} 指向项目根目录后重试。"
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if chdir:
        os.chdir(root)
    _cached = root
    return root


def project_root() -> Path:
    """只解析项目根、**不**改变 cwd 的变体（供库代码/测试用）。"""
    return ensure_project_root(chdir=False)
