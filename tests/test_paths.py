"""路径解析与安全边界的回归测试（src/utils/paths.py）。

这些断言不是形式主义的：`is_path_allowed` 是 `/stream?source=` 防任意文件
读取的唯一防线（H1），一旦回归就会重新暴露本机文件。
"""
from pathlib import Path

import pytest

from src.utils import paths


def test_project_root_is_real_project():
    """解析出的项目根必须真的含 configs/data.yaml（标记校验）。"""
    assert (paths.PROJECT_ROOT / "configs" / "data.yaml").is_file()


def test_key_paths_live_under_project_root():
    for p in (paths.DATA_YAML, paths.HYP_YAML, paths.ONNX_PATH,
              paths.BEST_PT, paths.PROCESSED_DIR, paths.RUNS_DIR):
        assert Path(p).is_absolute()
        assert Path(p).is_relative_to(paths.PROJECT_ROOT)


def test_allowed_roots_defaults_to_project():
    assert paths.PROJECT_ROOT in paths.ALLOWED_SOURCE_ROOTS


def test_path_allowed_inside_project():
    assert paths.is_path_allowed(paths.PROCESSED_DIR / "images" / "test")


def test_path_denied_for_outside_dir(tmp_path):
    """项目外的目录（临时目录）必须被拒绝。"""
    assert paths.is_path_allowed(tmp_path) is False


def test_path_denied_for_parent_dir():
    assert paths.is_path_allowed(paths.PROJECT_ROOT.parent) is False


def test_path_traversal_is_resolved_and_denied():
    """`项目根/../../Windows/win.ini` 这种写法要先 resolve 再比较，不能被绕过。"""
    evil = paths.PROJECT_ROOT / ".." / ".." / "Windows" / "win.ini"
    assert paths.is_path_allowed(evil) is False


def test_nonexistent_path_inside_root_is_still_allowed():
    """允许性判断只看路径位置，不要求文件存在（存在性由后续打开步骤报错）。"""
    assert paths.is_path_allowed(paths.PROJECT_ROOT / "not_created_yet.bin") is True


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_path_not_allowed(bad):
    assert paths.is_path_allowed(bad) is False
