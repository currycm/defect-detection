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


def test_data_yaml_is_portable():
    """入库的 configs/data.yaml 不能带本机绝对路径（否则泄露用户名且换机不可用）。"""
    text = paths.DATA_YAML.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("path:"):
            assert ":" not in stripped.split(":", 1)[1], f"path 疑似绝对路径: {stripped}"
            break
    else:
        pytest.fail("data.yaml 缺少 path 字段")


def test_runtime_data_yaml_absolutizes_path():
    """交给 ultralytics 的那份必须有绝对 path（cwd 变化时才会解析正确）。"""
    import yaml

    out = paths.runtime_data_yaml()
    assert out.is_file()
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert Path(cfg["path"]).is_absolute()
    assert Path(cfg["path"]) == paths.PROCESSED_DIR.resolve()
    assert cfg["nc"] == 6 and len(cfg["names"]) == 6


def test_runtime_data_yaml_stable_across_cwd(tmp_path, monkeypatch):
    """核心保证：cwd 换到别处，生成的 path 与 train/val/test 仍然指向项目内。"""
    import yaml

    monkeypatch.chdir(tmp_path)
    out = paths.runtime_data_yaml()
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert Path(cfg["path"]) == paths.PROCESSED_DIR.resolve()
    for split in ("train", "val", "test"):
        # 无论 cwd 在哪，三路都解析到项目内（数据未生成时也能断言）
        assert (Path(cfg["path"]) / cfg[split]).is_relative_to(paths.PROCESSED_DIR)
        if paths.PROCESSED_DIR.is_dir():
            assert (Path(cfg["path"]) / cfg[split]).is_dir(), split


def test_runtime_data_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        paths.runtime_data_yaml(tmp_path / "nope.yaml")
