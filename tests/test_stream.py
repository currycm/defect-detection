"""流式检测核心逻辑测试（src/inference/stream.py）。

全部使用「假 predictor」（不加载真模型），因此测试很快且可离线运行。
覆盖三个曾经出过问题、且属于需求硬指标的机制：
  A. 限频：detect_interval 内的帧复用上次结果，不重复推理
  B. 去抖：连续 N 帧相同才确认，且只在状态**变化**时回调
  C. 边界：负值参数会被钳制（旧实现 detect_interval=-1 时推理占比 88.9%）
  D. 安全：合成帧源的路径必须落在 allowed_roots 内
"""
import numpy as np
import pytest

from src.inference.stream import (
    FPS_RANGE,
    HOLD_RANGE,
    INTERVAL_RANGE,
    ResultStabilizer,
    StreamDetector,
    open_source,
)
from src.utils import paths

FRAME = np.zeros((32, 32, 3), dtype=np.uint8)
DET = {"xyxy": [2, 2, 20, 20], "conf": 0.9, "cls": 0}


class FakePredictor:
    """记录调用次数的假推理器。"""

    def __init__(self, dets=None):
        self.dets = dets if dets is not None else [DET]
        self.calls = 0

    def predict_image(self, frame, conf=0.25, iou=0.5):
        self.calls += 1
        return [dict(d) for d in self.dets]


# ---------------------------------------------------------------- A. 限频
def test_process_frame_reuses_result_within_interval():
    pred = FakePredictor()
    det = StreamDetector(pred, source=0, detect_interval=60.0)
    for _ in range(5):
        det.process_frame(FRAME)
    assert pred.calls == 1, "间隔内的帧不应重复推理"


def test_frames_inferred_counter_tracks_calls():
    pred = FakePredictor()
    det = StreamDetector(pred, source=0, detect_interval=60.0)
    det.process_frame(FRAME)
    det.process_frame(FRAME)
    assert det._frames_inferred == 1


# ---------------------------------------------------------------- C. 边界钳制
def test_params_are_clamped_to_valid_ranges():
    det = StreamDetector(FakePredictor(), source=0,
                         target_fps=-5, detect_interval=-1,
                         hold=-10, conf=5.0, iou=-1.0)
    assert det.target_fps == FPS_RANGE[0]
    assert det.detect_interval == INTERVAL_RANGE[0]
    assert det.hold == HOLD_RANGE[0]
    assert det.conf == 1.0 and det.iou == 0.0


def test_stable_frames_clamped_to_at_least_one():
    det = StreamDetector(FakePredictor(), source=0, stable_frames=0)
    assert det.stable_frames == 1


# ---------------------------------------------------------------- B. 去抖
def test_stabilizer_emits_only_on_state_change():
    st = ResultStabilizer(stable_frames=2, pos_tolerance=0.1)
    d = [{"xyxy": [0, 0, 10, 10], "cls": 0}]

    assert st.update(d, 100, 100) is False   # 第 1 帧：未达确认帧数
    assert st.update(d, 100, 100) is True    # 第 2 帧：确认并发射
    assert st.update(d, 100, 100) is False   # 状态未变：不重复发射
    assert st.update([], 100, 100) is False  # 变为空：重新计数
    assert st.update([], 100, 100) is True   # 确认变化


def test_stabilizer_ignores_subpixel_jitter():
    """几十像素内的抖动不应被当成新状态（位置被网格量化）。"""
    st = ResultStabilizer(stable_frames=1, pos_tolerance=0.1)
    a = [{"xyxy": [0, 0, 10, 10], "cls": 0}]
    b = [{"xyxy": [1, 1, 11, 11], "cls": 0}]   # 几乎同一位置
    assert st.update(a, 100, 100) is True
    assert st.update(b, 100, 100) is False


def test_on_change_callback_fires_once_per_state():
    seen = []
    det = StreamDetector(FakePredictor(), source=0,
                         detect_interval=0.0, stable_frames=1,
                         on_change=lambda dets, frame: seen.append(len(dets)))
    for _ in range(4):
        det.process_frame(FRAME)
    assert seen == [1], f"同一稳定状态应只回调一次，实际 {seen}"


# ---------------------------------------------------------------- D. 安全边界
def test_open_source_rejects_path_outside_allowed_roots(tmp_path):
    with pytest.raises(PermissionError):
        open_source(f"synthetic:{tmp_path}", allowed_roots=(paths.PROJECT_ROOT,))


def test_open_source_rejects_path_traversal(tmp_path):
    sneaky = paths.PROJECT_ROOT / ".." / ".." / "Windows"
    with pytest.raises(PermissionError):
        open_source(f"synthetic:{sneaky}", allowed_roots=(paths.PROJECT_ROOT,))


def test_open_source_accepts_path_inside_root():
    target = paths.PROCESSED_DIR / "images" / "test"
    if not target.is_dir() or not any(target.iterdir()):
        pytest.skip("需要 data/processed/images/test 下有图片")
    src = open_source(f"synthetic:{target}", allowed_roots=(paths.PROJECT_ROOT,))
    ok, frame = src.read()
    assert ok and frame is not None


def test_synthetic_source_requires_images(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(f"synthetic:{tmp_path}")
