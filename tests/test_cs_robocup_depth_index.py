"""Unit tests for riskam.data.cs_robocup_2023.CSRobocup2023DepthIndex."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def patched_raw_dir(tmp_path):
    """Point CS_ROBOCUP_2023_ML_RAW_DIR at a tmp directory for the test."""
    with mock.patch(
        "riskam.data.cs_robocup_2023.CS_ROBOCUP_2023_ML_RAW_DIR", tmp_path
    ):
        yield tmp_path


def _write_depth_npy(run_dir: Path, timestamp: float, fill_mm: int) -> None:
    depth_dir = run_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    frame = np.full((4, 4), fill_mm, dtype=np.uint16)
    np.save(depth_dir / f"{timestamp:.3f}.npy", frame)


def test_empty_when_depth_dir_missing(patched_raw_dir):
    from riskam.data.cs_robocup_2023 import CSRobocup2023DepthIndex

    idx = CSRobocup2023DepthIndex("RB_01")
    assert not idx
    assert len(idx) == 0
    assert idx.load_for_rgb(Path("1000.000.png")) is None


def test_nearest_neighbour_picks_closer_timestamp(patched_raw_dir):
    from riskam.data.cs_robocup_2023 import CSRobocup2023DepthIndex

    run_dir = patched_raw_dir / "RB_01"
    _write_depth_npy(run_dir, 1000.000, fill_mm=1000)  # 1.0 m
    _write_depth_npy(run_dir, 1000.100, fill_mm=2000)  # 2.0 m
    _write_depth_npy(run_dir, 1000.300, fill_mm=3000)  # 3.0 m

    idx = CSRobocup2023DepthIndex("RB_01")
    assert len(idx) == 3

    # RGB timestamp 1000.040 is closer to 1000.000 than to 1000.100 → 1.0 m
    depth = idx.load_for_rgb(Path("1000.040.png"))
    assert depth is not None
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(1.0)

    # RGB timestamp 1000.080 is closer to 1000.100 → 2.0 m
    depth = idx.load_for_rgb(Path("1000.080.png"))
    assert depth[0, 0] == pytest.approx(2.0)


def test_returns_none_when_closest_frame_too_far(patched_raw_dir):
    from riskam.data.cs_robocup_2023 import (
        CSRobocup2023DepthIndex,
        RGB_DEPTH_MAX_DT_S,
    )

    run_dir = patched_raw_dir / "RB_01"
    _write_depth_npy(run_dir, 1000.000, fill_mm=1000)

    idx = CSRobocup2023DepthIndex("RB_01")
    far_ts = 1000.000 + RGB_DEPTH_MAX_DT_S + 0.1
    assert idx.load_for_rgb(Path(f"{far_ts:.3f}.png")) is None


def test_returns_none_for_non_timestamp_rgb_stem(patched_raw_dir):
    from riskam.data.cs_robocup_2023 import CSRobocup2023DepthIndex

    run_dir = patched_raw_dir / "RB_01"
    _write_depth_npy(run_dir, 1000.000, fill_mm=1000)

    idx = CSRobocup2023DepthIndex("RB_01")
    assert idx.load_for_rgb(Path("not_a_timestamp.png")) is None


def test_boundary_rgb_timestamps_use_endpoints(patched_raw_dir):
    from riskam.data.cs_robocup_2023 import CSRobocup2023DepthIndex

    run_dir = patched_raw_dir / "RB_01"
    _write_depth_npy(run_dir, 1000.000, fill_mm=1000)
    _write_depth_npy(run_dir, 1000.100, fill_mm=2000)

    idx = CSRobocup2023DepthIndex("RB_01")

    # RGB before the earliest depth frame, within tolerance → 1.0 m
    depth = idx.load_for_rgb(Path("999.950.png"))
    assert depth[0, 0] == pytest.approx(1.0)

    # RGB after the latest depth frame, within tolerance → 2.0 m
    depth = idx.load_for_rgb(Path("1000.150.png"))
    assert depth[0, 0] == pytest.approx(2.0)


def test_mm_to_m_conversion(patched_raw_dir):
    from riskam.data.cs_robocup_2023 import CSRobocup2023DepthIndex

    run_dir = patched_raw_dir / "RB_01"
    _write_depth_npy(run_dir, 1000.000, fill_mm=1500)

    idx = CSRobocup2023DepthIndex("RB_01")
    depth = idx.load_for_rgb(Path("1000.000.png"))
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(1.5)
