"""
data.cs_robocup_indices

Year-agnostic nearest-neighbour indices for the CoreSense RoboCup datasets.

Both the 2023 and 2024 recordings share the same on-disk layout
(``raw_dataset/<RUN>/{rgb/, depth/, camera_info.json, odom.csv}``); only the
raw-dataset root and the depth file format differ (2023: ``.npy``, 2024:
16-bit ``.png`` — both raw uint16 millimetres). The year-specific modules
(:mod:`riskam.data.cs_robocup_2023`, :mod:`riskam.data.cs_robocup_2024`)
bind these classes to their raw-dataset root.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from riskam.ml.depth import depth_mm_to_m


# Max allowable RGB↔depth timestamp mismatch (seconds). Colour and depth
# streams are typically synced within tens of milliseconds; anything beyond
# this threshold is treated as missing depth data.
RGB_DEPTH_MAX_DT_S = 0.2

# Max allowable RGB↔odometry timestamp mismatch (seconds). Odometry is dense
# (~40–100 Hz in the RoboCup bags), so a small tolerance suffices.
RGB_ODOM_MAX_DT_S = 0.25


def _load_depth_mm(path: Path) -> np.ndarray:
    """Load a raw uint16-millimetre depth frame (``.npy`` or 16-bit PNG)."""
    if path.suffix == ".npy":
        return np.load(path)
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # pylint: disable=no-member


class CSRobocupDepthIndex:
    """Nearest-neighbour depth lookup for CS RoboCup frames.

    Depth frames are stored as ``{timestamp:.3f}.npy`` or ``.png`` under
    ``<RUN>/depth/`` (raw uint16 millimetres). This index sorts available
    timestamps once, then resolves the closest depth frame for any RGB frame
    via binary search. If the closest depth frame is more than
    ``RGB_DEPTH_MAX_DT_S`` seconds away, depth is treated as unavailable.
    """

    def __init__(self, run: str, raw_dir: Path) -> None:
        depth_dir = raw_dir / run / "depth"
        if not depth_dir.is_dir():
            self._timestamps = np.empty(0, dtype=float)
            self._paths: list[Path] = []
            return
        paths = sorted(
            list(depth_dir.glob("*.npy")) + list(depth_dir.glob("*.png"))
        )
        self._paths = paths
        self._timestamps = np.array([float(p.stem) for p in paths], dtype=float)

    def __bool__(self) -> bool:
        return self._timestamps.size > 0

    def __len__(self) -> int:
        return self._timestamps.size

    def _nearest(self, rgb_path: Path) -> int | None:
        """Index of the depth frame nearest to ``rgb_path``, or None."""
        if self._timestamps.size == 0:
            return None
        try:
            target_ts = float(rgb_path.stem)
        except ValueError:
            return None
        idx = int(np.searchsorted(self._timestamps, target_ts))
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < self._timestamps.size:
            candidates.append(idx)
        best = min(candidates, key=lambda i: abs(self._timestamps[i] - target_ts))
        if abs(self._timestamps[best] - target_ts) > RGB_DEPTH_MAX_DT_S:
            return None
        return best

    def has_for_rgb(self, rgb_path: Path) -> bool:
        """Whether a depth frame exists within tolerance (no file I/O)."""
        return self._nearest(rgb_path) is not None

    def load_for_rgb(self, rgb_path: Path) -> np.ndarray | None:
        """Return the depth frame (float32 metres) nearest to ``rgb_path``.

        Returns ``None`` if no depth frames are indexed, the RGB stem is not
        a valid timestamp, or the closest depth frame is farther than
        ``RGB_DEPTH_MAX_DT_S`` away.
        """
        best = self._nearest(rgb_path)
        if best is None:
            return None
        depth_mm = _load_depth_mm(self._paths[best])
        return depth_mm_to_m(depth_mm)


class CSRobocupOdomIndex:
    """Nearest-neighbour odometry-twist lookup for CS RoboCup frames.

    Reads ``<RUN>/odom.csv`` (columns ``t_s, linear_x, linear_y, angular_z``,
    written by the aux bag-extraction pass in
    :mod:`riskam.data.extract_cs_robocup`). The file is optional: when it is
    absent the index is falsy and the kinematic pipeline degrades to the
    no-ego (v_robot = 0) behaviour. Twists are in the robot base frame; the
    camera is assumed aligned with base forward (the TIAGo head-pan angle is
    ignored — a documented approximation).
    """

    def __init__(
        self, run: str, raw_dir: Path, tolerance_s: float = RGB_ODOM_MAX_DT_S
    ) -> None:
        self._tolerance_s = tolerance_s
        odom_path = raw_dir / run / "odom.csv"
        if not odom_path.is_file():
            self._data = np.empty((0, 4), dtype=float)
            return
        self._data = np.loadtxt(
            odom_path, delimiter=",", skiprows=1, dtype=float
        ).reshape(-1, 4)

    def __bool__(self) -> bool:
        return self._data.shape[0] > 0

    def __len__(self) -> int:
        return self._data.shape[0]

    def twist_for(self, t_s: float):
        """Return the nearest :class:`riskam.kinematics.PlanarTwist`, or
        ``None`` when no sample lies within the tolerance."""
        from riskam.kinematics import PlanarTwist

        if self._data.shape[0] == 0:
            return None
        ts = self._data[:, 0]
        idx = int(np.searchsorted(ts, t_s))
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(ts)]
        best = min(candidates, key=lambda i: abs(ts[i] - t_s))
        if abs(ts[best] - t_s) > self._tolerance_s:
            return None
        _, vx, vy, wz = self._data[best]
        return PlanarTwist(t_s=float(ts[best]), vx_ms=vx, vy_ms=vy, wz_rads=wz)


def load_camera_model(run: str, raw_dir: Path, image_width: int, sensor):
    """Camera model for a run: exact intrinsics when extracted, FOV fallback.

    Prefers ``<RUN>/camera_info.json`` (written by the aux bag-extraction
    pass; exact fx/cx). Falls back to the pinhole model derived from the
    sensor's datasheet HFOV. Returns ``(camera_model, source_str)``.
    """
    from riskam.kinematics import CameraModel

    info_path = raw_dir / run / "camera_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        return (
            CameraModel.from_intrinsics(info["fx"], info["cx"]),
            "camera_info",
        )
    if sensor.rgb_hfov_deg is None:
        raise ValueError(
            f"No camera_info.json for run {run} and sensor "
            f"{sensor.name!r} has no rgb_hfov_deg fallback."
        )
    return CameraModel.from_hfov(sensor.rgb_hfov_deg, image_width), "hfov_fallback"
