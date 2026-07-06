"""
data.cs_robocup_2024

The CoreSense RoboCup 2024 dataset (RoboCup 2024, Eindhoven) recorded with a
PAL TIAGo. Same on-disk layout as cs_robocup_2023 with two differences:
depth frames are 16-bit PNGs (still raw uint16 millimetres) and runs carry
task-based names instead of RB_XX. Ego-motion twists in ``odom.csv`` are
derived from the bags' ``/tf`` odom→base transforms (the 2024 bags carry no
odometry topic); see :mod:`riskam.data.extract_cs_robocup`.
"""

from riskam.data.cs_robocup_indices import (
    CSRobocupDepthIndex,
    CSRobocupOdomIndex,
    RGB_ODOM_MAX_DT_S,
    load_camera_model as _load_camera_model,
)
from riskam.data.paths import CS_ROBOCUP_2024_ML_RAW_DIR

# The 10 runs across 6 RoboCup@Home tasks. "storing" has no try-1 recording;
# run IDs keep the original try numbers from the published archives.
RUNS = [
    "carry_1",
    "carry_2",
    "gpsr_1",
    "gpsr_2",
    "receptionist_1",
    "receptionist_2",
    "restaurant_1",
    "stickler_1",
    "stickler_2",
    "storing_2",
]


class CSRobocup2024DepthIndex(CSRobocupDepthIndex):
    """:class:`CSRobocupDepthIndex` bound to the 2024 raw dataset."""

    def __init__(self, run: str) -> None:
        super().__init__(run, CS_ROBOCUP_2024_ML_RAW_DIR)


class CSRobocup2024OdomIndex(CSRobocupOdomIndex):
    """:class:`CSRobocupOdomIndex` bound to the 2024 raw dataset."""

    def __init__(self, run: str, tolerance_s: float = RGB_ODOM_MAX_DT_S) -> None:
        super().__init__(run, CS_ROBOCUP_2024_ML_RAW_DIR, tolerance_s)


def load_camera_model(run: str, image_width: int, sensor):
    """:func:`riskam.data.cs_robocup_indices.load_camera_model` for 2024."""
    return _load_camera_model(run, CS_ROBOCUP_2024_ML_RAW_DIR, image_width, sensor)
