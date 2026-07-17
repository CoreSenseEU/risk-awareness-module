"""
data.crowdbot_v2

The CrowdBot v2 dataset: EPFL's Qolo standing mobility robot navigating a
dense outdoor pedestrian street (Lausanne market, 2021-04-24) under RDS
shared control, faces defaced. Same on-disk layout as the CS RoboCup
datasets (depth as 16-bit PNG millimetres, like 2024), so the year-agnostic
nearest-neighbour indices are reused unchanged. See
:mod:`riskam.data.extract_crowdbot` for provenance and extraction quirks.
"""

from riskam.data.cs_robocup_indices import (
    CSRobocupDepthIndex,
    CSRobocupOdomIndex,
    RGB_ODOM_MAX_DT_S,
    load_camera_model as _load_camera_model,
)
from riskam.data.extract_crowdbot import RUNS as _RUNS
from riskam.data.paths import CROWDBOT_V2_ML_RAW_DIR

# The 7 RDS runs, labelled by recording start time (hhmm).
RUNS = list(_RUNS)


class CrowdbotV2DepthIndex(CSRobocupDepthIndex):
    """:class:`CSRobocupDepthIndex` bound to the crowdbot_v2 raw dataset."""

    def __init__(self, run: str) -> None:
        super().__init__(run, CROWDBOT_V2_ML_RAW_DIR)


class CrowdbotV2OdomIndex(CSRobocupOdomIndex):
    """:class:`CSRobocupOdomIndex` bound to the crowdbot_v2 raw dataset."""

    def __init__(self, run: str, tolerance_s: float = RGB_ODOM_MAX_DT_S) -> None:
        super().__init__(run, CROWDBOT_V2_ML_RAW_DIR, tolerance_s)


def load_camera_model(run: str, image_width: int, sensor):
    """:func:`riskam.data.cs_robocup_indices.load_camera_model` for crowdbot_v2."""
    return _load_camera_model(run, CROWDBOT_V2_ML_RAW_DIR, image_width, sensor)
