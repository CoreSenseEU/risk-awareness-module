"""
data.run_datasets

Wiring registry for the per-run RGB-D datasets (the CoreSense RoboCup
recordings). One :class:`RunDataset` per dataset bundles everything a
consumer needs to walk runs and frames: raw-dataset root, run inventory,
recording platform, depth/odometry index classes, camera loader, and the
optional ground-truth/split files (2023 only — 2024 is unannotated by
design).

This module is deliberately torch-free and does NOT import
``riskam.experiments`` or ``riskam.ml`` (importing those loads the YOLO
model); scripts that only need dataset wiring (video rendering, event-table
builds) import from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from riskam.data import cs_robocup_2023, cs_robocup_2024
from riskam.data.paths import (
    CS_ROBOCUP_2023,
    CS_ROBOCUP_2023_GROUND_TRUTH_PATH,
    CS_ROBOCUP_2023_ML_RAW_DIR,
    CS_ROBOCUP_2024,
    CS_ROBOCUP_2024_ML_RAW_DIR,
)
from riskam.data.splits import CS_ROBOCUP_2023_SPLIT_PATH
from riskam.platforms import TIAGO_XTION, RobotPlatform


@dataclass(frozen=True)
class RunDataset:
    """Everything needed to consume one per-run RGB-D dataset."""

    name: str
    raw_dir: Path
    platform: RobotPlatform
    depth_index_cls: type
    odom_index_cls: type
    # (run, image_width, sensor) -> (CameraModel, source_str)
    camera_loader: Callable
    ground_truth_path: Path | None = None
    split_path: Path | None = None

    @property
    def runs(self) -> tuple[str, ...]:
        """Runs present on disk with a non-empty ``rgb/`` directory.

        Discovered dynamically (not hardcoded) so partially-extracted
        checkouts degrade gracefully — e.g. 2023's RB_04 exists as a
        directory but has no extracted frames and is therefore excluded.
        """
        if not self.raw_dir.is_dir():
            return ()
        found = []
        for run_dir in sorted(self.raw_dir.iterdir()):
            rgb_dir = run_dir / "rgb"
            if rgb_dir.is_dir() and any(rgb_dir.glob("*.png")):
                found.append(run_dir.name)
        return tuple(found)


RUN_DATASETS: dict[str, RunDataset] = {
    CS_ROBOCUP_2023: RunDataset(
        name=CS_ROBOCUP_2023,
        raw_dir=CS_ROBOCUP_2023_ML_RAW_DIR,
        platform=TIAGO_XTION,
        depth_index_cls=cs_robocup_2023.CSRobocup2023DepthIndex,
        odom_index_cls=cs_robocup_2023.CSRobocup2023OdomIndex,
        camera_loader=cs_robocup_2023.load_camera_model,
        ground_truth_path=CS_ROBOCUP_2023_GROUND_TRUTH_PATH,
        split_path=CS_ROBOCUP_2023_SPLIT_PATH,
    ),
    CS_ROBOCUP_2024: RunDataset(
        name=CS_ROBOCUP_2024,
        raw_dir=CS_ROBOCUP_2024_ML_RAW_DIR,
        platform=TIAGO_XTION,
        depth_index_cls=cs_robocup_2024.CSRobocup2024DepthIndex,
        odom_index_cls=cs_robocup_2024.CSRobocup2024OdomIndex,
        camera_loader=cs_robocup_2024.load_camera_model,
        # Unannotated by design (paper plan: eval layers, not labels).
        ground_truth_path=None,
        split_path=None,
    ),
}

# Per-run depth-index classes keyed by dataset name. Historically lived in
# riskam.experiments (which re-imports it for backwards compatibility);
# defined here so scripts can get it without triggering the YOLO model load
# that importing riskam.experiments implies.
RUN_BASED_DEPTH_INDEXES: dict[str, type] = {
    name: ds.depth_index_cls for name, ds in RUN_DATASETS.items()
}


def all_frames_in_time_order(ds: RunDataset, run: str) -> list[Path]:
    """All RGB frame paths of a run, sorted by their timestamp stem."""
    rgb_dir = ds.raw_dir / run / "rgb"
    if not rgb_dir.is_dir():
        return []
    return sorted(rgb_dir.glob("*.png"), key=lambda p: float(p.stem))


def labelled_gt(ds: RunDataset) -> dict[str, dict]:
    """The run-nested ground-truth dict, or ``{}`` when the dataset has none."""
    import json

    if ds.ground_truth_path is None or not ds.ground_truth_path.is_file():
        return {}
    return json.loads(ds.ground_truth_path.read_text())
