"""
data.ml_datasets

"Umbrella" service functionality for the ML datasets.
"""

import torch.nn as nn
from torch.utils.data import Dataset

from riskam.data.cs_robocup_2023 import CSRoboCup2023
from riskam.data.paths import (
    CS_ROBOCUP_2023_ML_RAW_DIR,
    CS_ROBOCUP_2023_GROUND_TRUTH_PATH,
    CS_ROBOCUP_2024_ML_RAW_DIR,
    CS_ROBOCUP_2024_GROUND_TRUTH_PATH,
)
from riskam.platforms import TIAGO_XTION

DATASETS = {
    "cs_robocup_2023": {
        "class": CSRoboCup2023,
        "img_dir": CS_ROBOCUP_2023_ML_RAW_DIR,
        "ground_truth_path": CS_ROBOCUP_2023_GROUND_TRUTH_PATH,
        # Recording platform (T2.7). Carries depth-sensor physics
        # (near-clip dead zone, sparse-valid threshold) AND the platform's
        # safety distance, separately from the dataset proper. Adding a
        # second dataset recorded on the same TIAGo is then a one-liner.
        # See riskam/platforms.py for the canonical definitions.
        "platform": TIAGO_XTION,
    },
    "cs_robocup_2024": {
        # No PyTorch dataset class yet — the 2024 data currently serves the
        # scoring/eval pipeline only (no ML-task labels annotated so far).
        "class": None,
        "img_dir": CS_ROBOCUP_2024_ML_RAW_DIR,
        "ground_truth_path": CS_ROBOCUP_2024_GROUND_TRUTH_PATH,
        # Same TIAGo family as 2023 (head_front_camera namespace); the
        # per-run camera_info.json carries the exact intrinsics.
        "platform": TIAGO_XTION,
    },
}


def get_dataset(
    name: str, transform: nn.Module, task: str | None, split: str
) -> Dataset:
    """
    Get the dataset object by name.
    """

    try:
        dataset_obj = DATASETS[name]["class"](
            root_dir=DATASETS[name]["dir"], transform=transform, task=task, split=split
        )
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: '{name}'") from exc

    return dataset_obj
