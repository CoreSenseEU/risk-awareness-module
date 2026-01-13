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
)

DATASETS = {
    "cs_robocup_2023": {
        "class": CSRoboCup2023,
        "img_dir": CS_ROBOCUP_2023_ML_RAW_DIR,
        "ground_truth_path": CS_ROBOCUP_2023_GROUND_TRUTH_PATH,
    }
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
