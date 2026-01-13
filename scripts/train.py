"""
train.py

Trains a risk awareness model.
"""

import argparse
import joblib
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.cs_robocup import CSRoboCup2023
from riskam.data.ml_datasets import DATASETS
from riskam.ml.autoencoder import (
    VariationalAutoEncoder,
)
from riskam.ml import svm

MODELS = {"vae": VariationalAutoEncoder, "svm": None}

ML_MODELS_DIR = Path(__file__).parent.parent / "ml_models"


def train(
    model_type: str,
    dataset: str,
    task: str | None,
    batch_size: int,
    n_epochs: int,
    n_dataloader_workers: int,
) -> None:
    """
    Train a model.
    """

    # Train the model
    if model_type == "svm":
        model = svm.train(dataset, task)
    else:
        model = MODELS[model_type].train_model(
            dataset, task, batch_size, n_epochs, n_dataloader_workers
        )

    # Store it
    task_str = f"-{task}" if task else ""

    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_id = ML_MODELS_DIR / f"{dataset}{task_str}-{model_type}"

    if model_type == "svm":
        model_id = str(model_id) + ".joblib"
        joblib.dump(model, model_id)
    else:
        model_id += str(model_id) + f"-{n_epochs}ep.pt"
        torch.save(model, model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "model_type",
        choices=MODELS.keys(),
        type=str,
        help="The type of the model to train.",
    )
    parser.add_argument(
        "dataset", choices=DATASETS.keys(), type=str, help="The dataset to train on."
    )
    parser.add_argument(
        "--task",
        choices=CSRoboCup2023.TASKS,
        type=str,
        default=None,
        help="The dataset task to train the model on. Required for CS Robocup dataset.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="The training batch size."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="The number of epochs."
    )
    parser.add_argument(
        "--n_dataloader_workers",
        type=int,
        default=1,
        help="The number of CPU workers for the DataLoader.",
    )

    args = parser.parse_args()

    train(
        args.model_type,
        args.dataset,
        args.task,
        args.batch_size,
        args.n_epochs,
        args.n_dataloader_workers,
    )
