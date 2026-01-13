"""
data.annotator

A simple tool for annotating datasets.
"""

import argparse
import json

import cv2

from riskam.data.ml_datasets import DATASETS

RISK_CLASSES = {
    0: "No risk",
    1: "Low risk",
    2: "Medium risk",
    3: "High risk",
}

# pylint: disable=no-member


def annotate_dataset(dataset: str, run: str | None = None) -> None:
    """
    Launches the annotation tool for the given dataset
    """

    # Get the dataset paths
    try:
        dataset_img_dir = DATASETS[dataset]["img_dir"]
        dataset_ground_truth_path = DATASETS[dataset]["ground_truth_path"]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: '{dataset}'") from exc

    # Incorporate the "run" if CS RoboCup 2023 and load the existing ground truth
    annotations = {}

    if dataset == "cs_robocup_2023":
        dataset_img_dir = dataset_img_dir / run / "rgb"

        if dataset_ground_truth_path.exists():
            with open(dataset_ground_truth_path, "r", encoding="utf-8") as f:
                ground_truth = json.load(f)
        else:
            ground_truth = {}

    # Get the list of image files
    image_files = sorted(
        [
            img_path
            for img_path in dataset_img_dir.iterdir()
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
    )

    # Iterate over the images and annotate
    for img_path in image_files:
        print(f"Annotating: {img_path}")

        # Load the image
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"Failed to load the image: {img_path}")
            continue

        # Display the image
        cv2.imshow("Risk annotator", image)

        # Wait indefinitely for a key press
        key = cv2.waitKey(0)

        # ESC key (27) to exit the annotation process
        if key == 27:
            print("Exiting annotation...")
            break

        # Check if key pressed is between '0' and '9'
        if ord("0") <= key <= ord("9"):
            label = int(chr(key))
            annotations[img_path.name] = label
            print(f"Labeled {img_path.name} as {label}")
        else:
            print("Invalid key pressed, skipping this image.")

    # Close the annotator window
    cv2.destroyAllWindows()

    # Assemble ground truth JSON
    if dataset == "cs_robocup_2023":
        ground_truth[run] = annotations
    else:
        ground_truth = annotations

    # Save the annotations
    with open(dataset_ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=4)
