"""
data.cs_robocup

The CoreSense RoboCup dataset recorded with a social robot.

https://zenodo.org/records/13748065
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    YolosImageProcessor,
    YolosForObjectDetection,
)
from tqdm import tqdm

from riskam.data.paths import (
    CS_ROBOCUP_2023_ML_DIR,
    CS_ROBOCUP_2023_ML_RAW_DIR,
    CS_ROBOCUP_2023_ML_FEAT_DIR,
)


CS_ROBOCUP_N_HUMANS_PATH = CS_ROBOCUP_2023_ML_DIR / "n_humans.json"
CS_ROBOCUP_BBOXES_PATH = CS_ROBOCUP_2023_ML_DIR / "bboxes.json"
CS_ROBOCUP_DISTANCES_PATH = CS_ROBOCUP_2023_ML_DIR / "distances.json"

CS_ROBOCUP_HUMANS_PATH = CS_ROBOCUP_2023_ML_DIR / "humans.json"
CS_ROBOCUP_MULTIPLE_HUMANS_PATH = CS_ROBOCUP_2023_ML_DIR / "multiple_humans.json"
CS_ROBOCUP_CLOSE_HUMANS_PATH = CS_ROBOCUP_2023_ML_DIR / "close_humans.json"


# Script constants
ROUTINES = ["detect_humans", "visualize", "compute_distances", "construct"]

# Invalid depth value
INVALID_DEPTH = -1

# Random seed
RANDOM_SEED = 5


class CSRoboCup2023(Dataset):
    """
    The CoreSense RoboCup PyTorch dataset object.

    Currently implemented as a SUPERVISED dataset with run identifiers as labels. That said,
    it's an unsupervised dataset for ML intents and purposes.
    """

    TASKS = [
        "humans",  # 1 = at least one human, 0 = no humans in the frame
        "close_humans",  # 1 = at least one human is too close, 0 = no humans that are close
        "multiple_humans",  # 1 = there are at least n humans, 0 = there are fewer than n humans
    ]

    SPLIT = ["train", "val", "risk"]

    def __init__(
        self,
        root_dir: str,
        task: str,
        split: str,
        transform=None,
        use_features: bool = True,
    ):
        # Initialize the dataset's variables
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = []

        self.classes = None
        self.class_to_idx = None

        # Check if split valid
        if split not in self.SPLIT:
            raise ValueError(f"Unknown split: {split}")

        # Establish the task label source path
        if task == "humans":
            label_src_path = CS_ROBOCUP_HUMANS_PATH
        elif task == "close_humans":
            label_src_path = CS_ROBOCUP_CLOSE_HUMANS_PATH
        elif task == "multiple_humans":
            label_src_path = CS_ROBOCUP_MULTIPLE_HUMANS_PATH
        else:
            raise ValueError(f"Unknown task type: {task}")

        # Assign image paths and labels
        label_src = json.loads(label_src_path.read_text())

        rb = "RB_01"  # For now, we only have one run
        for img_name in label_src[rb][split]:
            self.image_paths.append(self.root_dir / rb / "rgb" / img_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load the image
        image = np.load(img_path)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image


def extract_features() -> None:
    """
    Extracts ViT features from the CoreSense RoboCup dataset.
    """

    if CS_ROBOCUP_2023_ML_FEAT_DIR.exists():
        print("The features have already been extracted. Skipping...")
        return

    print("+++ EXTRACTING FEATURES +++")

    CS_ROBOCUP_2023_ML_FEAT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pretrained feature extractor and model

    model_name = "google/vit-base-patch16-224"  # Vision Transformer model
    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set model to evaluation mode

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"

        print(f"Run RB_0{rb}:")

        rb_rgb_dir = CS_ROBOCUP_2023_ML_RAW_DIR / rb_label / "rgb"
        rb_feat_dir = CS_ROBOCUP_2023_ML_FEAT_DIR / rb_label / "rgb"
        rb_feat_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(list(rb_rgb_dir.glob("*.npy"))):
            image = np.load(image_path)
            inputs = feature_extractor(image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            feat_path = rb_feat_dir / f"{image_path.stem}.npy"

            np.save(feat_path, outputs.last_hidden_state.mean(dim=1).numpy())


def detect_humans_in_frames() -> None:
    """
    Go over the CoreSense Robocup dataset and for each frame, extract the number of humans in it
    and the corresponding bounding boxes.
    """

    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    if CS_ROBOCUP_N_HUMANS_PATH.exists() and CS_ROBOCUP_BBOXES_PATH.exists():
        print("The human bonding boxes have already been extracted. Skipping...")
        return

    n_humans = {}
    bboxes = {}

    print("+++ DETECTING HUMANS IN FRAMES +++")

    # Iterate over the runs
    for rb in range(1, 9):
        print(f"Run RB_0{rb}:")

        rb_label = f"RB_0{rb}"

        # Establish the RB in the dicts
        if f"RB_0{rb}" in n_humans and f"RB_0{rb}" in bboxes:
            print("This run has already been processed. Skipping...")
            continue

        n_humans[rb_label] = {}
        bboxes[rb_label] = {}

        rb_rgb_dir = CS_ROBOCUP_2023_ML_RAW_DIR / f"RB_0{rb}" / "rgb"

        for img_path in tqdm(list(rb_rgb_dir.glob("*.npy"))):
            # Establish the entry in the dicts
            n_humans[rb_label][img_path.name] = 0
            bboxes[rb_label][img_path.name] = []

            # Load the image
            try:
                image_array = np.load(img_path)
                image = Image.fromarray(np.uint8(image_array)).convert("RGB")
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error loading image {img_path.name}: {e}")
                continue

            # Process the image
            inputs = processor(image, return_tensors="pt")
            outputs = model(**inputs)

            # Process the results
            target_sizes = torch.tensor([image.size[::-1]])
            results = (
                processor.post_process_object_detection(  # pylint: disable=no-member
                    outputs, threshold=0.9, target_sizes=target_sizes
                )[0]
            )

            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if model.config.id2label[label.item()] != "person":
                    continue

                n_humans[rb_label][img_path.name] += 1
                bboxes[rb_label][img_path.name].append(box.tolist())

                box = [round(i, 2) for i in box.tolist()]
                """
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )
                """

        # Save the results
        CS_ROBOCUP_N_HUMANS_PATH.write_text(json.dumps(n_humans, indent=4))
        CS_ROBOCUP_BBOXES_PATH.write_text(json.dumps(bboxes, indent=4))


def _find_closest_rgb_image(
    rgb_images_sorted: list[float], depth_image_timestamp: float
) -> str:
    """
    Finds the closest RGB image based on depth image timestamp. Utilizes binary halving.
    """

    left = 0
    right = len(rgb_images_sorted) - 1

    while left <= right:
        mid = (left + right) // 2

        if rgb_images_sorted[mid] == depth_image_timestamp:
            return f"{rgb_images_sorted[mid]}.png"
        elif rgb_images_sorted[mid] < depth_image_timestamp:
            left = mid + 1
        else:
            right = mid - 1

    return f"{rgb_images_sorted[right]}.npy"


def compute_human_distances() -> None:
    """
    Computes the distances between the robot and the detected human. The method is
    median bbox value.
    """

    if CS_ROBOCUP_DISTANCES_PATH.exists():
        print("The human distances have already been computed. Skipping...")
        return

    print("+++ COMPUTING HUMAN DISTANCES +++")

    bboxes = json.loads(CS_ROBOCUP_BBOXES_PATH.read_text())
    distances = {}

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"

        distances[rb_label] = {}

        rb_depth_dir = CS_ROBOCUP_2023_ML_RAW_DIR / f"RB_0{rb}" / "depth"
        rgb_images = [
            float(".".join(i.split(".")[:-1]))
            for i in sorted(list(bboxes[rb_label].keys()))
        ]

        print(f"Run RB_0{rb}:")

        for img_path in tqdm(sorted(list(rb_depth_dir.glob("*.npy")))):
            try:
                depth_image = np.load(img_path)
            except Exception as e:
                print(f"Error loading image {img_path.name}: {e}")
                continue

            # Find the closest RGB image
            closest_rgb_image = _find_closest_rgb_image(
                rgb_images, float(img_path.stem)
            )

            # Some images may not have bounding boxes (no humans in the frame)
            if closest_rgb_image not in bboxes[rb_label]:
                continue

            distances[rb_label][closest_rgb_image] = []

            for box in bboxes[rb_label][closest_rgb_image]:
                x1, y1, x2, y2 = map(int, box)
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # Extract depth values within the bounding box
                sub_image = depth_image[y1:y2, x1:x2]

                # Compute the median depth value, ignoring NaNs
                if np.isnan(sub_image).all():
                    median_depth = np.nan
                else:
                    median_depth = np.nanmedian(sub_image)

                if np.isnan(median_depth):
                    median_depth = INVALID_DEPTH

                distances[rb_label][closest_rgb_image].append(float(median_depth))

    CS_ROBOCUP_DISTANCES_PATH.write_text(json.dumps(distances, indent=4))


def draw_bbox(image_cv2, box):
    """Draw bounding boxes on the image."""
    x1, y1, x2, y2 = box
    cv2.rectangle(  # pylint: disable=no-member
        image_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
    )

    return image_cv2


def trainval_split(dataset: dict) -> dict:
    """
    Splits the given dataset into train and validation sets
    """

    all_filenames = []

    for rb_label, rb_record in dataset.items():
        for img_name in rb_record.keys():
            all_filenames.append((rb_label, img_name))

    train, val = train_test_split(
        all_filenames, test_size=0.1, random_state=RANDOM_SEED
    )

    return {"train": train, "val": val}


def construct_cs_robocup() -> None:
    """
    Constructs the CS-Robocup dataset variants based on

    The output is JSON files that assign labels to images in the dataset.
    """

    if (
        CS_ROBOCUP_HUMANS_PATH.exists()
        and CS_ROBOCUP_CLOSE_HUMANS_PATH.exists()
        and CS_ROBOCUP_MULTIPLE_HUMANS_PATH.exists()
    ):
        print("The CS Robocup datasets have already been constructed. Skipping...")
        return

    print("+++ CONSTRUCTING CS ROBOCUP DATASETS +++")

    n_humans = json.loads(CS_ROBOCUP_N_HUMANS_PATH.read_text())
    distances = json.loads(CS_ROBOCUP_DISTANCES_PATH.read_text())

    # The humans dataset: any humans in the frame = 1, no humans = 0
    humans = {}

    # The close humans dataset: any humans too close = 1, no humans too close = 0
    # The "close" threshold is set to make the risk/no risk split roughly 20/80
    close_humans = {}

    # The multiple humans dataset: at least n humans = 1, fewer than n humans = 0
    # The "n" threshold is set to make the risk/no risk split roughly 20/80
    multiple_humans = {}

    # Compute the quantile for close humans
    dist_values = []

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"

        for img_name, dists in distances[rb_label].items():
            if len(dists) == 0:
                continue

            valid_dists = [d for d in dists if d != INVALID_DEPTH]

            if len(valid_dists) == 0:
                continue

            dist_values.append(min(valid_dists))

    dist_quantile = np.quantile(sorted(dist_values), 0.2)
    print(f"Quantile for distance: {dist_quantile}")

    # Compute the quantile for number of humans and sort the humans dataset while we're at it
    n_humans_values = []
    n_humans_img = [0] * 2

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"

        # Initialize the RB in the dicts
        if rb_label not in humans:
            humans[rb_label] = {}

        humans_img = [[] for _ in range(2)]

        for img_name, n in n_humans[rb_label].items():
            label = 1 if n > 0 else 0
            humans_img[label].append(img_name)
            # humans[rb_label][img_name] = label
            n_humans_img[label] += 1
            n_humans_values.append(n)

        # Split the RB run images into train and validation sets
        if len(humans_img[0]) == 0 or len(humans_img[1]) == 0:
            continue

        train, val = train_test_split(
            humans_img[0], test_size=0.1, random_state=RANDOM_SEED
        )

        humans[rb_label]["train"] = train
        humans[rb_label]["val"] = val
        humans[rb_label]["risk"] = humans_img[1]

    n_humans_quantile = np.quantile(sorted(n_humans_values), 0.8)
    print(f"Quantile for number of humans: {n_humans_quantile}")

    print()
    print("CS ROBOCUP - HUMANS TASK")
    print(f"Number of images with no humans: {n_humans_img[0]}")
    print(f"Number of images with humans: {n_humans_img[1]}")
    print(
        "Percentage of images with no humans: "
        f"{n_humans_img[0] / sum(n_humans_img) * 100}%"
    )
    print(
        "Percentage of images with humans: "
        f"{n_humans_img[1] / sum(n_humans_img) * 100}%"
    )

    # Go over the distances and fill in the close humans dataset
    n_close_humans_img = [0] * 2

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"

        if rb_label not in close_humans:
            close_humans[rb_label] = {}

        close_humans_img = [[] for _ in range(2)]

        for img_name, dists in distances[rb_label].items():
            # If there are no humans detected, then there are no close humans
            if len(dists) == 0:
                # close_humans[rb_label][img_name] = 0
                label = 0
                n_close_humans_img[0] += 1
            # If all the depth values are invalid, we skip the image altogether
            elif all(d == INVALID_DEPTH for d in dists):
                continue
            # Otherwise, we assign the label based on the quantile comparison
            else:
                valid_dists = [d for d in dists if d != INVALID_DEPTH]
                label = 1 if any(d < dist_quantile for d in valid_dists) else 0
                # close_humans[rb_label][img_name] = label
                n_close_humans_img[label] += 1

            close_humans_img[label].append(img_name)

        # Split the RB run images into train and validation sets
        if len(close_humans_img[0]) == 0 or len(close_humans_img[1]) == 0:
            continue

        train, val = train_test_split(
            close_humans_img[0], test_size=0.1, random_state=RANDOM_SEED
        )
        close_humans[rb_label]["train"] = train
        close_humans[rb_label]["val"] = val
        close_humans[rb_label]["risk"] = close_humans_img[1]

    # Print the stats
    print()
    print("+++ CS ROBOCUP - CLOSE_HUMANS TASK +++")
    print(f"Number of images with no close humans: {n_close_humans_img[0]}")
    print(f"Number of images with close humans: {n_close_humans_img[1]}")
    print(
        "Percentage of images with no close humans: "
        f"{n_close_humans_img[0] / sum(n_close_humans_img) * 100}%"
    )
    print(
        "Percentage of images with close humans: "
        f"{n_close_humans_img[1] / sum(n_close_humans_img) * 100}%"
    )

    # Go over the number of humans and fill in the multiple humans dataset
    n_multiple_humans_img = [0] * 2
    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"

        if rb_label not in multiple_humans:
            multiple_humans[rb_label] = {}

        multiple_humans_img = [[] for _ in range(2)]

        for img_name, n in n_humans[rb_label].items():
            label = 1 if n > n_humans_quantile else 0
            # multiple_humans[rb_label][img_name] = label
            multiple_humans_img[label].append(img_name)
            n_multiple_humans_img[label] += 1

        # Split the RB run images into train and validation sets
        if len(multiple_humans_img[0]) == 0 or len(multiple_humans_img[1]) == 0:
            continue

        train, val = train_test_split(
            multiple_humans_img[0], test_size=0.1, random_state=RANDOM_SEED
        )
        multiple_humans[rb_label]["train"] = train
        multiple_humans[rb_label]["val"] = val
        multiple_humans[rb_label]["risk"] = multiple_humans_img[1]

    # Print the stats
    print()
    print("+++ CS ROBOCUP - MULTIPLE_HUMANS TASK +++")
    print(f"Number of images with fewer than n humans: {n_multiple_humans_img[0]}")
    print(f"Number of images with at least n humans: {n_multiple_humans_img[1]}")
    print(
        "Percentage of images with fewer than n humans: "
        f"{n_multiple_humans_img[0] / sum(n_multiple_humans_img) * 100}%"
    )
    print(
        "Percentage of images with at least n humans: "
        f"{n_multiple_humans_img[1] / sum(n_multiple_humans_img) * 100}%"
    )

    # Save the datasets
    CS_ROBOCUP_2023_ML_DIR.mkdir(parents=True, exist_ok=True)

    CS_ROBOCUP_HUMANS_PATH.write_text(json.dumps(humans, indent=4))
    CS_ROBOCUP_CLOSE_HUMANS_PATH.write_text(json.dumps(close_humans, indent=4))
    CS_ROBOCUP_MULTIPLE_HUMANS_PATH.write_text(json.dumps(multiple_humans, indent=4))


def visualize_humans_bboxes() -> None:
    """
    Visualize the bounding boxes of the detected humans in the CoreSense RoboCup dataset.
    """

    bboxes = json.loads(CS_ROBOCUP_BBOXES_PATH.read_text())

    for rb in range(1, 9):
        rb_label = f"RB_0{rb}"
        rb_rgb_dir = CS_ROBOCUP_2023_ML_RAW_DIR / f"RB_0{rb}" / "rgb"

        for img_path in tqdm(sorted(list(rb_rgb_dir.glob("*.npy")))):
            try:
                image_array = np.load(img_path)
                image = Image.fromarray(np.uint8(image_array)).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path.name}: {e}")
                continue
            image_cv2 = cv2.cvtColor(  # pylint: disable=no-member
                np.array(image), cv2.COLOR_RGB2BGR  # pylint: disable=no-member
            )

            for box in bboxes[rb_label][img_path.name]:
                image = draw_bbox(image_cv2, box)

            # Display the image
            cv2.imshow(  # pylint: disable=no-member
                "Object Detection Viewer", image_cv2
            )

            # Handle key presses
            key = cv2.waitKey(0)  # pylint: disable=no-member
            if key == 27:  # Escape key
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS RoboCup dataset service.")
    parser.add_argument(
        "routine",
        choices=ROUTINES,
        type=str,
        help="The routine to execute.",
    )

    args = parser.parse_args()

    if args.routine == "detect_humans":
        detect_humans_in_frames()
    elif args.routine == "visualize":
        visualize_humans_bboxes()
    elif args.routine == "compute_distances":
        compute_human_distances()
    elif args.routine == "construct":
        construct_cs_robocup()
    else:
        raise ValueError(f"Unknown routine: {args.routine}")
