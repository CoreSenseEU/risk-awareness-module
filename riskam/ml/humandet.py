"""
ml.humandet

Human detection for risk awareness.
"""

from pathlib import Path

import torch
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

from riskam.data.paths import ML_MODELS_DIR

# OpenCV throws no-member linting errors
# pylint: disable=no-member

# Check if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO Pose model on GPU
YOLO_POSE_MODEL_PATH = ML_MODELS_DIR / "yolo11n-pose.pt"
model = YOLO(YOLO_POSE_MODEL_PATH, verbose=False)


FACE_OFFSET_LOWER_THRESHOLD_RATIO_EMPIRICAL_DEFAULT = 0.1
FACE_OFFSET_UPPER_THRESHOLD_RATIO_EMPIRICAL_DEFAULT = 0.2


class BboxTracker:
    """
    A simple bounding box tracker for tracking humans across frames, introducing
    spatio-temporal continuity.
    """

    IOU_THRESHOLD = 0.5
    CONSECUTIVE_FRAMES_THRESHOLD = 3

    def __init__(self, iou_threshold=0.5, min_consecutive_frames=3):
        self.iou_threshold = iou_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.tracked_bboxes = {}
        self.next_bbox_id = 0

    def iou(self, box1, box2):
        """
        Computes the Intersection over Union (IoU) score between two bounding boxes.
        """
        inter_left = max(box1[0], box2[0])
        inter_top = max(box1[1], box2[1])
        inter_right = min(box1[2], box2[2])
        inter_bottom = min(box1[3], box2[3])

        inter_area = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter_area / float(box1_area + box2_area - inter_area)

    def update(self, detections: list[tuple[int, int, int, int]]) -> list[bool]:
        """
        Updates the tracked bounding boxes with the new detections.
        """

        updated_tracked = {}
        bboxes_confirmed = [False] * len(detections)

        # Associate new detections with tracked boxes
        for d, det in enumerate(detections):
            matched_id = None
            max_iou = 0

            for bbox_id, bbox_info in self.tracked_bboxes.items():
                iou_score = self.iou(det, bbox_info["coords"])
                if iou_score > self.iou_threshold and iou_score > max_iou:
                    matched_id = bbox_id
                    max_iou = iou_score

            if matched_id is not None:
                # Update existing bbox
                updated_tracked[matched_id] = {
                    "coords": det,
                    "count": self.tracked_bboxes[matched_id]["count"] + 1,
                }

                # If the bbox has been confirmed for enough consecutive frames, mark as confirmed
                if updated_tracked[matched_id]["count"] >= self.min_consecutive_frames:
                    bboxes_confirmed[d] = True

            else:
                # Add new bbox
                updated_tracked[self.next_bbox_id] = {"coords": det, "count": 1}
                self.next_bbox_id += 1

        # Keep only updated tracked boxes
        self.tracked_bboxes = updated_tracked

        # Return bboxes confirmed by sufficient consecutive detections
        confirmed_bboxes = [
            bbox_info["coords"]
            for bbox_info in self.tracked_bboxes.values()
            if bbox_info["count"] >= self.min_consecutive_frames
        ]

        return bboxes_confirmed


bbox_tracker = BboxTracker()


def detect_humans(
    image_path: str, track_bboxes: bool = True
) -> tuple[list[tuple[int, int, int, int]], list[float], list[bool]]:
    """
    Detects humans in the given image and whether they are likely aware of the robot,
    using YOLOv11n.
    """
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb, verbose=False)

    human_bboxes = []
    keypoints_np = None

    # Extract detections
    for result in results:

        # Append the bounding box
        if len(result.boxes.xyxy) == 0:
            continue  # Skip if no humans detected

        # Extract human bounding boxes
        human_bboxes = result.boxes.xyxy.cpu().tolist()

        # Extract keypoints (only x, y coords)
        keypoints = result.keypoints
        keypoints_np = keypoints.data[..., :2].cpu().numpy()

        # Confirm the detected bounding boxes, filter out the unconfirmed ones
        if track_bboxes:
            bboxes_confirmed = bbox_tracker.update(human_bboxes)

            human_bboxes = [
                bbox
                for bbox, confirmed in zip(human_bboxes, bboxes_confirmed)
                if confirmed
            ]
            keypoints_np = keypoints_np[bboxes_confirmed, :, :]

    # Calculate bbox offsets from the center
    if len(image.shape) == 3:
        _, image_width, _ = image.shape  # Color image (3D)
    else:
        _, image_width = image.shape  # Grayscale image (2D)
    human_bbox_offsets = [
        _bbox_offset_score(bbox, image_width) for bbox in human_bboxes
    ]

    return human_bboxes, human_bbox_offsets, keypoints_np


def gaze_scores(
    keypoints_np: np.ndarray,
    face_offset_lower_threshold_ratio: float = FACE_OFFSET_LOWER_THRESHOLD_RATIO_EMPIRICAL_DEFAULT,
    face_offset_upper_threshold_ratio: float = FACE_OFFSET_UPPER_THRESHOLD_RATIO_EMPIRICAL_DEFAULT,
) -> list[float]:
    """
    Detects the gaze scores for each person in the image.
    """
    gaze_scores_all_bboxes = []

    if keypoints_np is not None:
        for i in range(len(keypoints_np)):
            gaze_scores_all_bboxes.append(
                detect_gaze(
                    keypoints_np,
                    i,
                    face_offset_lower_threshold_ratio,
                    face_offset_upper_threshold_ratio,
                )
            )

    return gaze_scores_all_bboxes


def detect_gaze(
    keypoints_np: np.ndarray,
    human_idx: int,
    face_offset_lower_threshold_ratio: float = FACE_OFFSET_LOWER_THRESHOLD_RATIO_EMPIRICAL_DEFAULT,
    face_offset_upper_threshold_ratio: float = FACE_OFFSET_UPPER_THRESHOLD_RATIO_EMPIRICAL_DEFAULT,
) -> float:
    """
    Detects whether the person at the given index is looking at the robot.
    """
    # Zero keypoints detected -> no human looking at the robot
    if keypoints_np.shape[0] == 0:
        return 0.0

    kpts = keypoints_np[human_idx]

    if kpts.shape[0] == 0:  # Handle cases where keypoints exist but are empty
        # print(f"Skipping person {person_id + 1} due to missing keypoints")
        return 0.0

    # Extract keypoints
    nose = kpts[0]  # Nose (x, y)
    left_shoulder = kpts[5] if kpts.shape[0] > 5 else None
    right_shoulder = kpts[6] if kpts.shape[0] > 6 else None
    left_eye = kpts[1] if kpts.shape[0] > 1 else None
    right_eye = kpts[2] if kpts.shape[0] > 2 else None

    # # ====== METHOD 1: SHOULDER-BASED (Preferred if shoulders are visible) ======
    # if left_shoulder is not None and right_shoulder is not None:
    #     # Compute shoulder midpoint (acts as "neck" reference)
    #     neck_midpoint = (left_shoulder + right_shoulder) / 2
    #     shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)

    #     # Compute head offset (nose vs. shoulders)
    #     head_offset = abs(nose[0] - neck_midpoint[0])
    #     normalized_offset = (
    #         head_offset / shoulder_width if shoulder_width > 0 else float("inf")
    #     )

    #     looking_at_robot = normalized_offset < HEAD_OFFSET_THRESHOLD_RATIO
    #     # print(
    #     #     f"Person {person_id+1} (Shoulder Method) Normalized Head Offset: {normalized_offset:.2f} | Looking: {looking_at_robot}"
    #     # )

    # ====== METHOD 2: FACE-BASED (Fallback if shoulders are occluded) ======
    if left_eye is not None and right_eye is not None:
        # Compute face width
        face_width = np.linalg.norm(right_eye - left_eye)

        # Use eye midpoint as the reference point instead of shoulders
        face_midpoint = (left_eye + right_eye) / 2
        face_offset = abs(nose[0] - face_midpoint[0])

        # Normalize by face width
        normalized_face_offset = (
            face_offset / face_width if face_width > 0 else float("inf")
        )

        looking_at_robot = min(
            1,
            max(
                0,
                (face_offset_upper_threshold_ratio - normalized_face_offset)
                / face_offset_lower_threshold_ratio,
            ),
        )
        # print(
        #     f"Person {person_id+1} (Face Method) Normalized Face Offset: {normalized_face_offset:.2f} | Looking: {looking_at_robot}"
        # )

    else:
        # No reliable keypoints available
        # print(f"Person {person_id+1} skipped due to missing keypoints.")
        looking_at_robot = 0.0

    return looking_at_robot


def _bbox_offset_score(bbox: tuple[int, int, int, int], image_width: int) -> float:
    """
    Calculates the offset score for the given bounding box.
    """
    x1, _, x2, _ = bbox
    bbox_center_x = (x1 + x2) / 2
    img_center_x = image_width / 2

    # Compute normalized offset from the motion axis
    offset = abs(bbox_center_x - img_center_x) / img_center_x

    # Compute motion-axis risk weighting (1 - offset^2)
    offset_score = 1 - offset**2

    return offset_score
