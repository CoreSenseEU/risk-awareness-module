"""
ml.featextr

Feature extraction module for risk awareness.
"""

from time import time


from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

from riskam.ml import depth, humandet
from riskam import score

HUMAN_DETECTOR_MODEL = "hustvl/yolos-tiny"


def extract_human_risk_awareness_features(
    image_path: str,
    depth_gamma: float,
    gaze_face_offset_lower_threshold_ratio: float,
    gaze_face_offset_upper_threshold_ratio: float,
    track_bboxes: bool = False,
) -> dict:
    """
    Extract the features related to human risk from the given image.
    """
    t = time()

    # Extract human bounding boxes, offset scores, and keypoints
    human_bboxes, offset_scores, keypoints_np = humandet.detect_humans(
        image_path, track_bboxes
    )

    # Estimate depth (MiDaS)
    rel_depth, bbox_relative_depths = depth.estimate_depth(
        image_path, human_bboxes, gamma=depth_gamma
    )

    # Estimate gaze scores
    gaze_scores = humandet.gaze_scores(
        keypoints_np,
        gaze_face_offset_lower_threshold_ratio,
        gaze_face_offset_upper_threshold_ratio,
    )

    # Process the features
    if len(human_bboxes) > 0:
        risk_features = {
            "proximity": np.array(bbox_relative_depths),
            "x_offset": np.array(offset_scores),
            "gaze": np.array(gaze_scores),
        }

    # If there are no humans, return None
    else:
        risk_features = None

    time_elapsed = time() - t

    return (
        human_bboxes,
        rel_depth,
        risk_features,
    )

    # Estimate the distance of the closest human
    # - MiDaS: https://github.com/isl-org/MiDaS
    # - Works great!
    # Estimate the human's emotion (discomfort)
    # - FER (facial expression recognition)
    # - AffectNet looks good -> although this is JUST on faces -> need to cut
    # Estimate the human's gaze (is he looking at the robot?)
    # Estimate environmental hazards


def _detect_humans(image: Image) -> list:
    """
    Detect humans in the given image. Returns a list of bounding boxes.
    """

    # Use a pre-trained model to detect humans in the image
    # Return a list of bounding boxes for the detected humans
