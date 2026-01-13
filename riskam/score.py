"""
riskam.score

The risk awareness score computation module.
"""

import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# pylint: disable=no-member

W_PROXIMITY_EMPIRICAL_DEFAULT = 0.7
W_GAZE_EMPIRICAL_DEFAULT = 0.25
W_POSITION_EMPIRICAL_DEFAULT = 0.05

RISK_SCORE_BREAKPOINTS = [0, 0.3, 0.6, 1]
VERY_SMALL_RISK_VALUE = 0.00001

# Report the risk score aggregated over n frames
N_FRAMES_AGGREGATE = 5
AGGREGATION_METHOD = "mean"

prev_risks = []


def risk_awareness_score(
    features: dict[str, np.ndarray] | None,
    w_proximity: float = W_PROXIMITY_EMPIRICAL_DEFAULT,
    w_gaze: float = W_GAZE_EMPIRICAL_DEFAULT,
    w_position: float = W_POSITION_EMPIRICAL_DEFAULT,
) -> tuple[float, int]:
    """
    For the given image features, calculates the risk awareness score.
    """
    global prev_risks  # pylint: disable=global-statement

    # If no features are available, return 0 (there are no humans)
    if features is None:
        return 0.0, -1

    # Calculate the risk scores
    risk_scores = (
        w_proximity * (1 - features["proximity"])  # Inversely proportional to proximity
        + w_gaze * (1 - features["gaze"])  # Inversely proportional to gaze
        + w_position * features["x_offset"]
    )

    # Select the maximum risk score & the corresponding index, clamp
    risk_score = max(VERY_SMALL_RISK_VALUE, min(1, max(risk_scores)))
    max_risk_idx = np.argmax(risk_scores)

    # Append the risk score to the previous values
    prev_risks.append(risk_score)
    prev_risks = prev_risks[-N_FRAMES_AGGREGATE:]

    # Aggregate the risk scores
    if AGGREGATION_METHOD == "mean":
        risk_score = np.mean(prev_risks)

    return risk_score, max_risk_idx
