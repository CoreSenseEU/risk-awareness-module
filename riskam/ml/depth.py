"""
ml.depth

The depth estimation module (based on MiDaS).

Using https://pytorch.org/hub/intelisl_midas_v2/
"""

import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt


# OpenCV throws no-member linting errors
# pylint: disable=no-member

# The inverse depth percentile to use as the distance of a bounding box to the camera
RELATIVE_DEPTH_PERCENTILE = 10

GAMMA_EMPIRICAL_DEFAULT = 0.6


MIDAS_MODEL_TYPE = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform


def estimate_depth(
    image_path: str,
    human_bboxes: list[list[int]],
    gamma: float = GAMMA_EMPIRICAL_DEFAULT,
) -> tuple[np.ndarray, dict]:
    """
    Estimates depth for the given image.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_batch = transform(image).to(device)

    # Predict
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize
    normalized_inverse_depths = _depth_gamma_normalization(depth_map, gamma=gamma)
    # Calculate the stats
    bbox_relative_depths = _bbox_relative_depths(
        normalized_inverse_depths, human_bboxes
    )

    return normalized_inverse_depths, bbox_relative_depths


def _bbox_relative_depths(
    inverse_depths: np.ndarray, human_bboxes: list[list[int]]
) -> list[float]:
    """
    Returns the relative depths of the bounding boxes.
    """
    # If no bounding boxes, return None
    if len(human_bboxes) == 0:
        return None

    # # Image-wide stats
    # inv_depths_std = np.std(inverse_depths)

    # BBOX STATS
    # # The list of inverse depth matrices for each bounding box
    # bbox_inv_depths_all = []

    # The list of relative depths, distances of each bounding box to the camera
    bbox_relative_depths = []

    # Iterate over the bounding boxes
    for bbox in human_bboxes:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Slice the image
        bbox_inv_depths = inverse_depths[y1:y2, x1:x2]
        # bbox_inv_depths_all.append(bbox_inv_depths)

        # Calculate the relative depth & append
        bbox_relative_depths.append(
            np.percentile(bbox_inv_depths, RELATIVE_DEPTH_PERCENTILE)
        )

    # Select the closest human
    # closest_idx = np.argmin(bbox_relative_depths)

    # Calculate the closest human's normalized IQR - if there is no variation, set to 0
    # p25, p75 = np.percentile(bbox_inv_depths_all[closest_idx], [25, 75])

    # if inv_depths_std == 0.0:
    #     closest_niqr = 0.0
    # else:
    #     closest_niqr = (p75 - p25) / inv_depths_std

    return bbox_relative_depths


def _depth_gamma_normalization(
    depth_map,
    clip_percentiles=(1, 99),
    gamma=GAMMA_EMPIRICAL_DEFAULT,
    eps=1e-6,
):
    """
    A flexible normalization for MiDaS depth maps that
    (1) robustly clips outliers,
    (2) inverts (closer -> larger),
    (3) applies optional gamma compression,
    (4) yields [0,1] output.

    Parameters
    ----------
    depth_map : np.ndarray
        The MxN relative depth map from MiDaS (or similar).
    clip_percentiles : (float, float), optional
        Percentiles used to clip the depth_map. For example, (1, 99)
        will clip anything below the 1st percentile or above the 99th percentile.
        If None, no clipping is performed.
    gamma : float, optional
        Gamma exponent for the compression.
        - If gamma < 1, near-depth differences are more emphasized.
        - If gamma = 1, no extra compression beyond min–max.
        - If gamma > 1, near-depth differences are less emphasized.
    eps : float, optional
        A small constant to avoid division by zero or NaNs.

    Returns
    -------
    normalized : np.ndarray
        A [0,1] float map. NaNs become 0 by default.
    """

    # 0) Handle NaNs by replacing them with the min of valid values
    valid_mask = ~np.isnan(depth_map)
    if not np.any(valid_mask):
        # If the entire map is NaN, return zeros
        return np.zeros_like(depth_map, dtype=np.float32)
    depth_map_no_nan = depth_map.copy()
    nan_replacement = np.nanmin(depth_map)
    depth_map_no_nan[~valid_mask] = nan_replacement

    # 1) Optional outlier clipping based on percentiles
    if clip_percentiles is not None:
        lo, hi = np.percentile(depth_map_no_nan, clip_percentiles)
        depth_map_no_nan = np.clip(depth_map_no_nan, lo, hi)

    # 2) Simple min-max normalization
    d_min = depth_map_no_nan.min()
    d_max = depth_map_no_nan.max()
    rng = d_max - d_min
    if rng < eps:
        # If everything is (nearly) the same, return zeros
        return np.zeros_like(depth_map_no_nan, dtype=np.float32)
    norm = (depth_map_no_nan - d_min) / rng

    # 3) Invert so that smaller depths (i.e. near camera) => bigger values
    #    (You can remove this step if you prefer smaller => smaller.)
    norm = 1.0 - norm

    # 4) Gamma correction (helps expand differences near 1 if gamma < 1)
    if gamma != 1.0:
        # Make sure all values are > 0 before exponentiation
        norm = np.clip(norm, 0, 1)  # avoid negative or above 1 from rounding errors
        norm = norm**gamma

    # Return the final normalized map
    return norm.astype(np.float32)


def _depth_log_normalization(depth_map: np.ndarray, eps=1e-6):
    """
    Normalize the inverse depths.
    """
    # Shift so all values are >= eps
    d_min = np.nanmin(depth_map)
    shifted = depth_map - d_min + eps  # shift & ensure strictly positive

    # Log-compress to emphasize small (near) values
    log_shifted = np.log(shifted)

    # Rescale to [0, 1]
    log_min, log_max = np.nanmin(log_shifted), np.nanmax(log_shifted)
    # If all values are the same, log_max == log_min. Handle that:
    if np.isclose(log_min, log_max):
        normalized = np.zeros_like(log_shifted, dtype=np.float32)
    else:
        normalized = (log_shifted - log_min) / (log_max - log_min)

    # Invert
    normalized = 1.0 - normalized

    return normalized
