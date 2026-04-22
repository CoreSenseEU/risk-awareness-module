"""
ml.humandet

Human detection, tracking, and per-person feature extraction for risk awareness.

Changes from the prototype:
  - BboxTracker replaced by Ultralytics ByteTrack (model.track with persist=True).
    ByteTrack uses motion-aware matching and handles the 3–7 Hz frame rates seen
    in live deployment.
  - Gaze estimation upgraded from a 1-D horizontal-symmetry heuristic (which
    incorrectly scored people looking up/down as "aware") to a 2-D head-pose
    score using yaw and pitch derived from YOLO11-Pose keypoints.
  - Per-track velocity estimation: centroid depth is tracked over a short window
    to detect people approaching vs. moving away.
"""

import math
from collections import deque
from time import monotonic

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from riskam.data.paths import ML_MODELS_DIR

# pylint: disable=no-member

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_POSE_MODEL_PATH = ML_MODELS_DIR / "yolo11n-pose.pt"
model = YOLO(YOLO_POSE_MODEL_PATH, verbose=False)

# ── Gaze head-pose defaults ──────────────────────────────────────────────────

# Gaussian σ for yaw (normalised by inter-eye distance).
# yaw_offset=SIGMA_YAW → gaze score drops to ~0.6.
SIGMA_YAW_DEFAULT = 0.3

# Gaussian σ for pitch deviation from the expected frontal ratio.
SIGMA_PITCH_DEFAULT = 0.5

# Expected (nose.y − eye_midpoint.y) / inter_eye_dist when facing the camera.
# Empirically ~0.7 for a typical frontal view at neutral pitch.
FRONTAL_PITCH_RATIO_DEFAULT = 0.7

# ── Velocity tracking ────────────────────────────────────────────────────────

# Number of frames to keep per track for velocity estimation.
VELOCITY_WINDOW = 8

# Approach velocity at which the approach score saturates at 1 (m/s).
MAX_APPROACH_VEL_MS = 1.0

# Module-level velocity history: track_id → deque of (timestamp_s, depth_m).
_velocity_history: dict[int, deque] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def detect_humans(
    image: np.ndarray,
    track_bboxes: bool = True,
) -> tuple[list, list[float], np.ndarray | None, list[int | None]]:
    """Detect humans and return bounding boxes, x-offset scores, keypoints, and track IDs.

    Parameters
    ----------
    image : np.ndarray  (BGR, as returned by cv_bridge)
    track_bboxes : bool
        When True, run ByteTrack so that detections carry persistent IDs.

    Returns
    -------
    human_bboxes : list of [x1, y1, x2, y2]
    bbox_offset_scores : list[float]
        Per-person x-offset scores in [0, 1] (image-centre proximity).
    keypoints_np : np.ndarray (N, 17, 2) or None
    track_ids : list[int | None]
        ByteTrack ID per person, or None if tracking is disabled / unavailable.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if track_bboxes:
        results = model.track(image_rgb, verbose=False, persist=True)
    else:
        results = model(image_rgb, verbose=False)

    human_bboxes: list = []
    keypoints_np: np.ndarray | None = None
    track_ids: list[int | None] = []

    for result in results:
        if result.boxes is None or len(result.boxes.xyxy) == 0:
            continue

        human_bboxes = result.boxes.xyxy.cpu().tolist()
        keypoints_np = result.keypoints.data[..., :2].cpu().numpy()

        if track_bboxes and result.boxes.id is not None:
            track_ids = [int(tid) for tid in result.boxes.id.cpu().tolist()]
        else:
            track_ids = [None] * len(human_bboxes)

    image_h, image_w = image.shape[:2]
    bbox_offset_scores = [_bbox_offset_score(b, image_w) for b in human_bboxes]

    return human_bboxes, bbox_offset_scores, keypoints_np, track_ids


def gaze_scores(
    keypoints_np: np.ndarray | None,
    sigma_yaw: float = SIGMA_YAW_DEFAULT,
    sigma_pitch: float = SIGMA_PITCH_DEFAULT,
    frontal_pitch_ratio: float = FRONTAL_PITCH_RATIO_DEFAULT,
) -> list[float]:
    """Compute 2-D head-pose gaze scores for each detected person.

    Returns a list of floats in [0, 1]:
      1 = person is facing the camera directly (both yaw and pitch near zero)
      0 = person is turned away or pitching strongly up/down
    """
    if keypoints_np is None or len(keypoints_np) == 0:
        return []
    return [
        _headpose_gaze(keypoints_np[i], sigma_yaw, sigma_pitch, frontal_pitch_ratio)
        for i in range(len(keypoints_np))
    ]


def update_velocity(
    track_ids: list[int | None],
    depths_m: list[float],
) -> None:
    """Record the current depth observation for each tracked person.

    Call this once per frame after obtaining per-bbox depths.
    """
    now = monotonic()
    active = set()
    for tid, d in zip(track_ids, depths_m):
        if tid is None:
            continue
        if tid not in _velocity_history:
            _velocity_history[tid] = deque(maxlen=VELOCITY_WINDOW)
        _velocity_history[tid].append((now, d))
        active.add(tid)

    # Prune IDs that have not been seen for a while (not in current frame).
    stale = [k for k in _velocity_history if k not in active]
    for k in stale:
        del _velocity_history[k]


def approach_scores(track_ids: list[int | None]) -> list[float]:
    """Return a per-person approach score in [0, 1].

    0.5  → stationary or unknown
    > 0.5 → approaching (higher = faster approach)
    < 0.5 → moving away
    """
    return [_approach_score(tid) for tid in track_ids]


def reset_velocity_history() -> None:
    """Clear the module-level per-track depth history.

    Call this between logically distinct sessions (e.g. different offline runs)
    so that depth observations from a prior session do not leak into velocity
    estimates for a new one.
    """
    _velocity_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _headpose_gaze(
    kpts: np.ndarray,
    sigma_yaw: float,
    sigma_pitch: float,
    frontal_pitch_ratio: float,
) -> float:
    """2-D head-pose gaze score for a single person.

    COCO keypoint indices used:
      0  nose
      1  left_eye  (person's left → appears on the RIGHT in the image)
      2  right_eye (person's right → appears on the LEFT in the image)

    Yaw  (horizontal head turn):
      nose is displaced horizontally from the eye midpoint when the head turns.
      Normalised by inter-eye distance → Gaussian around 0.

    Pitch (up/down tilt):
      Ratio of (nose.y − eye_midpoint.y) / inter_eye_dist.
      Positive when nose is below eye level (normal for a frontal view).
      Deviations from frontal_pitch_ratio → Gaussian penalty.
    """
    if kpts.shape[0] < 3:
        return 0.0

    nose = kpts[0]
    left_eye = kpts[1]
    right_eye = kpts[2]

    # Treat all-zero keypoints as undetected.
    if np.all(nose == 0) or np.all(left_eye == 0) or np.all(right_eye == 0):
        return 0.0

    eye_mid = (left_eye + right_eye) / 2.0
    inter_eye_dist = np.linalg.norm(right_eye - left_eye)

    if inter_eye_dist < 1.0:  # degenerate / unreliable
        return 0.0

    yaw_offset = (nose[0] - eye_mid[0]) / inter_eye_dist
    pitch_ratio = (nose[1] - eye_mid[1]) / inter_eye_dist
    pitch_deviation = pitch_ratio - frontal_pitch_ratio

    yaw_score = math.exp(-(yaw_offset**2) / (2.0 * sigma_yaw**2))
    pitch_score = math.exp(-(pitch_deviation**2) / (2.0 * sigma_pitch**2))

    return float(yaw_score * pitch_score)


def _approach_score(track_id: int | None) -> float:
    """Estimate approach score for one track from its depth history.

    Returns 0.5 when there is insufficient history.
    """
    if track_id is None:
        return 0.5

    hist = _velocity_history.get(track_id)
    if hist is None or len(hist) < 2:
        return 0.5

    times = np.array([t for t, _ in hist])
    depths = np.array([d for _, d in hist])
    dt = times[-1] - times[0]
    if dt < 0.05:  # too short a window
        return 0.5

    # Fit linear slope: negative slope = depth decreasing = person approaching.
    slope = float(np.polyfit(times - times[0], depths, 1)[0])

    # slope < 0 → approaching → high score
    # Clamp velocity to ±MAX_APPROACH_VEL_MS, then map to [0, 1].
    clamped = max(-MAX_APPROACH_VEL_MS, min(MAX_APPROACH_VEL_MS, slope))
    return float(0.5 - clamped / (2.0 * MAX_APPROACH_VEL_MS))


def _bbox_offset_score(bbox: list, image_width: int) -> float:
    """X-offset score: 1 at image centre, 0 at the edges."""
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2.0
    img_cx = image_width / 2.0
    offset = abs(center_x - img_cx) / img_cx
    return float(1.0 - offset**2)
