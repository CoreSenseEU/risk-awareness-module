"""
ml.subscores

Sub-score input-availability contract.

Each risk sub-score declares the inputs it needs and the fallback policy for
when an *optional* input is missing. *Required* inputs (RGB image + absolute
depth) are checked at the pipeline boundary and are never silently dropped;
if they are absent, frame processing fails loudly so downstream consumers
cannot mistake a degraded frame for a calibrated one.

Status semantics
----------------
ACTIVE        the sub-score was computed from its live inputs
FALLBACK      an optional input was missing; a declared fallback produced a value
UNAVAILABLE   inputs missing and no fallback; the sub-score does not contribute

Scoring policy
--------------
The scoring formula multiplies each sub-score value by its configured weight
regardless of status. Status flows out via ``FrameExtraction.subscore_status``
so the ROS diagnostic topic (and offline experiments) can report which
sub-scores were degraded on a given frame. Silent weight redistribution is
deliberately avoided — it would change the scoring semantics without telling
the caller.

Required vs optional inputs
---------------------------
Required (hard floor — failure is a runtime error):
    rgb, depth_m
Optional (missing → per-sub-score fallback or UNAVAILABLE):
    cmd_vel           (enables path-aware x_offset; FALLBACK = centre-offset)
    track continuity  (enables approach sub-score via ByteTrack IDs;
                       UNAVAILABLE when no track has enough depth history)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

from riskam.ml import depth as depth_mod
from riskam.ml import humandet


# ── Status enum ──────────────────────────────────────────────────────────────


class SubScoreStatus(str, Enum):
    ACTIVE = "active"
    FALLBACK = "fallback"
    UNAVAILABLE = "unavailable"


# ── Framework-neutral value types ────────────────────────────────────────────


@dataclass(frozen=True)
class RobotVelocity:
    """Instantaneous robot velocity in its base frame (ROS REP-103).

    This type is intentionally framework-neutral so the subscores module does
    not depend on ROS. Construct from a geometry_msgs/Twist via
    :meth:`from_twist` at the ROS boundary.
    """

    linear_x: float  # forward (m/s)
    linear_y: float  # left-positive (m/s)

    @property
    def speed(self) -> float:
        return math.hypot(self.linear_x, self.linear_y)

    @classmethod
    def from_twist(cls, twist) -> "RobotVelocity":
        return cls(float(twist.linear.x), float(twist.linear.y))


@dataclass
class FrameInputs:
    """External inputs for one frame of risk awareness processing.

    Required: ``rgb`` and ``depth_m`` must always be present — missing either
    raises ``ValueError`` at :meth:`validate`. Optional inputs are ``None``
    when their source topic is not available.
    """

    rgb: np.ndarray            # BGR, (H, W, 3) uint8
    depth_m: np.ndarray        # absolute depth, (H, W) float32 metres
    cmd_vel: RobotVelocity | None = None

    def validate(self) -> None:
        if self.rgb is None:
            raise ValueError("FrameInputs.rgb is required")
        if self.depth_m is None:
            raise ValueError(
                "FrameInputs.depth_m is required — RiskAM's supported "
                "minimum is RGB + absolute depth."
            )
        if self.rgb.ndim != 3:
            raise ValueError(f"FrameInputs.rgb must be 3-D, got shape {self.rgb.shape}")
        if self.depth_m.ndim != 2:
            raise ValueError(
                f"FrameInputs.depth_m must be 2-D, got shape {self.depth_m.shape}"
            )


@dataclass
class SubScoreResult:
    """One sub-score's output for a single frame."""

    values: np.ndarray                      # (n_persons,) float in [0, 1]
    status: SubScoreStatus
    reason: str = ""                        # short diagnostic text


@dataclass
class FrameExtraction:
    """Complete per-frame output of the feature-extraction pipeline."""

    human_bboxes: list
    depth_viz: np.ndarray | None
    features: dict | None                   # {name: np.ndarray} or None if no humans
    subscore_status: dict                   # {name: SubScoreStatus}
    subscore_reasons: dict = field(default_factory=dict)  # {name: str}
    track_ids: list = field(default_factory=list)


# ── Per-sub-score computation ────────────────────────────────────────────────


def compute_proximity(
    bboxes: list,
    depth_m: np.ndarray,
    d_safe: float = depth_mod.D_SAFE_DEFAULT,
) -> SubScoreResult:
    """Proximity from absolute depth. Required inputs — always ACTIVE here."""
    values = np.asarray(
        depth_mod.extract_bbox_proximities(depth_m, bboxes, d_safe=d_safe),
        dtype=float,
    )
    return SubScoreResult(values=values, status=SubScoreStatus.ACTIVE)


def compute_gaze(
    keypoints_np: np.ndarray | None,
    sigma_yaw: float = humandet.SIGMA_YAW_DEFAULT,
    sigma_pitch: float = humandet.SIGMA_PITCH_DEFAULT,
    frontal_pitch_ratio: float = humandet.FRONTAL_PITCH_RATIO_DEFAULT,
) -> SubScoreResult:
    """2-D head-pose gaze. Requires only RGB → always ACTIVE when persons detected."""
    values = np.asarray(
        humandet.gaze_scores(
            keypoints_np,
            sigma_yaw=sigma_yaw,
            sigma_pitch=sigma_pitch,
            frontal_pitch_ratio=frontal_pitch_ratio,
        ),
        dtype=float,
    )
    return SubScoreResult(values=values, status=SubScoreStatus.ACTIVE)


# Robot speeds below this are treated as "stationary" and trigger the x_offset
# fallback (centre-offset heuristic) rather than the path-aware projection.
CMD_VEL_STATIONARY_THRESHOLD = 0.05  # m/s


def compute_x_offset(
    bboxes: list,
    image_width: int,
    image_height: int,
    cmd_vel: RobotVelocity | None,
) -> SubScoreResult:
    """Path-aware x-offset sub-score with centre-offset fallback.

    ACTIVE:   ``cmd_vel`` supplied and speed ≥ CMD_VEL_STATIONARY_THRESHOLD —
              score is the projection of the bbox onto the robot's motion
              direction (Gaussian around the dangerous zone).
    FALLBACK: ``cmd_vel`` is None or the robot is effectively stationary —
              score is the centre-offset heuristic (1 at image centre,
              0 at the edges).
    """
    if cmd_vel is not None and cmd_vel.speed >= CMD_VEL_STATIONARY_THRESHOLD:
        values = np.asarray(
            [_path_projection_score(b, image_width, cmd_vel) for b in bboxes],
            dtype=float,
        )
        return SubScoreResult(values=values, status=SubScoreStatus.ACTIVE)

    values = np.asarray(
        [_centre_offset_score(b, image_width) for b in bboxes], dtype=float
    )
    reason = (
        "cmd_vel unavailable" if cmd_vel is None else "robot effectively stationary"
    )
    return SubScoreResult(values=values, status=SubScoreStatus.FALLBACK, reason=reason)


def compute_approach(
    track_ids: list,
    depth_was_used: bool = True,
) -> SubScoreResult:
    """Per-track approach velocity.

    Requires ByteTrack IDs (optional input) *and* a sufficient depth history
    maintained by :func:`humandet.update_velocity`. When no track has yielded
    more than one observation (new scene / tracking off), all scores are the
    neutral 0.5 and the status is UNAVAILABLE — the sub-score is computed but
    the caller should know it carries no information.
    """
    values = np.asarray(humandet.approach_scores(track_ids), dtype=float)
    has_any_track = any(tid is not None for tid in track_ids)
    if not has_any_track or not depth_was_used:
        return SubScoreResult(
            values=values,
            status=SubScoreStatus.UNAVAILABLE,
            reason="no track continuity" if not has_any_track else "no depth history",
        )

    informative = any(v != 0.5 for v in values.tolist())
    if not informative:
        return SubScoreResult(
            values=values,
            status=SubScoreStatus.UNAVAILABLE,
            reason="insufficient velocity history",
        )
    return SubScoreResult(values=values, status=SubScoreStatus.ACTIVE)


# ── Internal helpers (centre-offset + path projection) ───────────────────────


def _centre_offset_score(bbox: list, image_width: int) -> float:
    """Legacy x-offset score: 1 at image centre, 0 at the edges."""
    x1, _, x2, _ = bbox
    centre = (x1 + x2) / 2.0
    img_cx = image_width / 2.0
    offset = abs(centre - img_cx) / img_cx
    return float(1.0 - offset ** 2)


def _path_projection_score(
    bbox: list,
    image_width: int,
    cmd_vel: RobotVelocity,
) -> float:
    """Path-aware score: Gaussian around the bbox lane the robot is heading into.

    Coordinate convention (ROS REP-103):
      vx > 0 → forward  → dangerous lane is image centre (norm_x ≈ 0)
      vy > 0 → left     → robot's left appears on the LEFT of the image
                          (negative norm_x)
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    norm_x = (cx - image_width / 2.0) / (image_width / 2.0)  # −1 … +1

    # Lateral fraction of motion: −vy because +vy=left maps to negative norm_x.
    lat_fraction = -cmd_vel.linear_y / cmd_vel.speed
    delta = norm_x - lat_fraction
    score = math.exp(-(delta ** 2) / 0.5)
    return float(np.clip(score, 0.0, 1.0))
