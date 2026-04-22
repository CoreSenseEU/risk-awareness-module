"""
riskam.score

Risk awareness score computation.

The former module-level ``prev_risks`` global is replaced by the ``RiskScorer``
class, which holds per-track history and is instantiated once by the ROS node.
This makes the scorer thread-safe to instantiate separately, testable, and safe
to use in multi-context pipelines.
"""

import math
from collections import deque

import numpy as np

# ── Default weights ───────────────────────────────────────────────────────────

W_PROXIMITY_EMPIRICAL_DEFAULT = 0.7
W_GAZE_EMPIRICAL_DEFAULT = 0.25
W_POSITION_EMPIRICAL_DEFAULT = 0.05
# Default 0.0 so wiring the T2.2 approach sub-score into the formula does not
# silently shift live behaviour. Non-zero values are exercised by the T3.3
# evaluation sweep; the final calibrated default will come out of T3.3.11.
W_APPROACH_EMPIRICAL_DEFAULT = 0.0

# ── Temporal aggregation ──────────────────────────────────────────────────────

N_FRAMES_AGGREGATE = 5

# ── Multi-person crowd penalty  (§3.3 of the improvement plan) ───────────────
# scene_risk = max(per_person) * (1 + CROWD_ALPHA * log(1 + n_extra_persons))

CROWD_ALPHA_DEFAULT = 0.1

# ── Visualisation / downstream thresholds ────────────────────────────────────

RISK_SCORE_BREAKPOINTS = [0, 0.3, 0.6, 1]
VERY_SMALL_RISK_VALUE = 0.00001


class RiskScorer:
    """Stateful risk scorer that maintains per-track temporal history.

    Parameters
    ----------
    n_frames : int
        Number of frames to keep in each track's history window.
    crowd_alpha : float
        Crowd-penalty coefficient; set to 0 to disable.
    """

    def __init__(
        self,
        n_frames: int = N_FRAMES_AGGREGATE,
        crowd_alpha: float = CROWD_ALPHA_DEFAULT,
    ) -> None:
        self._n_frames = n_frames
        self._crowd_alpha = crowd_alpha
        # track_id (or None for untracked) → deque of per-frame risk values
        self._history: dict[int | None, deque] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        features: dict[str, np.ndarray] | None,
        track_ids: list[int | None] | None = None,
        w_proximity: float = W_PROXIMITY_EMPIRICAL_DEFAULT,
        w_gaze: float = W_GAZE_EMPIRICAL_DEFAULT,
        w_position: float = W_POSITION_EMPIRICAL_DEFAULT,
        w_approach: float = W_APPROACH_EMPIRICAL_DEFAULT,
    ) -> tuple[float, int, dict[int | None, float]]:
        """Compute the scene risk score.

        The ``proximity`` feature is now the calibrated absolute-depth proximity
        score (1 = person at camera, 0 = person at or beyond d_safe), so the
        risk contribution is taken *directly* (not ``1 - proximity`` as in the
        prototype, which used MiDaS distance).  Gaze is still inverted because
        a high gaze score means the person *is* aware, which *lowers* risk.
        The ``approach`` sub-score is in [0, 1] with 0.5 = stationary/unknown,
        >0.5 = approaching, <0.5 = moving away — it is taken directly as a
        risk contribution (stationary persons therefore add a constant baseline
        of ``0.5 * w_approach``).

        Parameters
        ----------
        features : dict or None
            Keys: ``"proximity"``, ``"gaze"``, ``"x_offset"``, ``"approach"``,
            each a 1-D numpy array with one value per detected person.
            ``None`` if no humans are detected. ``"approach"`` is optional for
            backwards compatibility; if missing, it is treated as neutral 0.5.
        track_ids : list[int | None] or None
            ByteTrack ID per person (None entries for untracked detections).
        w_proximity, w_gaze, w_position, w_approach : float
            Weights; should sum to 1.

        Returns
        -------
        scene_risk : float   in [VERY_SMALL_RISK_VALUE, 1]
        max_risk_idx : int   index of the highest-risk person, or -1
        per_person : dict    track_id → smoothed individual risk value
        """
        if features is None:
            return 0.0, -1, {}

        n = len(features["proximity"])
        if track_ids is None or len(track_ids) != n:
            track_ids = [None] * n

        approach = features.get(
            "approach", np.full(n, 0.5, dtype=float)
        )

        # ── Per-person instantaneous risk ─────────────────────────────────────
        instant = (
            w_proximity * features["proximity"]       # close → high risk
            + w_gaze * (1.0 - features["gaze"])       # unaware → high risk
            + w_position * features["x_offset"]       # in-path → high risk
            + w_approach * approach                   # approaching → high risk
        )

        # ── Temporal smoothing per track ──────────────────────────────────────
        smoothed = np.zeros(n, dtype=float)
        for i, tid in enumerate(track_ids):
            key = tid  # None is a valid dict key (untracked bucket)
            if key not in self._history:
                self._history[key] = deque(maxlen=self._n_frames)
            self._history[key].append(float(instant[i]))
            smoothed[i] = float(np.mean(self._history[key]))

        # Prune stale untracked bucket so it doesn't grow forever.
        if None in self._history and len(self._history[None]) == self._n_frames:
            # Keep the deque; it auto-evicts old values via maxlen.
            pass

        # ── Scene aggregation with crowd penalty ──────────────────────────────
        max_risk_idx = int(np.argmax(smoothed))
        max_risk = float(smoothed[max_risk_idx])
        n_extra = max(0, n - 1)
        crowd_factor = 1.0 + self._crowd_alpha * math.log(1.0 + n_extra)
        scene_risk = float(np.clip(max_risk * crowd_factor, VERY_SMALL_RISK_VALUE, 1.0))

        per_person = {tid: float(smoothed[i]) for i, tid in enumerate(track_ids)}

        return scene_risk, max_risk_idx, per_person

    def reset(self) -> None:
        """Clear all temporal history (e.g. on node restart)."""
        self._history.clear()
