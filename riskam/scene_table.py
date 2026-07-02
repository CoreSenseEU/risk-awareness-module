"""
riskam.scene_table

Builds the per-frame, scene-level feature table that the experimental
metric comparison (``scripts/metric_lab.py``) fits and evaluates on.

One row per labelled frame, walking each run's frames in stem-timestamp
order with a fresh :class:`~riskam.score.RiskScorer` (the M0 baseline) and
a fresh :class:`~riskam.kinematics.KinematicTracker` (Direction A) per run,
reading only cached primitives — no GPU work.

Scene aggregation: the ground truth is a scene-level worst-case label, and
the deployed scorer also reports its worst actor, so Direction A's scene
row is its **max-fused-risk person** (``argmax fuse_risk(hazard, awareness)``).
The closest-person rule used by the exploratory probe mis-ranks a fast
closer behind a static bystander — exactly the case the kinematic metric
exists to catch. M0's sub-score columns describe M0's *own* max-risk
person, so each metric is reported on its own worst actor.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from riskam.kinematics import (
    CameraModel,
    KinematicParams,
    KinematicStatus,
    KinematicTracker,
    PlanarTwist,
    SceneRiskSmoother,
)

SCENE_COLUMNS = [
    "run", "frame", "t_s", "bucket", "gt", "n_humans",
    "m0_risk", "m0_class",
    "proximity", "gaze", "x_offset", "approach",
    "d_m", "bearing_rad", "closing_ms", "tan_speed_ms",
    "t_cpa_s", "d_min_m", "inv_ttc",
    "hazard", "awareness", "risk_a", "kin_status",
    "v_robot_speed_ms", "ego_available",
]

_STR_COLUMNS = {"run", "frame", "bucket", "kin_status"}
_INT_COLUMNS = {"gt", "n_humans", "m0_class", "ego_available"}


def _sentinel_row(run: str, frame: str, t_s: float, bucket: str, gt: int,
                  m0_risk: float, m0_class: int, d_safe_m: float,
                  ego: PlanarTwist | None) -> dict:
    """No-human frame: minimum hazard, finite features (probe convention)."""
    return {
        "run": run, "frame": frame, "t_s": t_s, "bucket": bucket, "gt": gt,
        "n_humans": 0, "m0_risk": m0_risk, "m0_class": m0_class,
        "proximity": 0.0, "gaze": 1.0, "x_offset": 0.0, "approach": 0.5,
        "d_m": 2.0 * d_safe_m, "bearing_rad": 0.0,
        "closing_ms": 0.0, "tan_speed_ms": 0.0,
        "t_cpa_s": 0.0, "d_min_m": 2.0 * d_safe_m, "inv_ttc": 0.0,
        "hazard": 0.0, "awareness": 1.0, "risk_a": 0.0,
        "kin_status": "no_human",
        "v_robot_speed_ms": ego.speed if ego is not None else None,
        "ego_available": int(ego is not None),
    }


def build_scene_table(
    *,
    cache,
    gt_per_run: dict,
    frames_per_run: dict,
    split_obj=None,
    kin_params: KinematicParams,
    camera_by_run: dict,
    odom_by_run: dict | None = None,
) -> list[dict]:
    """Emit one feature row per cached labelled frame.

    Parameters
    ----------
    cache : riskam.feature_cache.FeatureCache
    gt_per_run : {run: {frame_name: gt_class}}
    frames_per_run : {run: [Path, ...]} labelled RGB frame paths, sorted by
        stem timestamp (the caller owns dataset layout).
    split_obj : riskam.data.splits.Split or None — fills the ``bucket``
        column ("" when None).
    kin_params : KinematicParams for the Direction-A tracker.
    camera_by_run : {run: CameraModel}.
    odom_by_run : {run: odom index or None}; an index must expose
        ``twist_for(t_s) -> PlanarTwist | None``.
    """
    from riskam.eval_metrics import predicted_class
    from riskam.ml import featextr, humandet
    from riskam.score import RiskScorer

    odom_by_run = odom_by_run or {}
    rows: list[dict] = []

    for run in sorted(gt_per_run):
        gt = gt_per_run[run]
        scorer = RiskScorer()
        tracker = KinematicTracker(kin_params, camera_by_run[run])
        scene_smoother = SceneRiskSmoother(
            kin_params.release_tau_s, kin_params.hold_max_s
        )
        # The M0 approach sub-score keeps its deployed (wall-clock,
        # module-global) history; reset per run like the harness does.
        humandet.reset_velocity_history()
        odom = odom_by_run.get(run)

        for img_path in frames_per_run.get(run, []):
            prim = cache.get(run, img_path.stem)
            if prim is None:
                continue  # not populated
            t_s = float(img_path.stem)
            gt_class = int(gt[img_path.name])
            ego = odom.twist_for(t_s) if odom else None

            h, w = prim.depth_viz.shape[:2]
            extraction = featextr.extract_features(
                prim, image_shape=(h, w), cmd_vel=None,
                d_safe=kin_params.d_safe_m,
            )
            m0_risk, m0_idx, _ = scorer.score(
                extraction.features, track_ids=extraction.track_ids
            )
            m0_class = predicted_class(m0_risk)

            if extraction.features is None:
                tracker.update(
                    t_s, prim.human_bboxes, prim.bbox_depths_m,
                    prim.track_ids, ego=ego,
                )
                row = _sentinel_row(
                    run, img_path.name, t_s,
                    _bucket(split_obj, run, img_path.name), gt_class,
                    m0_risk, m0_class, kin_params.d_safe_m, ego,
                )
                # Scene release bridges detector dropouts: an empty frame
                # right after a risky one keeps a decaying scene risk.
                row["risk_a"] = scene_smoother.update(t_s, 0.0)
                rows.append(row)
                continue

            kins = tracker.update(
                t_s, prim.human_bboxes, prim.bbox_depths_m,
                prim.track_ids, ego=ego,
                awareness=extraction.features["gaze"],
            )
            fused = [k.risk for k in kins]
            a_idx = int(np.argmax(fused))
            k = kins[a_idx]
            awareness = float(k.awareness)
            scene_risk = scene_smoother.update(t_s, fused[a_idx])
            r = math.hypot(k.x_m, k.y_m)
            if k.status is KinematicStatus.DEAD_ZONE:
                inv_ttc = 1.0  # person inside the dead zone: max hazard rate
            else:
                inv_ttc = min(k.closing_ms / max(r, 1e-6), 1.0)

            rows.append({
                "run": run, "frame": img_path.name, "t_s": t_s,
                "bucket": _bucket(split_obj, run, img_path.name),
                "gt": gt_class, "n_humans": len(kins),
                "m0_risk": m0_risk, "m0_class": m0_class,
                "proximity": float(extraction.features["proximity"][m0_idx]),
                "gaze": float(extraction.features["gaze"][m0_idx]),
                "x_offset": float(extraction.features["x_offset"][m0_idx]),
                "approach": float(extraction.features["approach"][m0_idx]),
                "d_m": k.d_m, "bearing_rad": k.bearing_rad,
                "closing_ms": k.closing_ms, "tan_speed_ms": k.tan_speed_ms,
                "t_cpa_s": k.t_cpa_s, "d_min_m": k.d_min_m, "inv_ttc": inv_ttc,
                "hazard": k.hazard, "awareness": awareness,
                "risk_a": scene_risk, "kin_status": k.status.value,
                "v_robot_speed_ms": ego.speed if ego is not None else None,
                "ego_available": int(ego is not None),
            })

    return rows


def _bucket(split_obj, run: str, frame_name: str) -> str:
    if split_obj is None:
        return ""
    return split_obj.bucket(run, frame_name) or ""


def write_table_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCENE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {c: ("" if row.get(c) is None else row[c]) for c in SCENE_COLUMNS}
            )


def read_table_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for col, val in raw.items():
                if col in _STR_COLUMNS:
                    row[col] = val
                elif val == "":
                    row[col] = None
                elif col in _INT_COLUMNS:
                    row[col] = int(val)
                else:
                    row[col] = float(val)
            rows.append(row)
    return rows
