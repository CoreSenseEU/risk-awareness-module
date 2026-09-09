"""
riskam.event_table

The Layer-2 backbone: one row per frame (ALL frames, labelled or not) with
causal score channels and non-causal hindsight-oracle outcomes. Layers 2–4
of the evaluation are queries over this table.

Sibling of :mod:`riskam.scene_table` (the legacy, GT-gated table behind
``scripts/metric_lab.py``) — the causal walk here deliberately mirrors
``build_scene_table`` row-for-row; that module stays frozen. If you fix a
bug in one walk, check the other.

Circularity firewall: the causal pass (KinematicTracker + RiskScorer +
SSM margins) sees only the past; the oracle block is produced by
:mod:`riskam.hindsight`, which sees only cached detections and the future.
The two meet only in the joined row — no state flows between them.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from riskam.hindsight import (
    OracleParams,
    decode_frame,
    hindsight_columns,
    r_tag,
    smoothed_scene_min_d,
    t_tag,
)
from riskam.kinematics import (
    KinematicParams,
    KinematicStatus,
    KinematicTracker,
    SceneRiskSmoother,
)
from riskam.ssm import SSMParams, ssm_margins


def oracle_columns(params: OracleParams = OracleParams()) -> list[str]:
    """The oracle block's column names, in stable order."""
    cols = ["d_oracle_now_m", "deadzone_any"]
    for T in params.t_grid_s:
        cols += [f"future_min_d_{t_tag(T)}", f"oracle_cov_{t_tag(T)}"]
    for r in params.r_grid_m:
        cols += [f"in_event_{r_tag(r)}", f"t_to_onset_{r_tag(r)}"]
        cols += [f"event_{r_tag(r)}_{t_tag(T)}" for T in params.t_grid_s]
    return cols


def event_columns(params: OracleParams = OracleParams()) -> list[str]:
    return [
        "run", "frame", "t_s",
        "gt", "bucket", "is_labelled",
        "n_humans",
        "m0_risk", "m0_class",
        "proximity", "gaze", "x_offset", "approach",
        "gaze_measured", "n_faces_measured",
        "d_m", "bearing_rad", "closing_ms", "tan_speed_ms",
        "t_cpa_s", "d_min_m", "inv_ttc",
        "hazard", "awareness", "risk_a", "kin_status",
        "nearest_d_m",
        "ssm_margin_worst", "ssm_margin_aware",
        "v_robot_speed_ms", "ego_available",
    ] + oracle_columns(params)


_STR_COLUMNS = {"run", "frame", "bucket", "kin_status"}
_INT_COLUMNS = {"gt", "is_labelled", "n_humans", "m0_class", "ego_available",
                "deadzone_any", "gaze_measured", "n_faces_measured"}


def _person_planar_d(prim, camera, d_safe_m: float) -> np.ndarray:
    """Per-person planar range for SSM: dead zone → 0, far sentinel → d_safe."""
    n = len(prim.human_bboxes)
    d = np.empty(n)
    for i in range(n):
        depth = float(prim.bbox_depths_m[i])
        if depth == 0.0:
            d[i] = 0.0
        elif depth == d_safe_m:
            d[i] = d_safe_m  # unmeasurable-far: cannot alarm
        else:
            x1, _, x2, _ = prim.human_bboxes[i]
            u = 0.5 * (float(x1) + float(x2))
            d[i] = math.hypot(depth, -depth * (u - camera.cx_px) / camera.fx_px)
    return d


def build_event_table(
    *,
    cache,
    frames_per_run: dict,
    kin_params: KinematicParams,
    camera_by_run: dict,
    odom_by_run: dict | None = None,
    gt_per_run: dict | None = None,
    split_obj=None,
    oracle_params: OracleParams = OracleParams(),
    ssm_params: SSMParams = SSMParams(),
) -> list[dict]:
    """Emit one row per cached frame of every run (labelled or not).

    ``frames_per_run``: {run: [Path, ...]} ALL RGB frame paths sorted by
    stem timestamp. ``gt_per_run``/``split_obj`` fill the nullable
    ``gt``/``bucket`` columns when the dataset has annotations (2023);
    otherwise ``gt`` is -1 and ``bucket`` empty. Frames without a cache
    entry are skipped (and counted by the caller via row totals).
    """
    from riskam.eval_metrics import predicted_class
    from riskam.ml import featextr, humandet
    from riskam.score import RiskScorer

    odom_by_run = odom_by_run or {}
    gt_per_run = gt_per_run or {}
    rows: list[dict] = []

    for run in sorted(frames_per_run):
        camera = camera_by_run[run]
        gt = gt_per_run.get(run, {})
        scorer = RiskScorer()
        tracker = KinematicTracker(kin_params, camera)
        scene_smoother = SceneRiskSmoother(
            kin_params.release_tau_s, kin_params.hold_max_s
        )
        humandet.reset_velocity_history()
        odom = odom_by_run.get(run)

        run_rows: list[dict] = []
        observations = []  # aligned with run_rows, for the oracle pass

        # ── causal pass ──────────────────────────────────────────────────
        for img_path in frames_per_run[run]:
            prim = cache.get(run, img_path.stem)
            if prim is None:
                continue  # not populated
            t_s = float(img_path.stem)
            ego = odom.twist_for(t_s) if odom else None
            gt_class = int(gt[img_path.name]) if img_path.name in gt else -1
            bucket = _bucket(split_obj, run, img_path.name)

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
                row = {
                    "run": run, "frame": img_path.name, "t_s": t_s,
                    "gt": gt_class, "bucket": bucket,
                    "is_labelled": int(img_path.name in gt),
                    "n_humans": 0,
                    "m0_risk": m0_risk, "m0_class": m0_class,
                    "proximity": 0.0, "gaze": 1.0, "x_offset": 0.0,
                    "approach": 0.5,
                    "gaze_measured": -1, "n_faces_measured": 0,
                    "d_m": 2.0 * kin_params.d_safe_m, "bearing_rad": 0.0,
                    "closing_ms": 0.0, "tan_speed_ms": 0.0,
                    "t_cpa_s": 0.0, "d_min_m": 2.0 * kin_params.d_safe_m,
                    "inv_ttc": 0.0,
                    "hazard": 0.0, "awareness": 1.0,
                    "risk_a": scene_smoother.update(t_s, 0.0),
                    "kin_status": "no_human",
                    "nearest_d_m": math.inf,
                    "ssm_margin_worst": math.inf, "ssm_margin_aware": math.inf,
                    "v_robot_speed_ms": ego.speed if ego is not None else None,
                    "ego_available": int(ego is not None),
                }
            else:
                kins = tracker.update(
                    t_s, prim.human_bboxes, prim.bbox_depths_m,
                    prim.track_ids, ego=ego,
                    awareness=extraction.features["gaze"],
                )
                fused = [k.risk for k in kins]
                a_idx = int(np.argmax(fused))
                k = kins[a_idx]
                scene_risk = scene_smoother.update(t_s, fused[a_idx])
                r = math.hypot(k.x_m, k.y_m)
                if k.status is KinematicStatus.DEAD_ZONE:
                    inv_ttc = 1.0
                else:
                    inv_ttc = min(k.closing_ms / max(r, 1e-6), 1.0)

                person_d = _person_planar_d(prim, camera, kin_params.d_safe_m)
                v_r = ego.speed if ego is not None else 0.0
                margin_worst, margin_aware = ssm_margins(
                    person_d, np.asarray(extraction.features["gaze"], dtype=float),
                    v_r, ssm_params,
                )

                row = {
                    "run": run, "frame": img_path.name, "t_s": t_s,
                    "gt": gt_class, "bucket": bucket,
                    "is_labelled": int(img_path.name in gt),
                    "n_humans": len(kins),
                    "m0_risk": m0_risk, "m0_class": m0_class,
                    "proximity": float(extraction.features["proximity"][m0_idx]),
                    "gaze": float(extraction.features["gaze"][m0_idx]),
                    "x_offset": float(extraction.features["x_offset"][m0_idx]),
                    "approach": float(extraction.features["approach"][m0_idx]),
                    "gaze_measured": (
                        int(bool(extraction.gaze_measurable[m0_idx]))
                        if extraction.gaze_measurable is not None
                        else -1
                    ),
                    "n_faces_measured": (
                        int(extraction.gaze_measurable.sum())
                        if extraction.gaze_measurable is not None
                        else -1
                    ),
                    "d_m": k.d_m, "bearing_rad": k.bearing_rad,
                    "closing_ms": k.closing_ms, "tan_speed_ms": k.tan_speed_ms,
                    "t_cpa_s": k.t_cpa_s, "d_min_m": k.d_min_m,
                    "inv_ttc": inv_ttc,
                    "hazard": k.hazard, "awareness": float(k.awareness),
                    "risk_a": scene_risk, "kin_status": k.status.value,
                    "nearest_d_m": float(np.min(person_d)),
                    "ssm_margin_worst": margin_worst,
                    "ssm_margin_aware": margin_aware,
                    "v_robot_speed_ms": ego.speed if ego is not None else None,
                    "ego_available": int(ego is not None),
                }

            run_rows.append(row)
            observations.append(
                decode_frame(prim, camera, kin_params.d_safe_m, t_s=t_s)
            )

        # ── oracle pass (non-causal; riskam.hindsight only) ──────────────
        if not run_rows:
            continue
        t = np.array([r["t_s"] for r in run_rows])
        min_d = smoothed_scene_min_d(observations, oracle_params)
        oracle = hindsight_columns(t, min_d, oracle_params)
        deadzone_any = np.array(
            [int(bool(o.deadzone.any())) for o in observations], dtype=np.uint8
        )

        assert len(t) == len(run_rows) == len(observations)
        for i, row in enumerate(run_rows):
            row["d_oracle_now_m"] = float(min_d[i])
            row["deadzone_any"] = int(deadzone_any[i])
            for name, values in oracle.items():
                row[name] = values[i]
        rows.extend(run_rows)

    return rows


def _bucket(split_obj, run: str, frame_name: str) -> str:
    if split_obj is None:
        return ""
    return split_obj.assignments.get(run, {}).get(frame_name, "")


def write_event_table_csv(
    rows: list[dict], path: Path, params: OracleParams = OracleParams()
) -> None:
    """Write rows to CSV with a stable column order."""
    columns = event_columns(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def read_event_table_csv(
    path: Path, params: OracleParams = OracleParams()
) -> list[dict]:
    """Read the CSV back with numeric types restored."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for key, value in raw.items():
                if key in _STR_COLUMNS:
                    row[key] = value
                elif value == "" or value == "None":
                    row[key] = None
                elif key in _INT_COLUMNS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows
