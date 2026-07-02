#!/usr/bin/env python
"""
scripts/metric_lab.py

Offline laboratory for the risk-metric redesign of
``docs/private/paper-plan.md`` — implements and compares, WITHOUT touching
the deployed pipeline or the baseline experiments harness:

  m0       the deployed weighted-sum scorer + fixed breakpoints (baseline)
  m0c      ordinal calibration head on the m0 scalar (makes the baseline
           comparable on proper scoring rules)
  a_raw    Direction A: kinematic hazard × awareness, no fitting
  a_cal    ordinal calibration head on the Direction-A fused scalar
  b1_sub   Direction B1: proportional-odds logit on the four sub-scores
  b1_kin   B1 on the kinematic features — the A+B layered variant

Sub-commands (the one expensive step is isolated in ``populate``):

  check        no inference; label/depth/cache/odometry coverage report
  populate     THE GPU STEP: one YOLO-Pose pass over all labelled frames
               filling ``feature_cache/`` with parameter-independent
               primitives (bboxes, keypoints, track ids, per-bbox depths)
  build-table  CPU: cache → per-frame scene feature table (scene_table.csv)
  fit          fit the ordinal models on the val bucket; LR ablation tests
  evaluate     apply frozen fits to val AND (once) test; write reports,
               comparison.md and figures
  all          populate → build-table → fit → evaluate

Protocol notes: fits use the canonical stratified within-run split
(fit on val, report on test — the split module's stated purpose).
Scene labels are per-frame worst-case; Direction A represents the scene
by its max-fused-risk person, m0 by its own max-risk person.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))  # repo root (package = false)

from riskam.data.cs_robocup_2023 import (
    CSRobocup2023DepthIndex,
    CSRobocup2023OdomIndex,
    load_camera_model,
)
from riskam.data.ml_datasets import DATASETS
from riskam.data.paths import (
    CS_ROBOCUP_2023_GROUND_TRUTH_PATH,
    CS_ROBOCUP_2023_ML_RAW_DIR,
)
from riskam.data.splits import CS_ROBOCUP_2023_SPLIT_PATH, load_split
from riskam.feature_cache import FeatureCache, model_sha
from riskam.kinematics import KinematicParams
from riskam.scene_table import (
    build_scene_table,
    read_table_csv,
    write_table_csv,
)

DATASET = "cs_robocup_2023"
FEATURE_CACHE_ROOT = Path(__file__).parent.parent / "feature_cache"
OUT_DIR = Path(__file__).parent.parent / "exp_results" / DATASET / "metric_lab"
TABLE_PATH = OUT_DIR / "scene_table.csv"
META_PATH = OUT_DIR / "table_meta.json"
FITS_DIR = OUT_DIR / "fits"
FIGURES_DIR = OUT_DIR / "figures"
VIDEOS_DIR = OUT_DIR / "videos"

# Ordinal-model feature sets. b1_kin is the A-features + B-calibration
# layering; n_humans lets the fit adjudicate the crowd penalty.
MODEL_FEATURES = {
    "m0c": ["m0_risk"],
    "a_cal": ["risk_a"],
    "b1_sub": ["proximity", "gaze", "x_offset", "approach"],
    "b1_kin": [
        "d_m", "closing_ms", "tan_speed_ms", "d_min_m",
        "inv_ttc", "awareness", "n_humans",
    ],
}

# Likelihood-ratio ablations: full model → feature to drop.
LR_ABLATIONS = [
    ("b1_sub", "approach"),
    ("b1_sub", "gaze"),
    ("b1_kin", "awareness"),
    ("b1_kin", "n_humans"),
]

# Okabe–Ito subset, fixed order, validated with the dataviz six-checks
# (CVD-safe; the contrast WARN on orange/magenta is relieved by direct
# end-of-line labels in the reliability figure).
MODEL_COLORS = {
    "m0c": "#0072B2",
    "a_cal": "#E69F00",
    "b1_sub": "#009E73",
    "b1_kin": "#CC79A7",
}


# --------------------------------------------------------------------------- #
# Shared data wiring (no inference)
# --------------------------------------------------------------------------- #
def labelled_runs() -> dict[str, dict]:
    """{run: {frame_name: gt_class}} for runs that actually carry labels."""
    gt = json.loads(CS_ROBOCUP_2023_GROUND_TRUTH_PATH.read_text())
    return {run: d for run, d in gt.items() if isinstance(d, dict) and d}


def frames_in_time_order(run: str, gt_for_run: dict) -> list[Path]:
    """Labelled RGB frame paths for a run, sorted by timestamp (stem)."""
    rgb_dir = CS_ROBOCUP_2023_ML_RAW_DIR / run / "rgb"
    paths = [rgb_dir / name for name in gt_for_run if (rgb_dir / name).is_file()]
    return sorted(paths, key=lambda p: float(p.stem))


def default_kin_params() -> KinematicParams:
    platform = DATASETS[DATASET]["platform"]
    return KinematicParams(
        d_safe_m=platform.d_safe_m,
        footprint_radius_m=platform.footprint_radius_m,
    )


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    per_run = labelled_runs()
    print(f"dataset={DATASET}  cache_root={FEATURE_CACHE_ROOT}")
    header = f"{'run':8} {'labelled':>9} {'depth-ok':>9} {'cached':>8} {'odom':>6} {'caminfo':>8}"
    print(header)
    tot_lab = tot_depth = tot_cached = 0
    for run, gt in sorted(per_run.items()):
        depth_index = CSRobocup2023DepthIndex(run)
        cache_dir = FEATURE_CACHE_ROOT / DATASET / run
        cached = len(list(cache_dir.rglob("*.npz"))) if cache_dir.exists() else 0
        depth_ok = sum(
            depth_index.load_for_rgb(p) is not None
            for p in frames_in_time_order(run, gt)
        )
        odom_n = len(CSRobocup2023OdomIndex(run))
        caminfo = (CS_ROBOCUP_2023_ML_RAW_DIR / run / "camera_info.json").is_file()
        n = len(gt)
        tot_lab += n
        tot_depth += depth_ok
        tot_cached += cached
        print(
            f"{run:8} {n:9d} {depth_ok:9d} {cached:8d} "
            f"{odom_n:6d} {'yes' if caminfo else 'FOV':>8}"
        )
    print(f"{'TOTAL':8} {tot_lab:9d} {tot_depth:9d} {tot_cached:8d}")
    if tot_cached == 0:
        print("\n[next] cache empty -> run `populate` (the YOLO pass) on AC power.")
    elif tot_cached < tot_depth:
        print("\n[note] cache partially populated; `populate` will fill the rest.")
    else:
        print("\n[ok] cache covers depth-available frames -> `build-table` is ready.")
    return 0


# --------------------------------------------------------------------------- #
# populate  (THE GPU STEP)
# --------------------------------------------------------------------------- #
def cmd_populate(args: argparse.Namespace) -> int:
    # Imports that pull in YOLO live here so `check` stays inference-free.
    import cv2  # noqa: PLC0415
    from riskam.ml import featextr, humandet  # noqa: PLC0415
    from riskam.ml.subscores import FrameInputs  # noqa: PLC0415

    if not humandet.YOLO_POSE_MODEL_PATH.is_file():
        print(f"[error] YOLO model missing at {humandet.YOLO_POSE_MODEL_PATH}")
        return 1
    if args.device and args.device != "cpu":
        humandet.model.to(args.device)
        print(f"[info] YOLO model moved to {args.device}")

    cache = FeatureCache(
        FEATURE_CACHE_ROOT, model_sha(humandet.YOLO_POSE_MODEL_PATH), DATASET
    )
    platform = DATASETS[DATASET]["platform"]
    per_run = labelled_runs()
    runs = args.runs or sorted(per_run)

    # NOTE: always the FULL labelled frame sequence per run, never a split
    # subset — ByteTrack IDs are coupled to the frame sequence, and a
    # partially-populated run desynchronises cached IDs (see the
    # feature_cache module header). Retries should re-run whole runs.
    for run in runs:
        gt = per_run.get(run, {})
        depth_index = CSRobocup2023DepthIndex(run)
        humandet.reset_velocity_history()  # ByteTrack/velocity state per run
        frames = frames_in_time_order(run, gt)
        if args.limit_per_run:
            frames = frames[: args.limit_per_run]
        done = miss_depth = 0
        for img_path in frames:
            if cache.get(run, img_path.stem) is not None:
                continue  # already cached
            depth_m = depth_index.load_for_rgb(img_path)
            if depth_m is None:
                miss_depth += 1
                continue
            rgb = cv2.imread(str(img_path))
            primitives = featextr.extract_primitives(
                FrameInputs(rgb=rgb, depth_m=depth_m, cmd_vel=None),
                d_safe=platform.d_safe_m,
                depth_near_clip_m=platform.sensor.near_clip_m,
                near_clip_valid_frac_max=platform.sensor.valid_frac_max,
                track_bboxes=True,
            )
            cache.put(run, img_path.stem, primitives)
            done += 1
        print(f"{run}: cached {done} frames (skipped {miss_depth} w/o depth)")
    print(f"done. cache hits={cache.hits} misses={cache.misses}")
    return 0


# --------------------------------------------------------------------------- #
# build-table
# --------------------------------------------------------------------------- #
def cmd_build_table(args: argparse.Namespace) -> int:
    from riskam.ml import humandet  # noqa: PLC0415  (loads weights, no inference)
    from riskam.provenance import reproducibility_metadata  # noqa: PLC0415

    if TABLE_PATH.exists() and not args.force:
        print(f"[skip] {TABLE_PATH} exists; use --force to rebuild.")
        return 0

    cache = FeatureCache(
        FEATURE_CACHE_ROOT, model_sha(humandet.YOLO_POSE_MODEL_PATH), DATASET
    )
    platform = DATASETS[DATASET]["platform"]
    kin = default_kin_params()
    per_run = labelled_runs()
    split_obj = (
        load_split(CS_ROBOCUP_2023_SPLIT_PATH)
        if CS_ROBOCUP_2023_SPLIT_PATH.exists()
        else None
    )
    if split_obj is None:
        print("[warn] no split.json — bucket column will be empty; `fit` needs it.")

    frames_per_run = {run: frames_in_time_order(run, gt) for run, gt in per_run.items()}

    cameras: dict = {}
    camera_sources: dict = {}
    for run, frames in frames_per_run.items():
        # Image width for the FOV fallback comes from a cached frame.
        width = None
        for p in frames:
            prim = cache.get(run, p.stem)
            if prim is not None:
                width = int(prim.depth_viz.shape[1])
                break
        if width is None:
            continue  # nothing cached for this run yet
        cameras[run], camera_sources[run] = load_camera_model(
            run, width, platform.sensor
        )

    odoms = None
    if not args.no_odom:
        odoms = {run: (CSRobocup2023OdomIndex(run) or None) for run in cameras}

    rows = build_scene_table(
        cache=cache,
        gt_per_run={run: per_run[run] for run in cameras},
        frames_per_run={run: frames_per_run[run] for run in cameras},
        split_obj=split_obj,
        kin_params=kin,
        camera_by_run=cameras,
        odom_by_run=odoms,
    )
    if not rows:
        print("[error] no cached frames found. Run `populate` first.")
        return 1

    write_table_csv(rows, TABLE_PATH)
    status_hist: dict = {}
    for r in rows:
        status_hist[r["kin_status"]] = status_hist.get(r["kin_status"], 0) + 1
    meta = {
        "n_rows": len(rows),
        "model_sha": cache.model_sha,
        "kin_params": {
            "d_safe_m": kin.d_safe_m, "tau_s": kin.tau_s, "beta": kin.beta,
            "footprint_radius_m": kin.footprint_radius_m, "window": kin.window,
            "max_gap_s": kin.max_gap_s, "v_eps_ms": kin.v_eps_ms,
        },
        "camera_sources": camera_sources,
        "ego_available": {
            run: bool(odoms and odoms.get(run)) for run in cameras
        },
        "kin_status_histogram": status_hist,
        "bucket_counts": {
            b: sum(1 for r in rows if r["bucket"] == b)
            for b in ("val", "test", "")
        },
        "provenance": reproducibility_metadata(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[saved] {TABLE_PATH} ({len(rows)} rows)")
    print(f"        kin_status: {status_hist}")
    print(f"        cameras: {camera_sources}")
    return 0


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def _feature_matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[float(r[n]) for n in names] for r in rows])


def cmd_fit(args: argparse.Namespace) -> int:
    from riskam.ordinal import fit_ordinal, lr_test  # noqa: PLC0415
    from riskam.provenance import reproducibility_metadata  # noqa: PLC0415

    rows = read_table_csv(TABLE_PATH)
    val_rows = [r for r in rows if r["bucket"] == "val"]
    if not val_rows:
        print("[error] no val-bucket rows; build-table with split.json present.")
        return 1
    y = np.array([r["gt"] for r in val_rows])
    print(f"[info] fitting on {len(val_rows)} val rows")

    fits = {}
    for name, feats in MODEL_FEATURES.items():
        fit_path = FITS_DIR / f"{name}.json"
        if fit_path.exists() and not args.force:
            from riskam.ordinal import OrdinalFit  # noqa: PLC0415

            fits[name] = OrdinalFit.from_json(fit_path)
            print(f"[skip] {name} (exists; --force to refit)")
            continue
        # Zero-variance features make the Hessian singular (no SEs) and
        # carry no information — drop them, loudly. Known case: the
        # deployed `approach` sub-score is wall-clock-based, so offline
        # fast replay leaves it constant at 0.5 (the kinematic
        # closing_ms, fit on frame timestamps, is its replacement).
        X_all = _feature_matrix(val_rows, feats)
        keep = [i for i in range(len(feats)) if X_all[:, i].std() > 0]
        if len(keep) < len(feats):
            dropped = [f for i, f in enumerate(feats) if i not in keep]
            print(f"[warn] {name}: dropping zero-variance features {dropped}")
            feats = [feats[i] for i in keep]
        fit = fit_ordinal(X_all[:, keep], y, feats)
        fit.to_json(
            FITS_DIR / f"{name}.json",
            extra={"model": name, "provenance": reproducibility_metadata()},
        )
        fits[name] = fit
        print(f"\n# {name} — standardized coefficients (fit on val)")
        for f in feats:
            c = fit.coefs[f]
            print(
                f"  {f:14} coef={c['coef']:+.3f}  se={c['se']:.3f} "
                f"z={c['z']:+.2f}  p={c['p']:.2e}"
            )
        print(f"  converged={fit.converged}  llf={fit.llf:.1f}  aic={fit.aic:.1f}")

    # Likelihood-ratio ablations on the same val rows.
    lr_results = {}
    for full_name, dropped in LR_ABLATIONS:
        full_feats = fits[full_name].feature_names
        if dropped not in full_feats:
            lr_results[f"{full_name}_minus_{dropped}"] = {
                "note": f"{dropped} was zero-variance and already excluded"
            }
            continue
        reduced_feats = [f for f in full_feats if f != dropped]
        reduced = fit_ordinal(
            _feature_matrix(val_rows, reduced_feats), y, reduced_feats
        )
        lr_results[f"{full_name}_minus_{dropped}"] = lr_test(fits[full_name], reduced)
    (OUT_DIR / "lr_tests.json").write_text(json.dumps(lr_results, indent=2))
    print("\n# Likelihood-ratio ablations (does the dropped feature matter?)")
    for k, v in lr_results.items():
        if "note" in v:
            print(f"  {k:28} {v['note']}")
        else:
            print(f"  {k:28} LR={v['stat']:.1f}  df={v['df']}  p={v['p']:.2e}")
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def _class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from riskam.eval_metrics import (  # noqa: PLC0415
        confusion_matrix,
        macro_prf,
        micro_prf,
        per_class_prf,
    )

    cm = confusion_matrix(list(y_true), list(y_pred))
    per_class = per_class_prf(cm)
    return {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro": macro_prf(per_class),
        "micro": micro_prf(cm),
        "mae_class": float(np.mean(np.abs(y_true - y_pred))),
    }


def _per_run_breakdown(rows: list[dict], scores: np.ndarray, y: np.ndarray) -> dict:
    from riskam.proba_metrics import spearman_rho  # noqa: PLC0415

    runs = np.array([r["run"] for r in rows])
    out = {}
    for run in sorted(set(runs)):
        mask = runs == run
        n_cls = len(set(y[mask].tolist()))
        out[run] = {
            "n": int(mask.sum()),
            "spearman_rho": spearman_rho(scores[mask], y[mask]) if n_cls > 1 else None,
        }
    return out


def cmd_evaluate(args: argparse.Namespace) -> int:
    from riskam.ordinal import OrdinalFit  # noqa: PLC0415
    from riskam.proba_metrics import probabilistic_report, spearman_rho  # noqa: PLC0415

    rows = read_table_csv(TABLE_PATH)
    fits = {
        name: OrdinalFit.from_json(FITS_DIR / f"{name}.json")
        for name in MODEL_FEATURES
    }

    comparison_lines = [
        "# Metric comparison — cs_robocup_2023 "
        "(fit on val, frozen evaluation on test)",
        "",
        "| model | split | accuracy | macro-F1 | MAE(cls) | Brier | log-loss | RPS | ECE | Spearman ρ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for bucket in ("val", "test"):
        bucket_rows = [r for r in rows if r["bucket"] == bucket]
        if not bucket_rows:
            print(f"[warn] no rows in bucket {bucket!r}; skipping.")
            continue
        y = np.array([r["gt"] for r in bucket_rows])
        report: dict = {"n": len(bucket_rows)}

        # kin_status visibility: how often was full kinematics available?
        status_hist: dict = {}
        for r in bucket_rows:
            status_hist[r["kin_status"]] = status_hist.get(r["kin_status"], 0) + 1
        report["kin_status_histogram"] = status_hist

        # m0: deterministic baseline.
        m0_pred = np.array([r["m0_class"] for r in bucket_rows])
        m0_risk = np.array([r["m0_risk"] for r in bucket_rows])
        report["m0"] = {
            "classification": _class_report(y, m0_pred),
            "spearman_rho": spearman_rho(m0_risk, y),
            "per_run": _per_run_breakdown(bucket_rows, m0_risk, y),
        }

        # a_raw: fitting-free physics evidence.
        risk_a = np.array([r["risk_a"] for r in bucket_rows])
        hazard = np.array([r["hazard"] for r in bucket_rows])
        report["a_raw"] = {
            "spearman_rho": spearman_rho(risk_a, y),
            "spearman_rho_hazard_only": spearman_rho(hazard, y),
            "per_run": _per_run_breakdown(bucket_rows, risk_a, y),
        }

        # Fitted ordinal models (feature list from the fit itself — it may
        # exclude features that were zero-variance on val).
        for name in MODEL_FEATURES:
            feats = fits[name].feature_names
            proba = fits[name].predict_proba(_feature_matrix(bucket_rows, feats))
            pred = proba.argmax(axis=1)
            report[name] = {
                "classification": _class_report(y, pred),
                "probabilistic": probabilistic_report(y, proba),
            }

        out_path = OUT_DIR / f"report_{bucket}.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"[saved] {out_path}")

        def line(model, acc, f1, mae, brier="—", ll="—", rps="—", ece="—", rho="—"):
            comparison_lines.append(
                f"| {model} | {bucket} | {acc} | {f1} | {mae} "
                f"| {brier} | {ll} | {rps} | {ece} | {rho} |"
            )

        m0c = report["m0"]["classification"]
        line(
            "m0", f"{m0c['micro']['f1']:.3f}", f"{m0c['macro']['f1']:.3f}",
            f"{m0c['mae_class']:.3f}", rho=f"{report['m0']['spearman_rho']:+.3f}",
        )
        line("a_raw", "—", "—", "—", rho=f"{report['a_raw']['spearman_rho']:+.3f}")
        for name in MODEL_FEATURES:
            cls = report[name]["classification"]
            prob = report[name]["probabilistic"]
            line(
                name, f"{cls['micro']['f1']:.3f}", f"{cls['macro']['f1']:.3f}",
                f"{cls['mae_class']:.3f}", f"{prob['brier']:.3f}",
                f"{prob['log_loss']:.3f}", f"{prob['rps']:.3f}",
                f"{prob['reliability']['ece']:.3f}", f"{prob['spearman_rho']:+.3f}",
            )

    comparison_lines += [
        "",
        "m0 = deployed weighted sum + breakpoints; a_raw = kinematic hazard × "
        "awareness (no fitting); m0c/a_cal = ordinal heads on the m0 / "
        "Direction-A scalars; b1_sub = ordinal on the four sub-scores; "
        "b1_kin = ordinal on kinematic features (A+B layered).",
        "See lr_tests.json for the feature-ablation p-values and "
        "fits/*.json for coefficients.",
    ]
    (OUT_DIR / "comparison.md").write_text("\n".join(comparison_lines) + "\n")
    print(f"[saved] {OUT_DIR / 'comparison.md'}")

    if args.figures:
        _make_figures(rows, fits)
    return 0


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
def _risk_color_bgr(v: float) -> tuple:
    """Continuous green → yellow → red by fused risk (BGR)."""
    v = float(np.clip(v, 0.0, 1.0))
    return (0, int(255 * min(1.0, 2.0 * (1.0 - v))), int(255 * min(1.0, 2.0 * v)))


def _put_boxed_text(img, text, org, font_scale, color, thickness=1) -> None:
    import cv2  # noqa: PLC0415

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x = max(3, min(x, img.shape[1] - tw - 6))  # keep the label on-frame
    cv2.rectangle(
        img, (x - 3, y - th - 3), (x + tw + 3, y + baseline + 1), (0, 0, 0), -1
    )
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _annotate_kinematic_frame(
    img, bboxes, kins, scene_risk, scene_held, ego, run: str, rel_t: float
):
    """Overlay the Direction-A channels. Deliberately NO ground truth and
    no deployed-metric output — this is the new metric on its own."""
    import cv2  # noqa: PLC0415

    from riskam.kinematics import KinematicStatus  # noqa: PLC0415

    h, w = img.shape[:2]
    fused = [k.risk if k.risk is not None else k.hazard for k in kins]
    scene_idx = int(np.argmax(fused)) if fused else -1

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = (int(c) for c in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        k = kins[i]
        color = _risk_color_bgr(fused[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if k.status is KinematicStatus.DEAD_ZONE:
            geo = "TOO CLOSE (depth dead zone)"
        elif k.status is KinematicStatus.STATIC:
            geo = f"d {k.d_m:.1f}m  static"
        else:
            geo = f"d {k.d_m:.1f}m  miss {k.d_min_m:.1f}m  in {k.t_cpa_s:.1f}s"
        aw_txt = f"{k.awareness:.2f}" if k.awareness is not None else "n/a"
        chan = f"haz {k.hazard:.2f}  aw {aw_txt}  -> {fused[i]:.2f}"
        _put_boxed_text(img, geo, (x1, max(28, y1 - 22)), 0.45, (255, 255, 255))
        _put_boxed_text(img, chan, (x1, max(46, y1 - 6)), 0.45, color)

        if i == scene_idx:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(img, "*", (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, "*", (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # "hold" marks a value carried by the release (memory bridging a
    # detector dropout), not a live measurement.
    risk_txt = f"risk {scene_risk:.2f}" + (" hold" if scene_held else "")
    _put_boxed_text(
        img, risk_txt, (w - (220 if scene_held else 130), 32), 0.8,
        _risk_color_bgr(scene_risk), thickness=2,
    )
    ego_txt = f"ego {ego.speed:.2f} m/s" if ego is not None else "ego n/a"
    _put_boxed_text(
        img, f"{run}  t+{rel_t:5.1f}s  {ego_txt}  kinematic metric (A)",
        (10, 22), 0.45, (255, 255, 255),
    )
    return img


def cmd_video(args: argparse.Namespace) -> int:
    import cv2  # noqa: PLC0415

    from riskam.kinematics import KinematicTracker, SceneRiskSmoother  # noqa: PLC0415
    from riskam.ml import humandet  # noqa: PLC0415

    cache = FeatureCache(
        FEATURE_CACHE_ROOT, model_sha(humandet.YOLO_POSE_MODEL_PATH), DATASET
    )
    platform = DATASETS[DATASET]["platform"]
    kin = default_kin_params()
    per_run = labelled_runs()
    runs = args.runs or sorted(per_run)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    for run in runs:
        frames = frames_in_time_order(run, per_run.get(run, {}))
        first = next(
            ((p, cache.get(run, p.stem)) for p in frames
             if cache.get(run, p.stem) is not None),
            None,
        )
        if first is None:
            print(f"{run}: nothing cached; run `populate` first.")
            continue

        width = int(first[1].depth_viz.shape[1])
        camera, _ = load_camera_model(run, width, platform.sensor)
        odom = CSRobocup2023OdomIndex(run) or None
        tracker = KinematicTracker(kin, camera)
        scene_smoother = SceneRiskSmoother(kin.release_tau_s, kin.hold_max_s)
        t0 = float(first[0].stem)

        writer = None
        out_path = VIDEOS_DIR / f"{run}.mp4"
        n_written = 0
        for img_path in frames:  # lazy: one frame's primitives at a time
            prim = cache.get(run, img_path.stem)
            if prim is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ts = float(img_path.stem)
            ego = odom.twist_for(ts) if odom else None
            aware = np.asarray(
                humandet.gaze_scores(prim.keypoints_np), dtype=float
            ) if prim.human_bboxes else None
            kins = tracker.update(
                ts, prim.human_bboxes, prim.bbox_depths_m,
                prim.track_ids, ego=ego, awareness=aware,
            )
            instant = max(
                (k.risk if k.risk is not None else k.hazard for k in kins),
                default=0.0,
            )
            scene_risk = scene_smoother.update(ts, instant)
            annotated = _annotate_kinematic_frame(
                img, prim.human_bboxes, kins, scene_risk,
                scene_smoother.held, ego, run, ts - t0,
            )
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps, (w, h),
                )
                if not writer.isOpened():  # codec fallback
                    out_path = VIDEOS_DIR / f"{run}.avi"
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"XVID"),
                        args.fps, (w, h),
                    )
            writer.write(annotated)
            n_written += 1
        if writer is not None:
            writer.release()
            print(f"{run}: {n_written} frames -> {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _make_figures(rows: list[dict], fits: dict) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from riskam.proba_metrics import reliability_table, spearman_rho  # noqa: PLC0415

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ink, muted, grid = "#333333", "#666666", "#e0e0e0"
    plt.rcParams.update({
        "text.color": ink, "axes.labelcolor": ink,
        "xtick.color": muted, "ytick.color": muted,
        "axes.edgecolor": grid, "axes.grid": True,
        "grid.color": grid, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10,
    })

    test_rows = [r for r in rows if r["bucket"] == "test"]
    if not test_rows:
        test_rows = rows
    y = np.array([r["gt"] for r in test_rows])

    # 1 — reliability diagram (test bucket), one curve per fitted model.
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=muted, zorder=1)
    for name in MODEL_FEATURES:
        feats = fits[name].feature_names
        proba = fits[name].predict_proba(_feature_matrix(test_rows, feats))
        rel = reliability_table(y, proba)
        pts = [
            (c, a) for c, a, n in
            zip(rel["mean_confidence"], rel["accuracy"], rel["count"])
            if n > 0
        ]
        xs, ys = zip(*pts)
        # The curves converge at the top-right, so identity goes in a
        # legend (also the contrast relief for the palette WARN).
        ax.plot(
            xs, ys, marker="o", ms=5, lw=2, color=MODEL_COLORS[name],
            zorder=2, label=f"{name} (ECE {rel['ece']:.3f})",
        )
    ax.legend(loc="upper left", frameon=False, labelcolor=ink)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean predicted confidence (bin)")
    ax.set_ylabel("Empirical accuracy (bin)")
    ax.set_title("Reliability — test split (diagonal = perfectly calibrated)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reliability_test.png", dpi=150)
    plt.close(fig)

    # 2 — per-feature Spearman ρ vs gt class (all rows). Polarity encoding:
    # blue = positive, vermillion = negative, zero line as the anchor.
    feat_names = [
        "m0_risk", "risk_a", "hazard", "proximity", "gaze", "x_offset",
        "approach", "d_m", "closing_ms", "tan_speed_ms", "d_min_m",
        "inv_ttc", "awareness",
    ]
    y_all = np.array([r["gt"] for r in rows])
    rhos = [
        spearman_rho(np.array([r[f] for r in rows]), y_all) for f in feat_names
    ]
    order = np.argsort(rhos)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.barh(
        np.arange(len(feat_names)),
        [rhos[i] for i in order],
        color=["#0072B2" if rhos[i] >= 0 else "#D55E00" for i in order],
        height=0.62,
    )
    ax.axvline(0, color=muted, lw=1)
    ax.set_yticks(np.arange(len(feat_names)))
    ax.set_yticklabels([feat_names[i] for i in order])
    ax.set_xlabel("Spearman ρ with ground-truth class (all labelled frames)")
    ax.set_title("Univariate ordinal signal per feature")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spearman_features.png", dpi=150)
    plt.close(fig)

    # 3 — hazard channel by gt class: sequential single-hue (magnitude by
    # ordinal class), violin + median marker.
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    groups = [
        np.array([r["hazard"] for r in rows if r["gt"] == k]) for k in range(4)
    ]
    parts = ax.violinplot(
        [g if len(g) else np.array([0.0]) for g in groups],
        positions=range(4), showmedians=True, widths=0.8,
    )
    blues = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    for k, body in enumerate(parts["bodies"]):
        body.set_facecolor(blues[k])
        body.set_edgecolor("none")
        body.set_alpha(1.0)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(ink)
        parts[key].set_linewidth(1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"class {k}\n(n={len(g)})" for k, g in enumerate(groups)])
    ax.set_ylabel("Kinematic hazard (Direction A channel)")
    ax.set_title("Hazard distribution by annotated risk class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hazard_by_class_violin.png", dpi=150)
    plt.close(fig)

    print(f"[saved] figures under {FIGURES_DIR}")


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="data/cache/odometry coverage, no inference")
    pc.set_defaults(func=cmd_check)

    pp = sub.add_parser("populate", help="YOLO pass filling the feature cache (GPU step)")
    pp.add_argument("--runs", nargs="*", help="subset of runs (default: all)")
    pp.add_argument("--limit-per-run", type=int, default=0, help="cap frames/run for a smoke test")
    pp.add_argument("--device", default="cpu", help="torch device for YOLO (cpu|mps|cuda)")
    pp.set_defaults(func=cmd_populate)

    pt = sub.add_parser("build-table", help="cache -> scene_table.csv (CPU)")
    pt.add_argument("--force", action="store_true", help="rebuild an existing table")
    pt.add_argument("--no-odom", action="store_true", help="ignore odom.csv even if present")
    pt.set_defaults(func=cmd_build_table)

    pf = sub.add_parser("fit", help="fit ordinal models on the val bucket")
    pf.add_argument("--force", action="store_true", help="refit existing fits")
    pf.set_defaults(func=cmd_fit)

    pe = sub.add_parser("evaluate", help="frozen evaluation on val and test")
    pe.add_argument("--figures", action="store_true", help="write PNG figures")
    pe.set_defaults(func=cmd_evaluate)

    pv = sub.add_parser(
        "video",
        help="per-run videos annotated with the kinematic metric "
        "(no ground truth shown)",
    )
    pv.add_argument("--runs", nargs="*", help="subset of runs (default: all)")
    pv.add_argument("--fps", type=int, default=10, help="output frame rate")
    pv.set_defaults(func=cmd_video)

    pa = sub.add_parser("all", help="populate -> build-table -> fit -> evaluate")
    pa.add_argument("--device", default="cpu")
    pa.set_defaults(func=None)

    args = p.parse_args()
    if args.cmd == "all":
        ns = argparse.Namespace
        rc = cmd_populate(ns(runs=None, limit_per_run=0, device=args.device))
        rc = rc or cmd_build_table(ns(force=True, no_odom=False))
        rc = rc or cmd_fit(ns(force=True))
        return rc or cmd_evaluate(ns(figures=True))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
