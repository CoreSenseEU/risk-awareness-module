"""
layer2.py — the Layer-2 (hindsight-oracle) evaluation driver.

Builds and evaluates the per-frame event table (score channels + hindsight
outcomes) that the paper plan makes the new evaluation backbone. Works on
any dataset registered in ``riskam.data.run_datasets`` (cs_robocup_2023,
cs_robocup_2024) — no annotations required.

Sub-commands (the one expensive step is isolated in ``populate``):

  check      data/cache coverage per run, no inference
  populate   THE GPU STEP: one YOLO-Pose pass over ALL frames of each run.
             Resumable at whole-run granularity ONLY — ByteTrack IDs are
             coupled to the frame sequence (see riskam/feature_cache.py
             header), so a run is either skipped (already complete) or
             re-done from its first frame with unconditional cache.put.
             Overnight tip: `caffeinate -i uv run python scripts/layer2.py ...`
  build      cache -> event_table.csv + meta (CPU: tracker + oracle, no GPU)
  evaluate   event_table.csv -> early_warning.json
  report     early_warning.json -> report.md
  video      cache -> per-run videos annotated with the kinematic metric
             (exp_results/<dataset>/layer2/videos/; CPU, no GPU needed)
  all        populate -> build -> evaluate -> report

Outputs land in exp_results/<dataset>/layer2/.

Note: repopulating a partially-cached 2023 run (RB_01) overwrites its
cached ByteTrack IDs with full-sequence ones. Frozen metric_lab artifacts
on disk are unaffected; only a future `metric_lab build-table --force`
would see the (better) new IDs.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
import numpy as np

from riskam.data.run_datasets import (
    RUN_DATASETS,
    RunDataset,
    all_frames_in_time_order,
    labelled_gt,
)
from riskam.event_table import (
    build_event_table,
    read_event_table_csv,
    write_event_table_csv,
)
from riskam.early_warning import evaluate_event_table, render_report_md
from riskam.feature_cache import FeatureCache, model_sha
from riskam.hindsight import OracleParams
from riskam.kinematics import KinematicParams
from riskam.ssm import SSMParams

ROOT = Path(__file__).parent.parent
FEATURE_CACHE_ROOT = ROOT / "feature_cache"
EXP_ROOT = ROOT / "exp_results"

EVENT_TABLE_FNAME = "event_table.csv"
META_FNAME = "event_table_meta.json"
RESULTS_FNAME = "early_warning.json"
REPORT_FNAME = "report.md"


def _layer2_dir(ds: RunDataset) -> Path:
    return EXP_ROOT / ds.name / "layer2"


def _cache(ds: RunDataset) -> FeatureCache:
    from riskam.ml.humandet import YOLO_POSE_MODEL_PATH

    return FeatureCache(FEATURE_CACHE_ROOT, model_sha(YOLO_POSE_MODEL_PATH), ds.name)


def _coverage(ds: RunDataset, cache: FeatureCache, run: str) -> dict:
    """Frame/cache/depth coverage for one run."""
    frames = all_frames_in_time_order(ds, run)
    depth_index = ds.depth_index_cls(run)
    cached = sum(1 for p in frames if cache.get(run, p.stem) is not None)
    # Depth availability bounds what populate can cache (no file I/O).
    depth_avail = sum(1 for p in frames if depth_index.has_for_rgb(p)) \
        if len(frames) and bool(depth_index) else 0
    if cached >= depth_avail and depth_avail > 0:
        verdict = "complete"
    elif cached == 0:
        verdict = "empty"
    else:
        verdict = "partial -> whole-run refresh needed"
    return {
        "frames": len(frames), "depth_available": depth_avail,
        "cached": cached, "verdict": verdict,
        "odom": (ds.raw_dir / run / "odom.csv").is_file(),
        "camera_info": (ds.raw_dir / run / "camera_info.json").is_file(),
    }


# ── check ────────────────────────────────────────────────────────────────────


def cmd_check(args: argparse.Namespace) -> int:
    ds = RUN_DATASETS[args.dataset]
    cache = _cache(ds)
    print(f"dataset: {ds.name}  runs: {len(ds.runs)}")
    for run in ds.runs:
        cov = _coverage(ds, cache, run)
        print(
            f"  {run:<16} frames={cov['frames']:>6} depth={cov['depth_available']:>6} "
            f"cached={cov['cached']:>6} odom={'y' if cov['odom'] else 'N'} "
            f"caminfo={'y' if cov['camera_info'] else 'N'}  {cov['verdict']}"
        )
    return 0


# ── populate (THE GPU STEP) ──────────────────────────────────────────────────


def cmd_populate(args: argparse.Namespace) -> int:
    import cv2  # noqa: PLC0415
    from riskam.ml import featextr, humandet  # noqa: PLC0415
    from riskam.ml.subscores import FrameInputs  # noqa: PLC0415

    ds = RUN_DATASETS[args.dataset]
    if not humandet.YOLO_POSE_MODEL_PATH.is_file():
        print(f"[error] YOLO model missing at {humandet.YOLO_POSE_MODEL_PATH}")
        return 1
    if args.device and args.device != "cpu":
        humandet.model.to(args.device)
        print(f"[info] YOLO model moved to {args.device}")

    cache = _cache(ds)
    platform = ds.platform
    runs = args.runs or list(ds.runs)

    for run in runs:
        frames = all_frames_in_time_order(ds, run)
        if args.limit_per_run:
            frames = frames[: args.limit_per_run]
        cov = _coverage(ds, cache, run)
        if cov["verdict"] == "complete" and not args.refresh:
            print(f"[skip] {run}: cache complete ({cov['cached']} frames)")
            continue
        if cov["cached"] and cov["verdict"] != "complete":
            print(f"[info] {run}: partial cache -> re-running whole run "
                  "(ByteTrack IDs are sequence-coupled)")

        depth_index = ds.depth_index_cls(run)
        humandet.reset_velocity_history()  # ByteTrack/velocity state per run
        t0 = time.time()
        done = miss_depth = 0
        for img_path in frames:
            depth_m = depth_index.load_for_rgb(img_path)
            if depth_m is None:
                miss_depth += 1
                continue
            rgb = cv2.imread(str(img_path))
            if rgb is None:
                miss_depth += 1
                continue
            primitives = featextr.extract_primitives(
                FrameInputs(rgb=rgb, depth_m=depth_m, cmd_vel=None),
                d_safe=platform.d_safe_m,
                depth_near_clip_m=platform.sensor.near_clip_m,
                near_clip_valid_frac_max=platform.sensor.valid_frac_max,
                track_bboxes=True,
            )
            cache.put(run, img_path.stem, primitives)  # unconditional: full pass
            done += 1
            if done % 1000 == 0:
                fps = done / (time.time() - t0)
                print(f"    {run}: {done}/{len(frames)} ({fps:.1f} fps)")
        dt_min = (time.time() - t0) / 60.0
        print(f"[done] {run}: {done} cached, {miss_depth} no-depth, {dt_min:.1f} min")
    return 0


# ── build ────────────────────────────────────────────────────────────────────


def cmd_build(args: argparse.Namespace) -> int:
    from riskam.provenance import reproducibility_metadata
    from riskam.ml.humandet import YOLO_POSE_MODEL_PATH

    ds = RUN_DATASETS[args.dataset]
    out_dir = _layer2_dir(ds)
    table_path = out_dir / EVENT_TABLE_FNAME
    if table_path.exists() and not args.force:
        print(f"[skip] {table_path} exists; use --force to rebuild.")
        return 0

    cache = _cache(ds)
    kin_params = KinematicParams(
        d_safe_m=ds.platform.d_safe_m,
        footprint_radius_m=ds.platform.footprint_radius_m,
    )
    frames_per_run: dict = {}
    camera_by_run: dict = {}
    odom_by_run: dict = {}
    camera_sources: dict = {}
    for run in ds.runs:
        frames = all_frames_in_time_order(ds, run)
        if not frames:
            continue
        # Image width from the first cached frame's depth_viz.
        width = None
        for p in frames:
            prim = cache.get(run, p.stem)
            if prim is not None:
                width = prim.depth_viz.shape[1]
                break
        if width is None:
            print(f"[warn] {run}: nothing cached; run `populate` first. Skipping.")
            continue
        frames_per_run[run] = frames
        camera_by_run[run], camera_sources[run] = ds.camera_loader(
            run, width, ds.platform.sensor
        )
        odom = ds.odom_index_cls(run)
        odom_by_run[run] = odom or None

    if not frames_per_run:
        print("[error] no cached runs; run `populate` first.")
        return 1

    gt_per_run = labelled_gt(ds) or None
    split_obj = None
    if ds.split_path is not None and ds.split_path.exists():
        from riskam.data.splits import load_split

        split_obj = load_split(ds.split_path)

    oracle_params = OracleParams()
    ssm_params = SSMParams()
    print(f"building event table for {len(frames_per_run)} runs...")
    rows = build_event_table(
        cache=cache, frames_per_run=frames_per_run, kin_params=kin_params,
        camera_by_run=camera_by_run, odom_by_run=odom_by_run,
        gt_per_run=gt_per_run, split_obj=split_obj,
        oracle_params=oracle_params, ssm_params=ssm_params,
    )
    write_event_table_csv(rows, table_path, oracle_params)

    n_cached = {run: sum(1 for p in frames if cache.get(run, p.stem) is not None)
                for run, frames in frames_per_run.items()}
    event_counts = {}
    for r in oracle_params.r_grid_m:
        for T in oracle_params.t_grid_s:
            from riskam.hindsight import r_tag, t_tag

            col = f"event_{r_tag(r)}_{t_tag(T)}"
            event_counts[col] = int(sum(int(row[col]) for row in rows))
    meta = {
        "dataset": ds.name,
        "n_rows": len(rows),
        "runs": {run: {"frames": len(frames_per_run[run]), "cached": n_cached[run]}
                 for run in frames_per_run},
        "camera_sources": camera_sources,
        "model_sha": model_sha(YOLO_POSE_MODEL_PATH),
        "kin_params": dataclasses.asdict(kin_params),
        "oracle_params": {
            "r_grid_m": list(oracle_params.r_grid_m),
            "t_grid_s": list(oracle_params.t_grid_s),
            "savgol_window_s": oracle_params.savgol_window_s,
            "max_interp_gap_s": oracle_params.max_interp_gap_s,
            "hysteresis_m": oracle_params.hysteresis_m,
        },
        "ssm_params": ssm_params.__dict__,
        "event_frame_counts": event_counts,
        "provenance": reproducibility_metadata(),
    }
    (out_dir / META_FNAME).write_text(json.dumps(meta, indent=2, default=str))
    print(f"[done] {len(rows)} rows -> {table_path}")
    for col, n in event_counts.items():
        print(f"    {col}: {n} event frames")
    return 0


# ── evaluate / report ────────────────────────────────────────────────────────


def cmd_evaluate(args: argparse.Namespace) -> int:
    ds = RUN_DATASETS[args.dataset]
    out_dir = _layer2_dir(ds)
    table_path = out_dir / EVENT_TABLE_FNAME
    if not table_path.exists():
        print(f"[error] {table_path} missing; run `build` first.")
        return 1
    print(f"reading {table_path}...")
    rows = read_event_table_csv(table_path)
    print(f"evaluating {len(rows)} rows...")
    results = evaluate_event_table(rows)

    # Decode-bug canary: the oracle's "now" distance and the causal nearest
    # distance share measurements and must correlate strongly. Frames whose
    # nearest person is the far-fallback sentinel (== d_safe) are excluded:
    # the oracle censors those by design, nearest_d_m keeps them for SSM.
    d_safe = ds.platform.d_safe_m
    now = np.array([row["d_oracle_now_m"] for row in rows])
    nearest = np.array([row["nearest_d_m"] for row in rows])
    both = np.isfinite(now) & np.isfinite(nearest) & (nearest != d_safe)
    if both.sum() > 10:
        corr = float(np.corrcoef(now[both], nearest[both])[0, 1])
        results["oracle_vs_causal_distance_corr"] = corr
        if corr < 0.95:
            print(f"[warn] oracle/causal distance correlation {corr:.3f} < 0.95 "
                  "— check decode conventions.")
        else:
            print(f"[ok] oracle/causal distance correlation {corr:.3f}")

    (out_dir / RESULTS_FNAME).write_text(json.dumps(results, indent=2, default=str))
    print(f"[done] -> {out_dir / RESULTS_FNAME}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    ds = RUN_DATASETS[args.dataset]
    out_dir = _layer2_dir(ds)
    results_path = out_dir / RESULTS_FNAME
    if not results_path.exists():
        print(f"[error] {results_path} missing; run `evaluate` first.")
        return 1
    results = json.loads(results_path.read_text())
    md = render_report_md(results, ds.name)
    (out_dir / REPORT_FNAME).write_text(md)
    print(f"[done] -> {out_dir / REPORT_FNAME}")
    print()
    print(md)
    return 0


# ── video ────────────────────────────────────────────────────────────────────


def cmd_video(args: argparse.Namespace) -> int:
    """Per-run videos annotated with the kinematic metric, any dataset.

    Mirrors the event table's causal pass (same tracker, same awareness
    source: ``featextr.extract_features``), so what you see is what Layer 2
    scored. Default fps replays at the measured capture rate.
    """
    import cv2  # noqa: PLC0415

    from riskam.kinematics import KinematicTracker, SceneRiskSmoother  # noqa: PLC0415
    from riskam.ml import featextr, humandet  # noqa: PLC0415
    from riskam.visualization import annotate_kinematic_frame  # noqa: PLC0415

    ds = RUN_DATASETS[args.dataset]
    cache = _cache(ds)
    kin = KinematicParams(
        d_safe_m=ds.platform.d_safe_m,
        footprint_radius_m=ds.platform.footprint_radius_m,
    )
    videos_dir = _layer2_dir(ds) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    runs = args.runs or list(ds.runs)

    for run in runs:
        frames = all_frames_in_time_order(ds, run)
        first = next(
            ((p, cache.get(run, p.stem)) for p in frames
             if cache.get(run, p.stem) is not None),
            None,
        )
        if first is None:
            print(f"[skip] {run}: nothing cached; run `populate` first.")
            continue

        width = int(first[1].depth_viz.shape[1])
        camera, _ = ds.camera_loader(run, width, ds.platform.sensor)
        odom = ds.odom_index_cls(run) or None
        tracker = KinematicTracker(kin, camera)
        scene_smoother = SceneRiskSmoother(kin.release_tau_s, kin.hold_max_s)
        humandet.reset_velocity_history()
        t0 = float(first[0].stem)

        cached = [p for p in frames if cache.get(run, p.stem) is not None]
        span_s = float(cached[-1].stem) - t0 if len(cached) > 1 else 0.0
        fps = args.fps or (
            max(1.0, (len(cached) - 1) / span_s) if span_s > 0 else 10.0
        )

        writer = None
        out_path = videos_dir / f"{run}.mp4"
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
            h, w = prim.depth_viz.shape[:2]
            extraction = featextr.extract_features(
                prim, image_shape=(h, w), cmd_vel=None, d_safe=kin.d_safe_m
            )
            aware = (
                np.asarray(extraction.features["gaze"], dtype=float)
                if extraction.features is not None else None
            )
            kins = tracker.update(
                ts, prim.human_bboxes, prim.bbox_depths_m,
                prim.track_ids, ego=ego, awareness=aware,
            )
            instant = max(
                (k.risk if k.risk is not None else k.hazard for k in kins),
                default=0.0,
            )
            scene_risk = scene_smoother.update(ts, instant)
            annotated = annotate_kinematic_frame(
                img, prim.human_bboxes, kins, scene_risk,
                scene_smoother.held, ego, run, ts - t0,
            )
            if writer is None:
                fh, fw = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (fw, fh),
                )
                if not writer.isOpened():  # codec fallback
                    out_path = videos_dir / f"{run}.avi"
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"XVID"),
                        fps, (fw, fh),
                    )
            writer.write(annotated)
            n_written += 1
        if writer is not None:
            writer.release()
            print(f"[done] {run}: {n_written} frames @ {fps:.1f} fps -> {out_path}")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 2)[1])
    parser.add_argument(
        "--dataset", choices=sorted(RUN_DATASETS), default="cs_robocup_2023"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("check", help="data/cache coverage, no inference")
    pc.set_defaults(func=cmd_check)

    pp = sub.add_parser("populate", help="YOLO pass filling the cache (GPU step)")
    pp.add_argument("--runs", nargs="*", default=None)
    pp.add_argument("--device", default=None, help="e.g. mps")
    pp.add_argument("--limit-per-run", type=int, default=0)
    pp.add_argument("--refresh", action="store_true",
                    help="re-run runs whose cache is already complete")
    pp.set_defaults(func=cmd_populate)

    pb = sub.add_parser("build", help="cache -> event_table.csv (CPU)")
    pb.add_argument("--force", action="store_true")
    pb.set_defaults(func=cmd_build)

    pe = sub.add_parser("evaluate", help="event_table.csv -> early_warning.json")
    pe.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("report", help="early_warning.json -> report.md")
    pr.set_defaults(func=cmd_report)

    pv = sub.add_parser(
        "video", help="per-run videos annotated with the kinematic metric"
    )
    pv.add_argument("--runs", nargs="*", default=None)
    pv.add_argument("--fps", type=float, default=0.0,
                    help="output frame rate (default: measured capture rate)")
    pv.set_defaults(func=cmd_video)

    pa = sub.add_parser("all", help="populate -> build -> evaluate -> report")
    pa.add_argument("--device", default=None)
    pa.set_defaults(func=None)

    args = parser.parse_args()
    if args.cmd == "all":
        ns = argparse.Namespace
        rc = cmd_populate(ns(dataset=args.dataset, runs=None, device=args.device,
                             limit_per_run=0, refresh=False))
        if rc:
            return rc
        rc = cmd_build(ns(dataset=args.dataset, force=True))
        if rc:
            return rc
        rc = cmd_evaluate(ns(dataset=args.dataset))
        if rc:
            return rc
        return cmd_report(ns(dataset=args.dataset))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
