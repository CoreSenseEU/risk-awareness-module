"""
Build the THÖR-MAGNI ground-truth kinematics tables (evaluation Layer 1).

For every run under ``ros_datasets/thor_magni/CSVs_Scenarios/`` this writes

    ml_datasets/thor_magni/gt/<file_id>.csv.gz    the GT table
    ml_datasets/thor_magni/gt/<file_id>.meta.json facing signs, params, provenance

using the DARKO_KINECT platform constants. See ``riskam/mocap_gt.py`` for
definitions and frame conventions.

Usage:
    uv run python scripts/build_thor_magni_gt.py [--runs SC3A 130522_SC3A_R1 ...]
                                                 [--hz 25]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.paths import THOR_MAGNI_GT_DIR
from riskam.data.thor_magni import discover_runs, load_run
from riskam.kinematics import KinematicParams
from riskam.mocap_gt import OUT_HZ_DEFAULT, build_gt_table
from riskam.platforms import DARKO_KINECT
from riskam.provenance import reproducibility_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build THÖR-MAGNI GT kinematics tables")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Substring filters on run file names (e.g. SC3A, 130522_SC3A_R1); default all",
    )
    parser.add_argument("--hz", type=float, default=OUT_HZ_DEFAULT, help="Output sample rate")
    args = parser.parse_args()

    params = KinematicParams(
        d_safe_m=DARKO_KINECT.d_safe_m,
        footprint_radius_m=DARKO_KINECT.footprint_radius_m,
    )

    paths = discover_runs()
    if args.runs:
        paths = [p for p in paths if any(f in p.stem for f in args.runs)]
    if not paths:
        sys.exit("no runs matched")

    THOR_MAGNI_GT_DIR.mkdir(parents=True, exist_ok=True)
    skipped = []
    for i, path in enumerate(paths, 1):
        run = load_run(path)
        try:
            table, meta = build_gt_table(run, params, out_hz=args.hz)
        except ValueError as exc:
            skipped.append(str(exc))
            print(f"[{i}/{len(paths)}] SKIP {run.file_id}: {exc}")
            continue
        meta["provenance"] = reproducibility_metadata()
        table.to_csv(THOR_MAGNI_GT_DIR / f"{run.file_id}.csv.gz", index=False)
        with open(THOR_MAGNI_GT_DIR / f"{run.file_id}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[{i}/{len(paths)}] {run.file_id}: {len(table)} rows")
    if skipped:
        print(f"\n{len(skipped)} run(s) skipped:")
        for msg in skipped:
            print(f"  - {msg}")
