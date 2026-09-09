"""Backfill the per-person face-texture statistic into existing feature caches.

The awareness measurability gate (``riskam/ml/facegate.py``) needs a
per-person eye-region texture value that new ``extract_primitives`` calls
compute and cache automatically. Caches populated before the field existed
lack it, which silently degrades the gate to geometry-only. This script
upgrades them in place — CPU only, no YOLO re-run, ByteTrack IDs untouched:
for every cached frame it loads the RGB image, computes
``facegate.measure_face_texture`` from the cached keypoints, and rewrites the
``.npz`` with the extra field (atomic replace; resumable — already-upgraded
files are skipped).

Usage:
    uv run python scripts/add_face_texture_to_cache.py crowdbot_v2
    uv run python scripts/add_face_texture_to_cache.py all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riskam.data.paths import ROOT_DIR  # noqa: E402
from riskam.data.run_datasets import RUN_DATASETS  # noqa: E402
from riskam.feature_cache import CachedFrameFeatures  # noqa: E402
from riskam.ml import facegate  # noqa: E402

CACHE_ROOT = ROOT_DIR / "feature_cache"


def upgrade_dataset(name: str) -> None:
    ds = RUN_DATASETS[name]
    cache_dir = CACHE_ROOT / name
    if not cache_dir.is_dir():
        print(f"[{name}] no cache directory, skipping")
        return

    npz_files = sorted(cache_dir.glob("*/*/*.npz"))
    upgraded = skipped = missing = no_humans = 0
    t0 = time.monotonic()
    for i, npz_path in enumerate(npz_files):
        with np.load(npz_path) as data:
            if "face_texture_present" in data.files:
                skipped += 1
                continue
        prim = CachedFrameFeatures.from_npz(npz_path)

        if prim.keypoints_np is None or len(prim.keypoints_np) == 0:
            # No persons: mark upgraded with an empty texture array so the
            # file is not revisited on resume.
            prim.face_texture_np = np.zeros((0,), dtype=np.float32)
            no_humans += 1
        else:
            run = npz_path.parts[-3]
            rgb_path = ds.raw_dir / run / "rgb" / f"{npz_path.stem}.png"
            rgb = cv2.imread(str(rgb_path))
            if rgb is None:
                print(f"[{name}] missing frame for {run}/{npz_path.stem}, "
                      "leaving cache entry as-is")
                missing += 1
                continue
            prim.face_texture_np = facegate.measure_face_texture(
                rgb, prim.keypoints_np
            )
            upgraded += 1

        tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
        prim.to_npz(tmp)
        tmp.replace(npz_path)

        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.monotonic() - t0)
            print(f"[{name}] {i + 1}/{len(npz_files)} ({rate:.0f} files/s)",
                  flush=True)

    print(f"[{name}] done: {upgraded} upgraded, {no_humans} no-human frames "
          f"marked, {skipped} already current, {missing} missing frames "
          f"({time.monotonic() - t0:.0f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=[*RUN_DATASETS.keys(), "all"],
        help="dataset cache to upgrade, or 'all'",
    )
    args = parser.parse_args()
    names = list(RUN_DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        upgrade_dataset(name)


if __name__ == "__main__":
    main()
