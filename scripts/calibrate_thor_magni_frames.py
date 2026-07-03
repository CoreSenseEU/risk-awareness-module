"""
Empirical frame-convention calibration for THÖR-MAGNI rigid bodies.

QTM exports declare marker/centroid/rotation columns but not the semantic
axes of each rigid body's local frame. This script derives them from the
data, and its output backs the frame constants in ``riskam/mocap_gt.py``:

- Q1 — robot forward axis: in SC3A the DARKO robot drives with differential
  drive, so its planar velocity must align with exactly one chassis axis.
  For each body axis a_k (k-th column of the rotation matrix, which is the
  body axis expressed in world coordinates iff the matrix is body→world),
  report the mean cosine between the planar velocity direction and the
  planar projection of a_k while the robot moves. |mean| ≈ 1 for the
  forward axis (sign gives direction), ≈ 0 for the others — and axis
  ambiguity here would also falsify the body→world assumption itself.
- Q2 — helmet facing axis: same statistic per helmet; people predominantly
  face their walking direction when moving fast, so the facing axis shows
  a consistently high mean cosine across participants and runs.
- Q3 — robot footprint scale: planar spread of the DARKO markers around
  the centroid, a lower-bound anchor for ``footprint_radius_m``.

Usage: uv run python scripts/calibrate_thor_magni_frames.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position
from riskam.data.thor_magni import (
    ROBOT_BODY,
    RigidBodyTrack,
    _parse_metadata,
    discover_runs,
    load_run,
)

MIN_SPEED_MS = 0.3  # only clearly-moving samples constrain the axis
MIN_PLANAR_AXIS_NORM = 0.3  # skip near-vertical axes (no planar direction)


def axis_alignment(track: RigidBodyTrack, time_s: np.ndarray, label: str):
    """Mean cos(angle) between planar velocity and each planar body axis."""
    c = track.centroid_m
    v = np.full_like(c, np.nan)
    v[1:-1] = (c[2:] - c[:-2]) / (time_s[2:] - time_s[:-2])[:, None]
    speed = np.hypot(v[:, 0], v[:, 1])
    ok = (
        ~np.isnan(c[:, 0])
        & ~np.isnan(speed)
        & (speed > MIN_SPEED_MS)
        & ~np.isnan(track.rot[:, 0, 0])
    )
    if ok.sum() < 100:
        print(f"  {label}: only {ok.sum()} moving samples, skipping")
        return
    vhat = v[ok, :2] / speed[ok, None]
    parts = []
    for k in range(3):
        ax = track.rot[ok][:, :2, k]
        norm = np.linalg.norm(ax, axis=1)
        good = norm > MIN_PLANAR_AXIS_NORM
        cosang = np.einsum("ij,ij->i", vhat[good], ax[good] / norm[good, None])
        parts.append(f"axis{k}: mean={np.mean(cosang):+.3f} med={np.median(cosang):+.3f}")
    print(f"  {label} (n={ok.sum()}):  " + "  ".join(parts))


def robot_marker_spread(path: Path) -> None:
    """Planar extent of the DARKO markers around the centroid."""
    _, n_skip = _parse_metadata(path)
    cols = [f"{ROBOT_BODY} - {i} {ax}" for i in range(1, 8) for ax in "XY"]
    cols += [f"{ROBOT_BODY} Centroid_X", f"{ROBOT_BODY} Centroid_Y"]
    df = pd.read_csv(
        path,
        skiprows=n_skip,
        usecols=lambda c: c in cols,
        na_values=["N/A"],
        low_memory=False,
    )
    row = df.dropna().iloc[100]
    cx, cy = row[f"{ROBOT_BODY} Centroid_X"], row[f"{ROBOT_BODY} Centroid_Y"]
    radii = [
        np.hypot(row[f"{ROBOT_BODY} - {i} X"] - cx, row[f"{ROBOT_BODY} - {i} Y"] - cy)
        / 1000.0
        for i in range(1, 8)
    ]
    print(f"  marker planar radii (m): {[f'{r:.2f}' for r in radii]}")


if __name__ == "__main__":
    runs = discover_runs()
    sc3a = [p for p in runs if "SC3A" in p.name]
    print(f"{len(runs)} runs discovered, {len(sc3a)} SC3A")

    print("\n== Q1: robot forward axis (SC3A, differential drive) ==")
    for p in sc3a[:4]:
        r = load_run(p)
        print(p.stem)
        axis_alignment(r.robot, r.time_s, ROBOT_BODY)

    print("\n== Q2: helmet facing axis ==")
    for p in sc3a[:2]:
        r = load_run(p)
        print(p.stem)
        for name, h in list(r.helmets.items())[:4]:
            axis_alignment(h, r.time_s, name)

    print("\n== Q3: DARKO marker planar spread (footprint scale) ==")
    robot_marker_spread(sc3a[0])
