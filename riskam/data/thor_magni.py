"""
data.thor_magni

Parser for the THÖR-MAGNI motion-capture CSVs (Zenodo 10.5281/zenodo.10407223).

THÖR-MAGNI is the Layer-1 (measurement-validity) dataset of the evaluation
plan (``docs/private/paper-plan.md`` §6): Qualisys mocap at 100 Hz provides
ground-truth positions and orientations for every participant helmet and for
the DARKO robot, against which RiskAM's vision-derived kinematic quantities
are validated. This module only *reads* the per-run CSVs; the ground-truth
kinematics tables are built on top by ``riskam.mocap_gt``.

CSV layout (see the archive's ``docs/tutorials.md``): a metadata header
(``FILE_ID`` … ``MARKER_NAMES`` rows), then a ``Frame,Time,...`` column-header
line, then 100 Hz samples. Units: seconds and millimetres. Per rigid body the
columns are per-marker XYZ, ``<body> Centroid_{X,Y,Z}``, and a 9-element
rotation matrix ``<body> R0``–``R8`` whose row/column layout is declared by
the ``CONTIGUOUS_ROTATION_MATRIX`` header row. Missing samples are empty/
``N/A`` cells. Eye-tracking columns (present for some helmets) are ignored
here.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from riskam.data.paths import THOR_MAGNI_SRC_DIR

ROBOT_BODY = "DARKO_Robot"
_HELMET_PREFIX = "Helmet_"
MOCAP_HZ = 100.0

# Header rows are "KEY,val,val,..." lines preceding the data column header.
_DATA_HEADER_PREFIX = "Frame,Time"
_MM_TO_M = 1e-3


@dataclass(frozen=True)
class RigidBodyTrack:
    """One rigid body's 6-DOF mocap track over a run.

    ``centroid_m`` is (n, 3) in metres, world frame; ``rot`` is (n, 3, 3),
    the body→world rotation per sample. Samples where the body was not
    tracked are NaN in both.
    """

    name: str
    role: str
    centroid_m: np.ndarray
    rot: np.ndarray

    @property
    def is_helmet(self) -> bool:
        return self.name.startswith(_HELMET_PREFIX)


@dataclass(frozen=True)
class ThorMagniRun:
    """One THÖR-MAGNI recording: shared 100 Hz clock + rigid-body tracks."""

    file_id: str
    scenario: str  # "SC1A" … "SC5"
    time_s: np.ndarray
    bodies: dict[str, RigidBodyTrack]

    @property
    def robot(self) -> RigidBodyTrack | None:
        return self.bodies.get(ROBOT_BODY)

    @property
    def helmets(self) -> dict[str, RigidBodyTrack]:
        return {n: b for n, b in self.bodies.items() if b.is_helmet}


def discover_runs(root: Path = THOR_MAGNI_SRC_DIR) -> list[Path]:
    """All per-run CSVs under ``CSVs_Scenarios/``, sorted for determinism."""
    return sorted((root / "CSVs_Scenarios").glob("Scenario_*/THOR-Magni_*.csv"))


def _parse_metadata(path: Path) -> tuple[dict[str, list[str]], int]:
    """Read the metadata rows; return {key: values} and the number of rows
    to skip so ``pandas`` starts at the ``Frame,Time,...`` column header."""
    meta: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.startswith(_DATA_HEADER_PREFIX):
                return meta, i
            # Metadata rows are trivially comma-split except the rotation
            # layout, whose single value is a quoted Python literal
            # containing commas — pull it out with a regex instead.
            if line.startswith("CONTIGUOUS_ROTATION_MATRIX"):
                m = re.search(r'"(.*)"', line)
                meta["CONTIGUOUS_ROTATION_MATRIX"] = [m.group(1)] if m else []
                continue
            cells = line.rstrip("\n").split(",")
            meta[cells[0]] = [c for c in cells[1:] if c]
    raise ValueError(f"{path}: no '{_DATA_HEADER_PREFIX}' column header found")


def _rotation_layout(meta: dict[str, list[str]]) -> np.ndarray:
    """(3, 3) array of R-element indices: layout[i, j] = which ``R<k>``
    column holds matrix element [i, j]."""
    literal = meta.get("CONTIGUOUS_ROTATION_MATRIX", [None])[0]
    if not literal:
        raise ValueError("missing CONTIGUOUS_ROTATION_MATRIX header row")
    rows = ast.literal_eval(literal)  # e.g. [['R0','R3','R6'], ...]
    return np.array([[int(el[1:]) for el in row] for row in rows])


def load_run(path: Path) -> ThorMagniRun:
    """Parse one THÖR-MAGNI CSV into centroid + rotation tracks per body."""
    meta, n_skip = _parse_metadata(path)
    file_id = meta["FILE_ID"][0]
    scenario_match = re.search(r"SC\d[AB]?", file_id)
    if scenario_match is None:
        raise ValueError(f"{path}: cannot derive scenario from '{file_id}'")

    names = meta["BODY_NAMES"]
    roles = meta.get("BODY_ROLES", [""] * len(names))
    layout = _rotation_layout(meta)

    wanted = ["Time"]
    for name in names:
        wanted += [f"{name} Centroid_{ax}" for ax in "XYZ"]
        wanted += [f"{name} R{k}" for k in range(9)]

    df = pd.read_csv(
        path,
        skiprows=n_skip,
        usecols=lambda c: c in wanted,
        na_values=["N/A"],
        low_memory=False,
    )
    n = len(df)

    bodies: dict[str, RigidBodyTrack] = {}
    for name, role in zip(names, roles):
        centroid = np.full((n, 3), np.nan)
        for j, ax in enumerate("XYZ"):
            col = f"{name} Centroid_{ax}"
            if col in df:
                centroid[:, j] = df[col].to_numpy(dtype=float) * _MM_TO_M
        rot = np.full((n, 3, 3), np.nan)
        r_cols = [f"{name} R{k}" for k in range(9)]
        if all(c in df for c in r_cols):
            r_flat = df[r_cols].to_numpy(dtype=float)  # (n, 9), R0..R8
            rot = r_flat[:, layout]  # (n, 3, 3) via fancy indexing
        bodies[name] = RigidBodyTrack(
            name=name, role=role.strip(), centroid_m=centroid, rot=rot
        )

    return ThorMagniRun(
        file_id=file_id,
        scenario=scenario_match.group(0),
        time_s=df["Time"].to_numpy(dtype=float),
        bodies=bodies,
    )
