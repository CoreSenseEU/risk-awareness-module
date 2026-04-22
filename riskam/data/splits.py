"""
riskam.data.splits

Val/test partitions for offline experiments — a *research-methodology* tool
used by the RiskAM authors when evaluating, **not** a user-facing workflow.
Deployed RiskAM ships with calibrated defaults; roboticists do not need to
run a sweep before using the module.

Why no train split
------------------
RiskAM has no trainable parameters: every sub-score is a deterministic
function of its inputs and a small set of hyperparameters (weights, ``d_safe``,
gaze sigmas, etc.). There is therefore nothing to "train". The sweep selects
hyperparameters on the val set; the final report on the test set is computed
**once**, after all selection is finalised, so the published number is not
contaminated by config-selection bias.

Why stratified within-run, not whole-run holdout
------------------------------------------------
Whole-run holdout (some runs entirely in val, others entirely in test) would
be more rigorous for measuring scene-level generalisation, but the
``cs_robocup_2023`` ground truth makes it unworkable:

  * RB_07 has zero annotations.
  * RB_05 contains only class 3.
  * Classes 0 and 1 live almost entirely in RB_01–RB_03.

Any 2-run holdout therefore either loses entire classes from one bucket or
leaves only one run in the test side, neither of which gives stable headline
metrics. Stratified within-run preserves every class in both buckets in
their natural proportions; the trade-off is that we are measuring
config-selection bias on the dataset's scenes, not generalisation to
unseen scenes — a more honest claim given six usable runs.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from riskam.data.paths import CS_ROBOCUP_2023_ML_DIR


CS_ROBOCUP_2023_SPLIT_PATH = CS_ROBOCUP_2023_ML_DIR / "split.json"

DEFAULT_TEST_FRACTION = 0.3
DEFAULT_SEED = 42
SCHEME = "stratified_within_run"

SplitBucket = str  # "val" | "test"
VAL_BUCKETS: tuple[SplitBucket, SplitBucket] = ("val", "test")


@dataclass(frozen=True)
class Split:
    """Per-(run, frame) bucket assignment + the parameters that produced it."""

    test_fraction: float
    seed: int
    assignments: dict[str, dict[str, SplitBucket]] = field(default_factory=dict)
    notes: str = ""

    def bucket(self, run: str, frame_name: str) -> SplitBucket | None:
        return self.assignments.get(run, {}).get(frame_name)

    def to_json(self) -> dict:
        return {
            "scheme": SCHEME,
            "test_fraction": self.test_fraction,
            "seed": self.seed,
            "assignments": self.assignments,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Split":
        if data.get("scheme") != SCHEME:
            raise ValueError(
                f"Unsupported split scheme {data.get('scheme')!r}; "
                f"expected {SCHEME!r}."
            )
        return cls(
            test_fraction=float(data["test_fraction"]),
            seed=int(data["seed"]),
            assignments=data.get("assignments", {}),
            notes=data.get("notes", ""),
        )


def make_stratified_split(
    ground_truth: dict[str, dict[str, int]],
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    notes: str = "",
) -> Split:
    """Stratified within-run split.

    For each run, groups frames by class label and assigns ``test_fraction``
    of each class to the test bucket; the rest to val. Frame ordering is
    shuffled deterministically per ``seed``.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")

    rng = random.Random(seed)
    assignments: dict[str, dict[str, SplitBucket]] = {}
    for run, frames in ground_truth.items():
        if not frames:
            assignments[run] = {}
            continue
        per_class: dict[int, list[str]] = defaultdict(list)
        for fname, label in frames.items():
            per_class[int(label)].append(fname)

        run_assign: dict[str, SplitBucket] = {}
        for label, names in per_class.items():
            ordered = sorted(names)
            rng.shuffle(ordered)
            n_test = int(round(len(ordered) * test_fraction))
            test_set = set(ordered[:n_test])
            for n in ordered:
                run_assign[n] = "test" if n in test_set else "val"
        assignments[run] = run_assign

    return Split(
        test_fraction=test_fraction,
        seed=seed,
        assignments=assignments,
        notes=notes,
    )


def load_split(path: Path) -> Split:
    with path.open("r", encoding="utf-8") as f:
        return Split.from_json(json.load(f))


def save_split(path: Path, split: Split) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(split.to_json(), f, indent=2)


def filter_by_split(
    ground_truth_run: dict[str, int],
    run: str,
    split: Split,
    bucket: SplitBucket,
) -> dict[str, int]:
    """Return only the frames in ``ground_truth_run`` whose split bucket
    matches ``bucket``. ``ground_truth_run`` is the single-run dict
    ``{frame_name: label}``."""
    return {
        fname: label
        for fname, label in ground_truth_run.items()
        if split.bucket(run, fname) == bucket
    }


def split_summary(
    ground_truth: dict[str, dict[str, int]],
    split: Split,
) -> dict:
    """Per-bucket, per-run frame counts and class histograms."""
    out: dict = {b: {} for b in VAL_BUCKETS}
    for run, frames in ground_truth.items():
        for bucket in VAL_BUCKETS:
            bucket_frames = filter_by_split(frames, run, split, bucket)
            hist: dict = {}
            for label in bucket_frames.values():
                hist[int(label)] = hist.get(int(label), 0) + 1
            out[bucket][run] = {"frames": len(bucket_frames), "classes": hist}
    return out
