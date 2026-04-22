"""
riskam.sweep_config

Load YAML-defined experiment sweep configurations.

A sweep config declares two things:

  * ``weight_tuples`` — a list of coupled ``{w_proximity, w_gaze, w_position,
    w_approach}`` dicts. Each tuple must sum to 1.0 (± tolerance) so scores
    stay normalised.
  * Independent axes — ``gaze_sigma_yaw`` and ``gaze_sigma_pitch`` value lists.

The Cartesian product of these produces the sweep. Weight axes are
*coupled* (one tuple = one point) rather than Cartesian because a
full-product over four independent weights would mostly produce
non-normalised combinations.

Ablation configs are just sweep configs with specific tuples — e.g. all
tuples having ``w_approach = 0`` is the "approach-off" ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from riskam.ml.humandet import GAZE_ALGORITHMS, GAZE_ALGORITHM_DEFAULT


DEFAULT_SWEEP_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "sweeps" / "default.yaml"
)

WEIGHT_KEYS: tuple[str, ...] = ("w_proximity", "w_gaze", "w_position", "w_approach")
WEIGHT_SUM_TOLERANCE = 0.01


@dataclass(frozen=True)
class WeightTuple:
    w_proximity: float
    w_gaze: float
    w_position: float
    w_approach: float

    def validate(self) -> None:
        total = (
            self.w_proximity + self.w_gaze + self.w_position + self.w_approach
        )
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Weight tuple {self} sums to {total:.3f}, expected 1.0 "
                f"(tolerance {WEIGHT_SUM_TOLERANCE})."
            )


@dataclass(frozen=True)
class SweepConfig:
    name: str
    description: str
    weight_tuples: tuple[WeightTuple, ...]
    gaze_sigma_yaw: tuple[float, ...]
    gaze_sigma_pitch: tuple[float, ...]
    # Independent Cartesian axis: which gaze algorithm(s) to evaluate.
    # Defaults to ``("head_pose",)`` so old configs without this field continue
    # to behave exactly as before.
    gaze_algorithm: tuple[str, ...] = (GAZE_ALGORITHM_DEFAULT,)

    def __len__(self) -> int:
        return (
            len(self.weight_tuples)
            * len(self.gaze_sigma_yaw)
            * len(self.gaze_sigma_pitch)
            * len(self.gaze_algorithm)
        )

    def iter_experiments(self) -> Iterator[dict]:
        """Yield one ``params`` dict per sweep cell (Cartesian product).

        Keys match the shape expected by :func:`riskam.experiments.run_experiment`.
        """
        for wt in self.weight_tuples:
            for syaw in self.gaze_sigma_yaw:
                for spitch in self.gaze_sigma_pitch:
                    for algo in self.gaze_algorithm:
                        yield {
                            "w_prox": wt.w_proximity,
                            "w_gaze": wt.w_gaze,
                            "w_pos": wt.w_position,
                            "w_approach": wt.w_approach,
                            "gaze_sigma_yaw": syaw,
                            "gaze_sigma_pitch": spitch,
                            "gaze_algorithm": algo,
                        }


def load_sweep_config(path: Path | None = None) -> SweepConfig:
    """Load a sweep config from YAML. If ``path`` is None, use the default."""
    if path is None:
        path = DEFAULT_SWEEP_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Sweep config at {path} must be a YAML mapping.")
    return _parse(data, source=path)


def _parse(data: dict, source: Path | str) -> SweepConfig:
    for required in ("weight_tuples", "gaze_sigma_yaw", "gaze_sigma_pitch"):
        if required not in data:
            raise ValueError(f"Sweep config at {source} is missing key {required!r}.")

    weight_tuples_raw = data["weight_tuples"]
    if not isinstance(weight_tuples_raw, list) or not weight_tuples_raw:
        raise ValueError(
            f"Sweep config at {source}: 'weight_tuples' must be a non-empty list."
        )

    weight_tuples: list[WeightTuple] = []
    for i, raw in enumerate(weight_tuples_raw):
        if not isinstance(raw, dict):
            raise ValueError(
                f"weight_tuples[{i}] at {source} must be a mapping; got {type(raw).__name__}."
            )
        missing = [k for k in WEIGHT_KEYS if k not in raw]
        if missing:
            raise ValueError(
                f"weight_tuples[{i}] at {source} missing keys: {missing}."
            )
        extra = [k for k in raw if k not in WEIGHT_KEYS]
        if extra:
            raise ValueError(
                f"weight_tuples[{i}] at {source} has unknown keys: {extra}."
            )
        wt = WeightTuple(**{k: float(raw[k]) for k in WEIGHT_KEYS})
        wt.validate()
        weight_tuples.append(wt)

    def _float_list(key: str) -> tuple[float, ...]:
        xs = data[key]
        if not isinstance(xs, list) or not xs:
            raise ValueError(
                f"'{key}' at {source} must be a non-empty list; got {xs!r}."
            )
        return tuple(float(x) for x in xs)

    # gaze_algorithm: optional. Accept a single string or a list of strings;
    # default to the head-pose algorithm if absent.
    raw_algos = data.get("gaze_algorithm", GAZE_ALGORITHM_DEFAULT)
    if isinstance(raw_algos, str):
        raw_algos = [raw_algos]
    if not isinstance(raw_algos, list) or not raw_algos:
        raise ValueError(
            f"'gaze_algorithm' at {source} must be a string or non-empty list; "
            f"got {raw_algos!r}."
        )
    for algo in raw_algos:
        if algo not in GAZE_ALGORITHMS:
            raise ValueError(
                f"Unknown gaze algorithm {algo!r} at {source}; "
                f"expected one of {GAZE_ALGORITHMS}."
            )

    return SweepConfig(
        name=str(data.get("name", "unnamed")),
        description=str(data.get("description", "")),
        weight_tuples=tuple(weight_tuples),
        gaze_sigma_yaw=_float_list("gaze_sigma_yaw"),
        gaze_sigma_pitch=_float_list("gaze_sigma_pitch"),
        gaze_algorithm=tuple(str(a) for a in raw_algos),
    )
