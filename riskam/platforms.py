"""
riskam.platforms

Reproducible robot-platform and depth-sensor configurations.

These are factory-preset bundles of physical parameters used by the offline
research / evaluation toolchain to configure ``featextr`` correctly for the
hardware that recorded a dataset. The live ROS node reads scalar parameters
from ``riskam_config.yml`` directly — it does not consume these constants —
so a deployment can override or extend any field without inventing a new
platform name.

The two-level split (sensor → platform) reflects the physics:

- A :class:`DepthSensor` carries properties that depend only on the sensor
  model (near-clip dead zone, expected zero-fill noise floor).
- A :class:`RobotPlatform` bundles a sensor with platform-level safety
  parameters — notably ``d_safe_m``, the safety distance, which depends on
  the robot's speed and stopping distance, *not* the sensor.

Two robots sharing a sensor share the sensor object; two datasets recorded
by the same robot share the platform object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthSensor:
    """Physical properties of a depth sensor relevant to RiskAM scoring.

    Parameters
    ----------
    name : str
        Short, machine-readable identifier (e.g. ``"realsense_d4xx"``).
    near_clip_m : float
        Closest distance the sensor can measure, in metres. ``0.0`` means
        "no known near-clip dead zone" → close-fallback disabled (the
        RealSense / SamXL pre-T2.7 contract). Positive values (e.g. ``0.6``
        for Xtion) tell :func:`riskam.ml.depth.extract_bbox_depths` that
        "all-zero depth in a person-sized bbox" plausibly means "person too
        close to measure".
    valid_frac_max : float
        Maximum valid-pixel fraction within a bbox for the close-fallback
        to fire. ``0.0`` (RealSense) means strictly zero valid pixels;
        positive (e.g. ``0.05`` for Xtion) lets it fire on "mostly empty"
        bboxes whose stray valid pixels are likely background bleed-through.
    """

    name: str
    near_clip_m: float
    valid_frac_max: float


@dataclass(frozen=True)
class RobotPlatform:
    """A robot platform: a depth sensor + safety distance for risk evaluation.

    Parameters
    ----------
    name : str
        Short, machine-readable identifier (e.g. ``"tiago_xtion"``).
    d_safe_m : float
        Safety distance in metres. Persons at or beyond ``d_safe_m`` score
        0 on proximity; the score rises linearly to 1 at the camera. Tune
        to the robot's stopping distance (a function of mass, max speed,
        and braking) plus a margin.
    sensor : DepthSensor
        The platform's depth sensor.
    """

    name: str
    d_safe_m: float
    sensor: DepthSensor


# ── Depth-sensor presets ─────────────────────────────────────────────────────

REALSENSE_D4XX = DepthSensor(
    name="realsense_d4xx",
    near_clip_m=0.0,
    valid_frac_max=0.0,
)
"""Intel RealSense D435 / D435i / D455. Active stereo IR; clean absolute depth
from ~0.3 m onward; zero-pixels rare. Close-fallback disabled by default —
the legacy "all-zero in a bbox → person far away" interpretation is correct."""

PRIMESENSE_XTION = DepthSensor(
    name="primesense_xtion",
    near_clip_m=0.6,
    valid_frac_max=0.05,
)
"""PrimeSense Carmine 1.09 / PAL Xtion (TIAGo, PR2, etc.). Structured-light
with a hard ~0.6 m near-clip dead zone and 50–85% zero-fill in typical indoor
scenes. Close-fallback enabled with a 5% sparse-valid threshold to absorb
background bleed-through that would otherwise be misread as the person's
depth."""


# ── Robot-platform presets ───────────────────────────────────────────────────

RIDGEBACK_D435 = RobotPlatform(
    name="ridgeback_d435",
    d_safe_m=1.5,
    sensor=REALSENSE_D4XX,
)
"""Clearpath Ridgeback (≈70 kg, mecanum drive, ~1 m/s indoor) with RealSense
D435. The SamXL deployment platform; ``d_safe = 1.5 m`` matches the
Ridgeback's stopping distance plus margin and is the ``riskam_config.yml``
default."""

TIAGO_XTION = RobotPlatform(
    name="tiago_xtion",
    d_safe_m=2.5,
    sensor=PRIMESENSE_XTION,
)
"""PAL TIAGo (≈70 kg, differential drive, ~1 m/s indoor) with PAL Xtion. The
cs_robocup_2023 dataset platform; ``d_safe = 2.5 m`` accounts for the Xtion's
broader usable range (people at 1.5–2 m are routinely the closest measurable
objects) and gives the proximity gradient room to discriminate without
penalising mid-range pedestrians."""
