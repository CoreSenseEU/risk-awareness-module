"""Tests for riskam.ssm — closed-form protective-distance checks."""

import numpy as np
import pytest

from riskam.ssm import SSMParams, awareness_human_speed, protective_distance, ssm_margins

P = SSMParams()


def test_protective_distance_closed_form():
    # S_p = 1.6*(0.3+0.6) + 1.0*0.3 + 0.3 = 1.44 + 0.3 + 0.3
    assert protective_distance(1.0, P) == pytest.approx(2.04)
    # Stationary robot: human term + C only.
    assert protective_distance(0.0, P) == pytest.approx(1.74)


def test_protective_distance_monotone_in_robot_speed():
    speeds = [0.0, 0.3, 0.6, 1.0]
    sps = [protective_distance(v, P) for v in speeds]
    assert sps == sorted(sps)


def test_robot_speed_clamped():
    assert protective_distance(-0.5, P) == protective_distance(0.0, P)


def test_awareness_speed_interpolates():
    assert awareness_human_speed(0.0, P) == pytest.approx(P.v_h_ms)
    assert awareness_human_speed(1.0, P) == pytest.approx(P.v_h_aware_ms)
    mid = awareness_human_speed(0.5, P)
    assert P.v_h_aware_ms < mid < P.v_h_ms
    assert awareness_human_speed(float("nan"), P) == pytest.approx(P.v_h_ms)


def test_margins_aware_geq_worst():
    d = np.array([1.0, 2.0, 3.0])
    for aw in (np.array([0.0, 0.5, 1.0]), np.array([1.0, 1.0, 1.0]),
               np.array([np.nan, 0.2, 0.9])):
        worst, aware = ssm_margins(d, aw, v_r_ms=0.5, params=P)
        assert aware >= worst


def test_margins_worst_case_equals_zero_awareness():
    d = np.array([1.5])
    worst, aware = ssm_margins(d, np.array([0.0]), v_r_ms=0.2, params=P)
    assert aware == pytest.approx(worst)


def test_empty_scene_no_alarm():
    worst, aware = ssm_margins(np.array([]), np.array([]), v_r_ms=1.0, params=P)
    assert np.isinf(worst) and np.isinf(aware)


def test_deadzone_person_alarms():
    worst, _ = ssm_margins(np.array([0.0]), np.array([1.0]), v_r_ms=0.0, params=P)
    assert worst < 0
