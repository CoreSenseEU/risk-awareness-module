"""Sanity tests for the platform / sensor preset registry.

Pinning the documented preset values ensures (a) accidental edits are caught
and (b) the SamXL deployment configuration (RIDGEBACK_D435) keeps lining up
with the ROS YAML defaults — any drift between the two would mean either
the live deployment or the research code is being told something wrong.
"""

from riskam.ml.depth import D_SAFE_DEFAULT, NEAR_CLIP_M_DEFAULT
from riskam.platforms import (
    BETA_UNAWARE_DEFAULT,
    PRIMESENSE_XTION,
    REALSENSE_D4XX,
    RIDGEBACK_D435,
    TAU_REACTION_S_DEFAULT,
    TIAGO_XTION,
    DepthSensor,
    RobotPlatform,
)


class TestSensorPresets:
    def test_realsense_disables_close_fallback(self):
        # RealSense pre-T2.7 contract: zero near-clip + zero valid-frac
        # threshold. Any change here would silently shift SamXL behaviour.
        assert REALSENSE_D4XX.near_clip_m == 0.0
        assert REALSENSE_D4XX.valid_frac_max == 0.0

    def test_xtion_enables_close_fallback(self):
        assert PRIMESENSE_XTION.near_clip_m == 0.6
        assert PRIMESENSE_XTION.valid_frac_max == 0.05

    def test_rgb_hfov_presets(self):
        # Datasheet values consumed by the kinematic bearing fallback.
        assert PRIMESENSE_XTION.rgb_hfov_deg == 58.0
        assert REALSENSE_D4XX.rgb_hfov_deg == 69.0

    def test_rgb_hfov_defaults_to_none(self):
        s = DepthSensor(name="bare", near_clip_m=0.0, valid_frac_max=0.0)
        assert s.rgb_hfov_deg is None

    def test_sensors_are_immutable(self):
        # frozen=True guards against late mutations sneaking in.
        try:
            REALSENSE_D4XX.near_clip_m = 0.5  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("DepthSensor should be frozen")


class TestPlatformPresets:
    def test_ridgeback_matches_ros_defaults(self):
        # SamXL deployment regression guard: the Ridgeback preset must agree
        # with the ROS-side D_SAFE_DEFAULT and disabled-close-fallback so
        # research code and live deployment can't silently diverge.
        assert RIDGEBACK_D435.d_safe_m == D_SAFE_DEFAULT
        assert RIDGEBACK_D435.sensor.near_clip_m == NEAR_CLIP_M_DEFAULT
        assert RIDGEBACK_D435.sensor is REALSENSE_D4XX

    def test_tiago_xtion_calibration(self):
        # cs_robocup_2023 evaluation depends on these exact values; pin them.
        assert TIAGO_XTION.d_safe_m == 2.5
        assert TIAGO_XTION.sensor is PRIMESENSE_XTION

    def test_footprint_radii(self):
        # TIAGo base ⌀ 0.54 m; Ridgeback half chassis length.
        assert TIAGO_XTION.footprint_radius_m == 0.27
        assert RIDGEBACK_D435.footprint_radius_m == 0.48

    def test_kinematic_referent_constants(self):
        # Physical referents for the experimental kinematic hazard.
        assert TAU_REACTION_S_DEFAULT == 2.0
        assert BETA_UNAWARE_DEFAULT == 1.0

    def test_footprint_defaults_to_zero(self):
        s = DepthSensor(name="bare", near_clip_m=0.0, valid_frac_max=0.0)
        p = RobotPlatform(name="bare_bot", d_safe_m=1.0, sensor=s)
        assert p.footprint_radius_m == 0.0

    def test_platforms_are_immutable(self):
        try:
            TIAGO_XTION.d_safe_m = 3.0  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("RobotPlatform should be frozen")


class TestDatasetPlatformWiring:
    """The DATASETS dict must reference platform objects (not raw scalars)."""

    def test_cs_robocup_2023_references_tiago_xtion(self):
        from riskam.data.ml_datasets import DATASETS  # local: pulls torch
        assert DATASETS["cs_robocup_2023"]["platform"] is TIAGO_XTION


class TestDataclassConstruction:
    """Belt-and-braces: confirm new sensors / platforms can be defined."""

    def test_depth_sensor_construction(self):
        s = DepthSensor(name="custom", near_clip_m=0.4, valid_frac_max=0.02)
        assert s.name == "custom"
        assert s.near_clip_m == 0.4

    def test_robot_platform_construction(self):
        s = DepthSensor(name="custom", near_clip_m=0.4, valid_frac_max=0.0)
        p = RobotPlatform(name="my_robot", d_safe_m=2.0, sensor=s)
        assert p.sensor.name == "custom"
        assert p.d_safe_m == 2.0
