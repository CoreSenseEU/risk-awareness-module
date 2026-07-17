"""
data.tf_odometry

Deriving base-frame planar twists from odom→base transform streams.

Shared by the bag extractors (:mod:`riskam.data.extract_cs_robocup`, which
runs in the ROS 2 Docker wrapper, and :mod:`riskam.data.extract_crowdbot`,
which reads ROS 1 bags via the pure-Python ``rosbags`` library). Must stay
free of ROS and torch imports.
"""

import math

# Sample gaps larger than this mean the transform stream was interrupted;
# differentiating across the gap would be meaningless.
TF_MAX_SAMPLE_GAP_S = 0.5


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Yaw (rad) of a quaternion (planar robot assumption)."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def last_segment(frame: str) -> str:
    return frame.rstrip("/").rsplit("/", 1)[-1]


def tf_poses_to_twists(
    poses: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Finite-difference odom→base poses into base-frame planar twists.

    ``poses`` are (t, x, y, yaw) samples in the odom frame, sorted by t.
    Consecutive samples are differentiated; the world-frame velocity is
    rotated into the base frame at the midpoint yaw, matching the twist
    convention of nav_msgs/Odometry (and the 2023 odom.csv files).
    """
    twists: list[tuple[float, float, float, float]] = []
    for (t0, x0, y0, yaw0), (t1, x1, y1, yaw1) in zip(poses, poses[1:]):
        dt = t1 - t0
        if dt <= 0.0 or dt > TF_MAX_SAMPLE_GAP_S:
            continue
        vx_w = (x1 - x0) / dt
        vy_w = (y1 - y0) / dt
        yaw_mid = yaw0 + 0.5 * wrap_angle(yaw1 - yaw0)
        vx = math.cos(yaw_mid) * vx_w + math.sin(yaw_mid) * vy_w
        vy = -math.sin(yaw_mid) * vx_w + math.cos(yaw_mid) * vy_w
        wz = wrap_angle(yaw1 - yaw0) / dt
        twists.append((0.5 * (t0 + t1), vx, vy, wz))
    return twists
