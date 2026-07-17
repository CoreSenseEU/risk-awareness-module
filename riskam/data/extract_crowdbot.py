"""
data.extract_crowdbot

Extract the CrowdBot v2 recordings (EPFL Qolo, Lausanne street market,
2021-04-24, RDS shared-control session) into the standard per-run layout
``ml_datasets/crowdbot_v2/raw_dataset/<run>/{rgb/, depth/, camera_info.json,
odom.csv}`` shared with the CS RoboCup datasets.

Unlike the RoboCup extractors this reads ROS 1 bags via the pure-Python
``rosbags`` library — no ROS installation or Docker wrapper needed. The
source stream is the forward-facing RealSense (``/camera_left``): RGB at
~13 Hz plus depth aligned to color (16UC1 millimetres, ~6.5 Hz), so depth
shares the RGB intrinsics. Ego-motion twists are derived from the bags'
odom→tf_qolo transforms (there is no odometry topic).

Quirk: the defacing pipeline re-encoded the color stream — messages declare
``bgr8`` but the bytes are in RGB order (verified against street scenes:
blue sky, correct skin tones). Frames are therefore saved without a channel
swap, matching the RGB PNGs the RoboCup extractors produce.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from riskam.data.paths import CROWDBOT_V2_ML_RAW_DIR, CROWDBOT_V2_ROS_DIR
from riskam.data.tf_odometry import last_segment, quat_to_yaw, tf_poses_to_twists

TOPIC_RGB = "/camera_left/color/image_raw"
TOPIC_DEPTH = "/camera_left/aligned_depth_to_color/image_raw"
TOPIC_CAMERA_INFO = "/camera_left/color/camera_info"
TOPIC_TF = "/tf"

TF_ODOM_PARENT = "odom"
TF_ODOM_CHILD = "tf_qolo"

# Run label -> source bag file name. Labels are the recording's start time
# (hhmm); all seven RDS bags were recorded the same morning.
RUNS: dict[str, str] = {
    f"rds_{hhmm}": f"defaced_2021-04-24-{hhmm[:2]}-{hhmm[2:]}-{ss}_filtered_lidar_odom.bag"
    for hhmm, ss in [
        ("1120", "18"),
        ("1123", "43"),
        ("1135", "54"),
        ("1140", "33"),
        ("1143", "54"),
        ("1148", "21"),
        ("1155", "30"),
    ]
}


def _stamp(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def extract_crowdbot_v2(run: str | None = None) -> None:
    """Extract RGB, depth, odometry and intrinsics for the given run(s).

    Only runs whose bag is currently staged in ``ros_datasets/crowdbot_v2``
    are processed (the bags are staged one at a time — see
    ``scripts/prepare_crowdbot_v2.sh``).
    """
    from rosbags.highlevel import AnyReader  # local: keep module import cheap

    for run_label, bag_name in RUNS.items():
        if run is not None and run_label != run:
            continue
        bag_path = CROWDBOT_V2_ROS_DIR / bag_name
        if not bag_path.is_file():
            if run is not None:
                print(f"[error] {bag_path} not staged.")
            continue

        out_dir = CROWDBOT_V2_ML_RAW_DIR / run_label
        rgb_dir = out_dir / "rgb"
        depth_dir = out_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {run_label} from {bag_name}...")
        n_rgb = n_depth = 0
        cam_info: dict | None = None
        tf_poses: list[tuple[float, float, float, float]] = []

        with AnyReader([bag_path]) as reader:
            conns = [
                c
                for c in reader.connections
                if c.topic in (TOPIC_RGB, TOPIC_DEPTH, TOPIC_CAMERA_INFO, TOPIC_TF)
            ]
            for conn, _, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                if conn.topic == TOPIC_RGB:
                    # Bytes are RGB despite the declared bgr8 (see module
                    # docstring) — save without swapping.
                    rgb = np.frombuffer(msg.data, np.uint8).reshape(
                        msg.height, msg.width, 3
                    )
                    Image.fromarray(rgb).save(rgb_dir / f"{_stamp(msg.header):.3f}.png")
                    n_rgb += 1
                elif conn.topic == TOPIC_DEPTH:
                    if msg.encoding != "16UC1":
                        raise ValueError(f"Unexpected depth encoding {msg.encoding}")
                    depth_mm = np.frombuffer(msg.data, np.uint16).reshape(
                        msg.height, msg.width
                    )
                    # 16-bit PNG in raw millimetres, like cs_robocup_2024.
                    Image.fromarray(depth_mm).save(
                        depth_dir / f"{_stamp(msg.header):.3f}.png"
                    )
                    n_depth += 1
                elif conn.topic == TOPIC_TF:
                    for tr in msg.transforms:
                        if (
                            last_segment(tr.header.frame_id) == TF_ODOM_PARENT
                            and last_segment(tr.child_frame_id) == TF_ODOM_CHILD
                        ):
                            q = tr.transform.rotation
                            tf_poses.append(
                                (
                                    _stamp(tr.header),
                                    tr.transform.translation.x,
                                    tr.transform.translation.y,
                                    quat_to_yaw(q.x, q.y, q.z, q.w),
                                )
                            )
                elif conn.topic == TOPIC_CAMERA_INFO and cam_info is None:
                    k = msg.k if hasattr(msg, "k") else msg.K
                    cam_info = {
                        "width": int(msg.width),
                        "height": int(msg.height),
                        "fx": float(k[0]),
                        "cx": float(k[2]),
                    }

        tf_poses.sort(key=lambda r: r[0])
        twists = tf_poses_to_twists(tf_poses)
        if twists:
            with (out_dir / "odom.csv").open("w", encoding="utf-8") as f:
                f.write("t_s,linear_x,linear_y,angular_z\n")
                for t, vx, vy, wz in twists:
                    f.write(f"{t:.6f},{vx:.6f},{vy:.6f},{wz:.6f}\n")
        else:
            print(f"  [warn] no odom->tf_qolo transforms in {run_label}")

        if cam_info is not None:
            (out_dir / "camera_info.json").write_text(json.dumps(cam_info, indent=2))
        else:
            print(f"  [warn] no {TOPIC_CAMERA_INFO} messages in {run_label}")

        fx_txt = f"fx={cam_info['fx']:.1f}" if cam_info else "no camera info"
        print(f"  rgb: {n_rgb}, depth: {n_depth}, odom twists: {len(twists)}, {fx_txt}")
