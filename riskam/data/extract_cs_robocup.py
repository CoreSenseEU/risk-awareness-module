"""
extract_cs_robocup.py

Extract data from the CoreSense Robocup social image datasets (2023 + 2024)
in a ML-friendly format.

Both years share the output layout
``ml_datasets/<dataset>/raw_dataset/<RUN>/{rgb/, depth/, camera_info.json,
odom.csv}``; the differences (topic names, depth file format, odometry
source, run discovery) are captured in a per-dataset :class:`BagSpec`.

This module runs in a ROS-only environment (e.g. the macOS Docker wrapper)
and must stay torch-free.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
from sensor_msgs.msg import Image as RosImage  # pylint: disable=import-error
from cv_bridge import CvBridge  # pylint: disable=import-error
import numpy as np
from PIL import Image
from rclpy.serialization import deserialize_message  # pylint: disable=import-error
from rosbag2_py import (  # pylint: disable=import-error
    SequentialReader,
    StorageOptions,
    ConverterOptions,
)

from riskam.data.paths import (
    CS_ROBOCUP_2023_ROS_DIR,
    CS_ROBOCUP_2023_ML_RAW_DIR,
    CS_ROBOCUP_2024_ROS_DIR,
    CS_ROBOCUP_2024_ML_RAW_DIR,
)
from riskam.data.tf_odometry import (
    last_segment as _last_segment,
    quat_to_yaw,
    tf_poses_to_twists as _tf_poses_to_twists,
)


@dataclass(frozen=True)
class BagSpec:
    """Per-dataset extraction parameters."""

    name: str
    ros_dir: Path
    ml_raw_dir: Path
    topic_rgb: str
    topic_depth: str
    topic_camera_info: str
    # Odometry-twist source: a nav_msgs/Odometry topic, or None to derive
    # twists from /tf odom→base transforms (finite differences).
    topic_odom: str | None
    depth_ext: str  # "npy" (raw np.save) or "png" (16-bit PNG)


CS_ROBOCUP_2023_SPEC = BagSpec(
    name="cs_robocup_2023",
    ros_dir=CS_ROBOCUP_2023_ROS_DIR,
    ml_raw_dir=CS_ROBOCUP_2023_ML_RAW_DIR,
    topic_rgb="/xtion/rgb/image_raw",
    topic_depth="/xtion/depth/image_raw",
    topic_camera_info="/xtion/rgb/camera_info",
    # /cmd_vel is deliberately NOT used — it carries zero messages in
    # RB_01/RB_06/RB_07, whereas controller odometry is dense in every run.
    topic_odom="/mobile_base_controller/odom",
    depth_ext="npy",
)

CS_ROBOCUP_2024_SPEC = BagSpec(
    name="cs_robocup_2024",
    ros_dir=CS_ROBOCUP_2024_ROS_DIR,
    ml_raw_dir=CS_ROBOCUP_2024_ML_RAW_DIR,
    topic_rgb="/head_front_camera/rgb/image_raw",
    topic_depth="/head_front_camera/depth/image_raw",
    topic_camera_info="/head_front_camera/rgb/camera_info",
    # The 2024 bags carry no odometry topic; twists are derived from /tf.
    topic_odom=None,
    depth_ext="png",
)

TOPIC_TF = "/tf"
TOPIC_CMD_VEL = "/cmd_vel"

# tf-derived odometry: accepted frame names (matched on the last path
# segment, so "robot/odom" and "/odom" both count).
TF_ODOM_PARENTS = {"odom"}
TF_ODOM_CHILDREN = {"base_footprint", "base_link"}



def _bag_dirs_2023():
    """Yield (run_label, bag_dir) for the 2023 runs present on disk."""
    for rb in range(1, 9):
        bag_dir = CS_ROBOCUP_2023_ROS_DIR / f"RB_0{rb}"
        if rb == 1:
            bag_dir = bag_dir / "mapeo1"
        if bag_dir.exists():
            yield f"RB_0{rb}", bag_dir


def _bag_dirs_2024():
    """Yield (run_label, bag_dir) for the 2024 runs present on disk.

    2024 runs are direct subdirectories of the ROS dir, each holding a
    single-file sqlite3 bag next to its ``metadata.yaml``.
    """
    if not CS_ROBOCUP_2024_ROS_DIR.is_dir():
        return
    for run_dir in sorted(CS_ROBOCUP_2024_ROS_DIR.iterdir()):
        if run_dir.is_dir() and (run_dir / "metadata.yaml").is_file():
            yield run_dir.name, run_dir


def _bag_dirs(spec: BagSpec, run: str | None):
    """Yield (run_label, bag_dir), optionally restricted to a single run."""
    dirs = _bag_dirs_2023() if spec.name == "cs_robocup_2023" else _bag_dirs_2024()
    for run_label, bag_dir in dirs:
        if run is None or run_label == run:
            yield run_label, bag_dir


def _open_reader(bag_path: Path) -> SequentialReader:
    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def _depth_to_uint16_mm(cv_image: np.ndarray) -> np.ndarray:
    """Normalize a decoded depth frame to raw uint16 millimetres.

    Sensors emit either 16UC1 (already mm) or 32FC1 (metres, NaN = missing).
    Zero is the missing-data sentinel in both representations.
    """
    if cv_image.dtype == np.uint16:
        return cv_image
    if np.issubdtype(cv_image.dtype, np.floating):
        mm = cv_image.astype(np.float64) * 1000.0
        mm[~np.isfinite(mm)] = 0.0
        return np.clip(mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    raise ValueError(f"Unsupported depth dtype: {cv_image.dtype}")


def extract_images(spec: BagSpec, run: str | None = None) -> None:
    """Extract RGB (PNG) and depth (uint16 mm) frames for the given dataset."""
    bridge = CvBridge()

    for run_label, bag_dir in _bag_dirs(spec, run):
        output_dir = spec.ml_raw_dir / run_label
        rgb_output_dir = output_dir / "rgb"
        depth_output_dir = output_dir / "depth"
        rgb_output_dir.mkdir(parents=True, exist_ok=True)
        depth_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting data from {run_label}...")
        n_rgb = n_depth = 0

        for bag_path in sorted(bag_dir.glob("*.db3")):
            reader = _open_reader(bag_path)
            try:
                from rosbag2_py import StorageFilter  # pylint: disable=import-error

                reader.set_filter(
                    StorageFilter(topics=[spec.topic_rgb, spec.topic_depth])
                )
            except ImportError:
                pass  # older rosbag2: fall back to reading everything

            while reader.has_next():
                (topic_name, data, _) = reader.read_next()

                if topic_name not in (spec.topic_rgb, spec.topic_depth):
                    continue

                img_msg = deserialize_message(data, RosImage)

                try:
                    timestamp = (
                        img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
                    )

                    if topic_name == spec.topic_rgb:
                        cv_image = bridge.imgmsg_to_cv2(
                            img_msg, desired_encoding="rgb8"
                        )
                        rgb_path = rgb_output_dir / f"{timestamp:.3f}.png"
                        Image.fromarray(cv_image).save(rgb_path)
                        n_rgb += 1
                    else:
                        cv_image = bridge.imgmsg_to_cv2(
                            img_msg, desired_encoding="passthrough"
                        )
                        # Raw uint16 millimetres. Unit conversion happens at
                        # load time via riskam.ml.depth.depth_mm_to_m.
                        depth_mm = _depth_to_uint16_mm(cv_image)
                        if spec.depth_ext == "png":
                            depth_path = depth_output_dir / f"{timestamp:.3f}.png"
                            cv2.imwrite(  # pylint: disable=no-member
                                str(depth_path), depth_mm
                            )
                        else:
                            depth_path = depth_output_dir / f"{timestamp:.3f}.npy"
                            np.save(depth_path, depth_mm)
                        n_depth += 1

                except Exception as e:  # pylint: disable=broad-except
                    print(f"Error processing image: {e}")

        print(f"  rgb: {n_rgb} frames, depth: {n_depth} frames")


def _quat_to_yaw(q) -> float:
    """Yaw (rad) of a geometry_msgs Quaternion (planar robot assumption)."""
    return quat_to_yaw(q.x, q.y, q.z, q.w)


def extract_aux(spec: BagSpec, run: str | None = None) -> None:
    """Extract odometry twists + camera intrinsics (no image decoding).

    Per run writes, next to ``rgb/`` and ``depth/``:

    - ``odom.csv`` — header ``t_s,linear_x,linear_y,angular_z``, sorted by
      timestamp (the same time base as the image filenames). Source: the
      spec's odometry topic when present; otherwise derived from /tf
      odom→base transforms, with /cmd_vel (commanded, not measured) as a
      loudly-flagged last resort;
    - ``camera_info.json`` — ``{width, height, fx, cx}`` from the first
      camera-info message (intrinsics are static per run).

    This pass is fast (no cv_bridge, no image blobs) and idempotent —
    existing outputs are overwritten.
    """
    from nav_msgs.msg import Odometry  # pylint: disable=import-error
    from sensor_msgs.msg import CameraInfo  # pylint: disable=import-error
    from tf2_msgs.msg import TFMessage  # pylint: disable=import-error
    from geometry_msgs.msg import Twist  # pylint: disable=import-error

    aux_topics = [spec.topic_camera_info]
    if spec.topic_odom is not None:
        aux_topics.append(spec.topic_odom)
    else:
        aux_topics.extend([TOPIC_TF, TOPIC_CMD_VEL])

    for run_label, bag_dir in _bag_dirs(spec, run):
        output_dir = spec.ml_raw_dir / run_label
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting aux data from {run_label}...")
        odom_rows: list[tuple[float, float, float, float]] = []
        tf_poses: list[tuple[float, float, float, float]] = []
        cmd_vel_rows: list[tuple[float, float, float, float]] = []
        tf_pairs_seen: set[tuple[str, str]] = set()
        cam_info: dict | None = None

        for bag_path in sorted(bag_dir.glob("*.db3")):
            reader = _open_reader(bag_path)
            try:
                from rosbag2_py import StorageFilter  # pylint: disable=import-error

                reader.set_filter(StorageFilter(topics=aux_topics))
            except ImportError:
                pass  # older rosbag2: fall back to reading everything

            while reader.has_next():
                (topic_name, data, t_recv_ns) = reader.read_next()
                if spec.topic_odom is not None and topic_name == spec.topic_odom:
                    msg = deserialize_message(data, Odometry)
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    tw = msg.twist.twist
                    odom_rows.append((t, tw.linear.x, tw.linear.y, tw.angular.z))
                elif topic_name == TOPIC_TF:
                    msg = deserialize_message(data, TFMessage)
                    for tr in msg.transforms:
                        parent = _last_segment(tr.header.frame_id)
                        child = _last_segment(tr.child_frame_id)
                        tf_pairs_seen.add((parent, child))
                        if parent in TF_ODOM_PARENTS and child in TF_ODOM_CHILDREN:
                            t = (
                                tr.header.stamp.sec
                                + tr.header.stamp.nanosec * 1e-9
                            )
                            tf_poses.append(
                                (
                                    t,
                                    tr.transform.translation.x,
                                    tr.transform.translation.y,
                                    _quat_to_yaw(tr.transform.rotation),
                                )
                            )
                elif topic_name == TOPIC_CMD_VEL:
                    msg = deserialize_message(data, Twist)
                    # Twist has no header; use the bag receive time.
                    cmd_vel_rows.append(
                        (t_recv_ns * 1e-9, msg.linear.x, msg.linear.y, msg.angular.z)
                    )
                elif topic_name == spec.topic_camera_info and cam_info is None:
                    msg = deserialize_message(data, CameraInfo)
                    # K = [fx 0 cx; 0 fy cy; 0 0 1] row-major.
                    cam_info = {
                        "width": int(msg.width),
                        "height": int(msg.height),
                        "fx": float(msg.k[0]),
                        "cx": float(msg.k[2]),
                    }

        odom_source = "odom_topic"
        if spec.topic_odom is None:
            if tf_poses:
                tf_poses.sort(key=lambda r: r[0])
                odom_rows = _tf_poses_to_twists(tf_poses)
                odom_source = "tf"
                print(f"  odom source: /tf ({len(tf_poses)} odom→base samples)")
            elif cmd_vel_rows:
                odom_rows = cmd_vel_rows
                odom_source = "cmd_vel"
                print(
                    "  [warn] no odom→base transforms on /tf — falling back "
                    "to /cmd_vel (COMMANDED velocities, not measured). "
                    f"tf frame pairs seen: {sorted(tf_pairs_seen)}"
                )

        if odom_rows:
            odom_rows.sort(key=lambda r: r[0])
            odom_path = output_dir / "odom.csv"
            with odom_path.open("w", encoding="utf-8") as f:
                f.write("t_s,linear_x,linear_y,angular_z\n")
                for t, vx, vy, wz in odom_rows:
                    f.write(f"{t:.6f},{vx:.6f},{vy:.6f},{wz:.6f}\n")
            print(f"  {odom_path.name}: {len(odom_rows)} twists ({odom_source})")
        else:
            print(f"  [warn] no odometry data found in {run_label}")

        if cam_info is not None:
            info_path = output_dir / "camera_info.json"
            info_path.write_text(json.dumps(cam_info, indent=2))
            print(f"  {info_path.name}: fx={cam_info['fx']:.1f} cx={cam_info['cx']:.1f}")
        else:
            print(f"  [warn] no {spec.topic_camera_info} messages in {run_label}")


def extract_cs_robocup2023(run: str | None = None) -> None:
    """Extract RGB + depth frames for cs_robocup_2023."""
    extract_images(CS_ROBOCUP_2023_SPEC, run)


def extract_cs_robocup2023_aux(run: str | None = None) -> None:
    """Extract odometry twists + camera intrinsics for cs_robocup_2023."""
    extract_aux(CS_ROBOCUP_2023_SPEC, run)


def extract_cs_robocup2024(run: str | None = None) -> None:
    """Extract RGB + depth frames for cs_robocup_2024."""
    extract_images(CS_ROBOCUP_2024_SPEC, run)


def extract_cs_robocup2024_aux(run: str | None = None) -> None:
    """Extract odometry twists + camera intrinsics for cs_robocup_2024."""
    extract_aux(CS_ROBOCUP_2024_SPEC, run)


if __name__ == "__main__":
    extract_cs_robocup2023()
