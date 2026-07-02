"""
extract_cs_robocup.py

Extract data from the CoreSense Robocup social image dataset in a ML-friendly format.
"""

import json
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

from riskam.data.paths import CS_ROBOCUP_2023_ROS_DIR, CS_ROBOCUP_2023_ML_RAW_DIR

TOPIC_RGB_IMAGE = "/xtion/rgb/image_raw"
TOPIC_DEPTH_IMAGE = "/xtion/depth/image_raw"

# Aux (non-image) topics for the kinematic metric: robot ego-motion and
# camera intrinsics. /cmd_vel is deliberately NOT used — it carries zero
# messages in RB_01/RB_06/RB_07, whereas controller odometry is dense in
# every run.
TOPIC_ODOM = "/mobile_base_controller/odom"
TOPIC_CAMERA_INFO = "/xtion/rgb/camera_info"

EXTRACTION_PARAMS = {
    TOPIC_RGB_IMAGE: {"encoding": "rgb8", "dir": "rgb"},
    TOPIC_DEPTH_IMAGE: {"encoding": "passthrough", "dir": "depth"},
}


def extract_cs_robocup2023() -> None:
    """
    The main extractor function.
    """

    for rb in range(1, 9):
        bag_dir = CS_ROBOCUP_2023_ROS_DIR / f"RB_0{rb}"

        if rb == 1:
            bag_dir = bag_dir / "mapeo1"

        output_dir = CS_ROBOCUP_2023_ML_RAW_DIR / f"RB_0{rb}"
        rgb_output_dir = output_dir / "rgb"
        depth_output_dir = output_dir / "depth"

        if not bag_dir.exists():
            continue

        rgb_output_dir.mkdir(parents=True, exist_ok=True)
        depth_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting data from RB_0{rb}...")

        for bag_path in bag_dir.glob("*.db3"):
            storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
            converter_options = ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )

            reader = SequentialReader()
            reader.open(storage_options, converter_options)

            # topics = reader.get_all_topics_and_types()
            bridge = CvBridge()

            # for topic in topics:
            #    print(f"Found topic: {topic.name} of type {topic.type}")

            while reader.has_next():
                (topic_name, data, _) = reader.read_next()

                if topic_name in [TOPIC_RGB_IMAGE, TOPIC_DEPTH_IMAGE]:
                    # Deserialize
                    img_msg = deserialize_message(data, RosImage)

                    try:
                        cv_image = bridge.imgmsg_to_cv2(
                            img_msg,
                            desired_encoding=EXTRACTION_PARAMS[topic_name]["encoding"],
                        )

                        timestamp = (
                            img_msg.header.stamp.sec
                            + img_msg.header.stamp.nanosec * 1e-9
                        )

                        if topic_name == TOPIC_RGB_IMAGE:
                            rgb_path = rgb_output_dir / f"{timestamp:.3f}.png"
                            Image.fromarray(cv_image).save(rgb_path)
                        elif topic_name == TOPIC_DEPTH_IMAGE:
                            # Raw uint16 millimetres (RealSense native).
                            # Unit conversion happens at load time via
                            # riskam.ml.depth.depth_mm_to_m.
                            depth_path = depth_output_dir / f"{timestamp:.3f}.npy"
                            np.save(depth_path, cv_image)

                    except Exception as e:  # pylint: disable=broad-except
                        print(f"Error processing image: {e}")


def _bag_dirs():
    """Yield (run_label, bag_dir) for the runs present on disk."""
    for rb in range(1, 9):
        bag_dir = CS_ROBOCUP_2023_ROS_DIR / f"RB_0{rb}"
        if rb == 1:
            bag_dir = bag_dir / "mapeo1"
        if bag_dir.exists():
            yield f"RB_0{rb}", bag_dir


def extract_cs_robocup2023_aux() -> None:
    """Extract odometry twists + camera intrinsics (no image decoding).

    Per run writes, next to ``rgb/`` and ``depth/``:

    - ``odom.csv`` — header ``t_s,linear_x,linear_y,angular_z``, one row per
      ``/mobile_base_controller/odom`` message, sorted by header stamp (the
      same time base as the image filenames);
    - ``camera_info.json`` — ``{width, height, fx, cx}`` from the first
      ``/xtion/rgb/camera_info`` message (intrinsics are static per run).

    This pass is fast (no cv_bridge, no image blobs) and idempotent —
    existing outputs are overwritten.
    """
    from nav_msgs.msg import Odometry  # pylint: disable=import-error
    from sensor_msgs.msg import CameraInfo  # pylint: disable=import-error

    for run, bag_dir in _bag_dirs():
        output_dir = CS_ROBOCUP_2023_ML_RAW_DIR / run
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting aux data from {run}...")
        odom_rows: list[tuple[float, float, float, float]] = []
        cam_info: dict | None = None

        for bag_path in sorted(bag_dir.glob("*.db3")):
            storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
            converter_options = ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )
            reader = SequentialReader()
            reader.open(storage_options, converter_options)
            try:
                from rosbag2_py import StorageFilter  # pylint: disable=import-error

                reader.set_filter(StorageFilter(topics=[TOPIC_ODOM, TOPIC_CAMERA_INFO]))
            except ImportError:
                pass  # older rosbag2: fall back to reading everything

            while reader.has_next():
                (topic_name, data, _) = reader.read_next()
                if topic_name == TOPIC_ODOM:
                    msg = deserialize_message(data, Odometry)
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    tw = msg.twist.twist
                    odom_rows.append(
                        (t, tw.linear.x, tw.linear.y, tw.angular.z)
                    )
                elif topic_name == TOPIC_CAMERA_INFO and cam_info is None:
                    msg = deserialize_message(data, CameraInfo)
                    # K = [fx 0 cx; 0 fy cy; 0 0 1] row-major.
                    cam_info = {
                        "width": int(msg.width),
                        "height": int(msg.height),
                        "fx": float(msg.k[0]),
                        "cx": float(msg.k[2]),
                    }

        if odom_rows:
            odom_rows.sort(key=lambda r: r[0])
            odom_path = output_dir / "odom.csv"
            with odom_path.open("w", encoding="utf-8") as f:
                f.write("t_s,linear_x,linear_y,angular_z\n")
                for t, vx, vy, wz in odom_rows:
                    f.write(f"{t:.6f},{vx:.6f},{vy:.6f},{wz:.6f}\n")
            print(f"  {odom_path.name}: {len(odom_rows)} twists")
        else:
            print(f"  [warn] no {TOPIC_ODOM} messages in {run}")

        if cam_info is not None:
            info_path = output_dir / "camera_info.json"
            info_path.write_text(json.dumps(cam_info, indent=2))
            print(f"  {info_path.name}: fx={cam_info['fx']:.1f} cx={cam_info['cx']:.1f}")
        else:
            print(f"  [warn] no {TOPIC_CAMERA_INFO} messages in {run}")


if __name__ == "__main__":
    extract_cs_robocup2023()
