"""
extract_cs_robocup.py

Extract data from the CoreSense Robocup social image dataset in a ML-friendly format.
"""

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


if __name__ == "__main__":
    extract_cs_robocup2023()
