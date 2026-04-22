#!/usr/bin/env python3
"""
riskam_bagger

Data-logging node that records RiskAM topics to a rosbag2 MCAP file.

Topic names are now configurable via ROS parameters (T3.5) so that the bagger
works when the camera or riskam topics are remapped.
"""

import datetime

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image

from riskam_msgs.msg import FloatStamped


class RiskBagger(Node):
    def __init__(self):
        super().__init__("riskam_bagger")

        # ── Filename parameters ───────────────────────────────────────────────
        self.declare_parameter("bag_folder", "bags")
        self.declare_parameter("bag_name", "riskam_test")
        self.declare_parameter("tag_with_time", True)

        # ── What to log ───────────────────────────────────────────────────────
        self.declare_parameter("log_images", True)
        self.declare_parameter("log_annotated_images", True)
        self.declare_parameter("log_subscores", True)

        # ── Configurable topic names ──────────────────────────────────────────
        self.declare_parameter("topic_camera_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_annotated_image", "/riskam/annotated_image")
        self.declare_parameter("topic_risk_score", "/riskam/risk_score")
        self.declare_parameter("topic_gaze", "/riskam/gaze")
        self.declare_parameter("topic_depth", "/riskam/depth")
        self.declare_parameter("topic_x_pose", "/riskam/x_pose")
        self.declare_parameter("topic_approach", "/riskam/approach")

        p = self.get_parameter
        bag_folder = p("bag_folder").value
        bag_name = p("bag_name").value
        tag_with_time = p("tag_with_time").value
        log_images = p("log_images").value
        log_annotated = p("log_annotated_images").value
        log_subscores = p("log_subscores").value

        t_camera = p("topic_camera_image").value
        t_annotated = p("topic_annotated_image").value
        t_score = p("topic_risk_score").value
        t_gaze = p("topic_gaze").value
        t_depth = p("topic_depth").value
        t_x_pose = p("topic_x_pose").value
        t_approach = p("topic_approach").value

        # ── Build output path ─────────────────────────────────────────────────
        if tag_with_time:
            ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"{bag_name}_{ts}"
        else:
            filename = bag_name
        bag_uri = f"{bag_folder}/{filename}"

        # ── Initialise writer ─────────────────────────────────────────────────
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="mcap")
        converter_options = rosbag2_py.ConverterOptions("", "")
        self.writer.open(storage_options, converter_options)

        # ── Register topics ───────────────────────────────────────────────────
        image_type = "sensor_msgs/msg/Image"
        float_type = "riskam_msgs/msg/FloatStamped"

        topics_to_register: list[tuple[str, str, bool]] = [
            (t_camera, image_type, log_images),
            (t_annotated, image_type, log_annotated),
            (t_score, float_type, True),  # risk score is always logged
            (t_gaze, float_type, log_subscores),
            (t_depth, float_type, log_subscores),
            (t_x_pose, float_type, log_subscores),
            (t_approach, float_type, log_subscores),
        ]

        self._active_topics: set[str] = set()
        for name, type_str, enabled in topics_to_register:
            if not enabled:
                continue
            info = rosbag2_py.TopicMetadata(
                id=0, name=name, type=type_str, serialization_format="cdr"
            )
            self.writer.create_topic(info)
            self._active_topics.add(name)

        # ── Subscriptions ─────────────────────────────────────────────────────
        def _sub_image(topic: str) -> None:
            if topic in self._active_topics:
                self.create_subscription(
                    Image, topic, lambda m: self._write(topic, m), 10
                )

        def _sub_float(topic: str) -> None:
            if topic in self._active_topics:
                self.create_subscription(
                    FloatStamped, topic, lambda m: self._write(topic, m), 10
                )

        _sub_image(t_camera)
        _sub_image(t_annotated)
        _sub_float(t_score)
        _sub_float(t_gaze)
        _sub_float(t_depth)
        _sub_float(t_x_pose)
        _sub_float(t_approach)

        self.get_logger().info(f"RiskBagger writing to {bag_uri}")

    def _write(self, topic: str, msg) -> None:
        self.writer.write(topic, serialize_message(msg), self.get_clock().now().nanoseconds)


def main(args=None):
    rclpy.init(args=args)
    bagger = RiskBagger()
    rclpy.spin(bagger)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
