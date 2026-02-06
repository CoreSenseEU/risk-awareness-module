#!/usr/bin/env python3

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

        # Load Parameters
        self.declare_parameter("bag_folder", "bags")
        self.declare_parameter("bag_name", "riskam_test")
        self.declare_parameter("tag_with_time", True)
        self.declare_parameter("log_images", True)
        self.declare_parameter("log_annotated_images", True)
        self.declare_parameter("log_subscores", True)

        self.bag_folder = self.get_parameter("bag_folder").get_parameter_value()._string_value
        self.bag_name = self.get_parameter("bag_name").get_parameter_value()._string_value
        self.tag_with_time = self.get_parameter("tag_with_time").get_parameter_value()._bool_value
        self.log_images = self.get_parameter("log_images").get_parameter_value()._bool_value
        self.log_annotated_images = (
            self.get_parameter("log_annotated_images").get_parameter_value()._bool_value
        )
        self.log_subscores = self.get_parameter("log_subscores").get_parameter_value()._bool_value

        # Instantiate the writer
        self.writer = rosbag2_py.SequentialWriter()

        # Create the filename for the bag path(prefix + time)
        if self.tag_with_time:
            time_suffix = (
                str(datetime.datetime.now()).replace(" ", "-").replace(":", "-").split(".")[0]
            )
            filename = "_".join((self.bag_name, time_suffix))
        else:
            filename = self.bag_name

        bag_uri = "/".join((self.bag_folder, filename))

        # Use mcap files
        storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="mcap")

        # Base serialization conversion(i.e. no conversion)
        converter_options = rosbag2_py.ConverterOptions("", "")
        self.writer.open(storage_options, converter_options)

        # register topics w/ bagger
        topics = [
            ("/camera/realsense/color/image_raw", "sensor_msgs/msg/Image"),
            ("/riskam/annotated_image", "sensor_msgs/msg/Image"),
            ("/riskam/risk_score", "riskam_msgs/msg/FloatStamped"),
            ("/riskam/gaze", "riskam_msgs/msg/FloatStamped"),
            ("/riskam/depth", "riskam_msgs/msg/FloatStamped"),
            ("/riskam/x_pose", "riskam_msgs/msg/FloatStamped"),
        ]

        for name, type_str in topics:
            topic_info = rosbag2_py.TopicMetadata(
                id=0, name=name, type=type_str, serialization_format="cdr"
            )
            self.writer.create_topic(topic_info)

        # Subscribe to Desired Topics
        # Use Lambda functions to pass topic into function as well for logging
        self.create_subscription(
            Image,
            "/camera/realsense/color/image_raw",
            lambda msg: self.write_callback("/camera/realsense/color/image_raw", msg),
            10,
        )

        self.create_subscription(
            Image,
            "/riskam/annotated_image",
            lambda msg: self.write_callback("/riskam/annotated_image", msg),
            10,
        )

        self.create_subscription(
            FloatStamped,
            "/riskam/risk_score",
            lambda msg: self.write_callback("/riskam/risk_score", msg),
            10,
        )

        self.create_subscription(
            FloatStamped, "/riskam/gaze", lambda msg: self.write_callback("/riskam/gaze", msg), 10
        )

        self.create_subscription(
            FloatStamped,
            "/riskam/depth",
            lambda msg: self.write_callback("/riskam/depth", msg),
            10,
        )

        self.create_subscription(
            FloatStamped,
            "/riskam/x_pose",
            lambda msg: self.write_callback("/riskam/x_pose", msg),
            10,
        )

    # Define the callback for topics
    def write_callback(self, topic_name, msg):
        self.writer.write(topic_name, serialize_message(msg), self.get_clock().now().nanoseconds)


def main(args=None):
    rclpy.init(args=args)
    bagger = RiskBagger()
    rclpy.spin(bagger)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
