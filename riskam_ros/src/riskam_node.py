#!/usr/bin/env python3

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from riskam import score
from riskam import visualization as vis
from riskam.ml import featextr
from sensor_msgs.msg import Image

from riskam_msgs.msg import FloatStamped


class RiskAM(Node):
    def __init__(self):
        super().__init__("riskam_node")

        # Declare Parameters
        self.declare_parameter("camera_topic", "/camera/realsense/color/image_raw")
        self.declare_parameter("w_proximity", 0.7)
        self.declare_parameter("w_gaze", 0.25)
        self.declare_parameter("w_position", 0.05)
        self.declare_parameter("gamma", 0.5)
        self.declare_parameter("gaze_thresh", [0.1, 0.2])
        self.declare_parameter("visualize_image", True)

        self.camera_topic = self.get_parameter("camera_topic").get_parameter_value()._string_value
        self.w_proximity = self.get_parameter("w_proximity").get_parameter_value()._double_value
        self.w_gaze = self.get_parameter("w_gaze").get_parameter_value()._double_value
        self.w_position = self.get_parameter("w_position").get_parameter_value()._double_value
        self.gamma = self.get_parameter("gamma").get_parameter_value()._double_value
        self.gaze_thresh = (
            self.get_parameter("gaze_thresh").get_parameter_value()._double_array_value
        )
        self.visualize_image = (
            self.get_parameter("visualize_image").get_parameter_value()._bool_value
        )

        # Create bridge
        self.bridge = CvBridge()

        # Declare Subscriptions (May want to make better QoS settings)
        # Set low queue to make sure newer images are
        # taken if images come in faster than RiskAM can handle
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 2
        )

        # Declare Publishers (May want to make better QoS settings)
        self.score_pub = self.create_publisher(FloatStamped, "/riskam/risk_score", 10)
        self.gaze_pub = self.create_publisher(FloatStamped, "/riskam/gaze", 10)
        self.depth_pub = self.create_publisher(FloatStamped, "/riskam/depth", 10)
        self.x_pose_pub = self.create_publisher(FloatStamped, "/riskam/x_pose", 10)
        self.annotated_img_pub = self.create_publisher(Image, "/riskam/annotated_image", 10)

    def image_callback(self, image_msg):
        # Convert the image to cv2 type(rgb image)
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        # Pass the cv_image to the riskam humandet
        human_bboxes, rel_depth, risk_features = featextr.extract_human_risk_awareness_features(
            cv_image,
            depth_gamma=self.gamma,
            gaze_face_offset_lower_threshold_ratio=self.gaze_thresh[0],
            gaze_face_offset_upper_threshold_ratio=self.gaze_thresh[1],
            track_bboxes=True,
        )

        if risk_features is not None:
            # self.get_logger().info(f'Risk Features are: {risk_features}')
            depth_subscore = float(np.max(risk_features["proximity"]))
            gaze_subscore = float(np.max(risk_features["gaze"]))
            x_pose_subscore = float(np.max(risk_features["x_offset"]))
        else:
            depth_subscore = 0.0
            gaze_subscore = 0.0
            x_pose_subscore = 0.0

        # Compute the risk score and the index of the highest risk bbox
        risk_score, max_risk_idx = score.risk_awareness_score(
            risk_features,
            w_proximity=self.w_proximity,
            w_gaze=self.w_gaze,
            w_position=self.w_position,
        )

        # If enables, publish the visualization images
        if self.visualize_image:
            visualized_image = vis.visualize_risk(
                cv_image,
                human_bboxes,
                rel_depth,
                risk_features,
                risk_score,
                max_risk_idx,
            )

        # Parse the Features and Publish them to the risk feature topics
        common_header = image_msg.header  # Could use the clock time instead if needed
        score_msg = FloatStamped()
        score_msg.header = common_header
        score_msg.score = risk_score
        self.score_pub.publish(score_msg)

        depth_msg = FloatStamped()
        depth_msg.header = common_header
        depth_msg.score = depth_subscore
        self.depth_pub.publish(depth_msg)

        gaze_msg = FloatStamped()
        gaze_msg.header = common_header
        gaze_msg.score = gaze_subscore
        self.gaze_pub.publish(gaze_msg)

        x_pose_msg = FloatStamped()
        x_pose_msg.header = common_header
        x_pose_msg.score = x_pose_subscore
        self.x_pose_pub.publish(x_pose_msg)

        if self.visualize_image:
            annotated_img_msg = self.bridge.cv2_to_imgmsg(visualized_image)
            self.annotated_img_pub.publish(annotated_img_msg)


def main(args=None):
    rclpy.init(args=args)
    riskam_node = RiskAM()
    rclpy.spin(riskam_node)
    riskam_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
