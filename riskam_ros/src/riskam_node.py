#!/usr/bin/env python3
"""
riskam_node

Main RiskAM ROS 2 node.

Improvements over the prototype:
  T1.1/T1.2  — subscribes to the RealSense depth topic; proximity score is
               now calibrated in metres against a configurable d_safe threshold.
  T1.3/T1.4  — gaze uses 2-D head pose (yaw + pitch Gaussian); ByteTrack
               replaces the custom IoU tracker.
  T1.5       — image + depth callbacks only store the latest frame; a dedicated
               processing thread consumes it, avoiding callback backup at 3–7 Hz.
  T1.6       — RiskScorer is instantiated per-node (no global mutable state).
  T2.1       — subscribes to /cmd_vel and uses robot velocity to compute a
               path-aware position sub-score; gracefully falls back to the
               centre-offset heuristic when velocity is unavailable.
  T2.3       — scene risk includes a crowd-penalty factor for multi-person scenes.
  T2.4       — publishes diagnostic_msgs/DiagnosticArray with per-frame timing
               and detection counts.
  T3.2       — weight parameters are validated on startup.
"""

import math
import threading
from time import monotonic

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
import message_filters

from riskam import visualization as vis
from riskam.ml import depth as depth_mod, featextr
from riskam.ml.depth import D_SAFE_DEFAULT
from riskam.ml.humandet import (
    FRONTAL_PITCH_RATIO_DEFAULT,
    SIGMA_PITCH_DEFAULT,
    SIGMA_YAW_DEFAULT,
)
from riskam.score import (
    CROWD_ALPHA_DEFAULT,
    N_FRAMES_AGGREGATE,
    RiskScorer,
    W_APPROACH_EMPIRICAL_DEFAULT,
    W_GAZE_EMPIRICAL_DEFAULT,
    W_POSITION_EMPIRICAL_DEFAULT,
    W_PROXIMITY_EMPIRICAL_DEFAULT,
)
from riskam_msgs.msg import FloatStamped


class RiskAM(Node):
    def __init__(self):
        super().__init__("riskam_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("camera_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("w_proximity", W_PROXIMITY_EMPIRICAL_DEFAULT)
        self.declare_parameter("w_gaze", W_GAZE_EMPIRICAL_DEFAULT)
        self.declare_parameter("w_position", W_POSITION_EMPIRICAL_DEFAULT)
        self.declare_parameter("w_approach", W_APPROACH_EMPIRICAL_DEFAULT)
        self.declare_parameter("d_safe", D_SAFE_DEFAULT)
        self.declare_parameter("crowd_alpha", CROWD_ALPHA_DEFAULT)
        self.declare_parameter("n_frames_aggregate", N_FRAMES_AGGREGATE)
        self.declare_parameter("gaze_sigma_yaw", SIGMA_YAW_DEFAULT)
        self.declare_parameter("gaze_sigma_pitch", SIGMA_PITCH_DEFAULT)
        self.declare_parameter("gaze_frontal_pitch_ratio", FRONTAL_PITCH_RATIO_DEFAULT)
        self.declare_parameter("visualize_image", True)
        self.declare_parameter("sync_slop", 0.1)

        p = self.get_parameter
        self.camera_topic = p("camera_topic").value
        self.depth_topic = p("depth_topic").value
        self.cmd_vel_topic = p("cmd_vel_topic").value
        self.w_proximity = p("w_proximity").value
        self.w_gaze = p("w_gaze").value
        self.w_position = p("w_position").value
        self.w_approach = p("w_approach").value
        self.d_safe = p("d_safe").value
        self.visualize_image = p("visualize_image").value
        self.gaze_sigma_yaw = p("gaze_sigma_yaw").value
        self.gaze_sigma_pitch = p("gaze_sigma_pitch").value
        self.gaze_frontal_pitch_ratio = p("gaze_frontal_pitch_ratio").value
        sync_slop = p("sync_slop").value

        # Validate weights.
        weight_sum = (
            self.w_proximity + self.w_gaze + self.w_position + self.w_approach
        )
        if abs(weight_sum - 1.0) > 0.01:
            self.get_logger().warn(
                f"Risk weights sum to {weight_sum:.3f} (expected 1.0). "
                "Scores will be un-normalised."
            )

        # ── Stateful scorer ───────────────────────────────────────────────────
        self.scorer = RiskScorer(
            n_frames=p("n_frames_aggregate").value,
            crowd_alpha=p("crowd_alpha").value,
        )

        # ── CV bridge ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Latest-frame buffer (async processing pattern) ────────────────────
        self._lock = threading.Lock()
        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None
        self._frame_event = threading.Event()

        # ── Latest cmd_vel ────────────────────────────────────────────────────
        self._cmd_vel: Twist | None = None
        self._cmd_vel_lock = threading.Lock()

        # ── Subscriptions ─────────────────────────────────────────────────────
        color_sub = message_filters.Subscriber(self, Image, self.camera_topic)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=2, slop=sync_slop
        )
        self._sync.registerCallback(self._synced_callback)

        self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_vel_callback, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self.score_pub = self.create_publisher(FloatStamped, "/riskam/risk_score", 10)
        self.gaze_pub = self.create_publisher(FloatStamped, "/riskam/gaze", 10)
        self.depth_pub = self.create_publisher(FloatStamped, "/riskam/depth", 10)
        self.x_pose_pub = self.create_publisher(FloatStamped, "/riskam/x_pose", 10)
        self.annotated_img_pub = self.create_publisher(Image, "/riskam/annotated_image", 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/riskam/diagnostics", 10)

        # ── Processing thread ─────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._proc_thread = threading.Thread(
            target=self._process_loop, name="riskam_proc", daemon=True
        )
        self._proc_thread.start()

        self.get_logger().info(
            f"RiskAM node started | color={self.camera_topic} "
            f"| depth={self.depth_topic} | cmd_vel={self.cmd_vel_topic}"
        )

    # ── Callbacks (just store latest frame) ───────────────────────────────────

    def _synced_callback(self, color_msg: Image, depth_msg: Image) -> None:
        with self._lock:
            self._latest_color = color_msg
            self._latest_depth = depth_msg
        self._frame_event.set()

    def _cmd_vel_callback(self, msg: Twist) -> None:
        with self._cmd_vel_lock:
            self._cmd_vel = msg

    # ── Processing thread ─────────────────────────────────────────────────────

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            signalled = self._frame_event.wait(timeout=1.0)
            if not signalled:
                continue
            self._frame_event.clear()

            with self._lock:
                color_msg = self._latest_color
                depth_msg = self._latest_depth

            if color_msg is None or depth_msg is None:
                continue

            try:
                self._process_frame(color_msg, depth_msg)
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f"Frame processing error: {exc}", throttle_duration_sec=5)

    def _process_frame(self, color_msg: Image, depth_msg: Image) -> None:
        t_start = monotonic()

        # ── Decode images ─────────────────────────────────────────────────────
        cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_image_m = depth_mod.depth_mm_to_m(depth_raw)

        # ── Feature extraction ────────────────────────────────────────────────
        human_bboxes, depth_viz, risk_features, track_ids = (
            featextr.extract_human_risk_awareness_features(
                cv_image,
                depth_image_m=depth_image_m,
                d_safe=self.d_safe,
                gaze_sigma_yaw=self.gaze_sigma_yaw,
                gaze_sigma_pitch=self.gaze_sigma_pitch,
                gaze_frontal_pitch_ratio=self.gaze_frontal_pitch_ratio,
                track_bboxes=True,
            )
        )

        # ── Path-aware x-offset using cmd_vel ─────────────────────────────────
        if risk_features is not None:
            with self._cmd_vel_lock:
                cmd_vel = self._cmd_vel
            if cmd_vel is not None:
                h, w = cv_image.shape[:2]
                path_scores = [
                    _path_proximity_score(b, w, h, cmd_vel) for b in human_bboxes
                ]
                risk_features["x_offset"] = np.array(path_scores, dtype=float)

        # ── Risk scoring ──────────────────────────────────────────────────────
        risk_score, max_risk_idx, per_person = self.scorer.score(
            risk_features,
            track_ids=track_ids,
            w_proximity=self.w_proximity,
            w_gaze=self.w_gaze,
            w_position=self.w_position,
            w_approach=self.w_approach,
        )

        # ── Sub-score extraction for publication ──────────────────────────────
        if risk_features is not None:
            proximity_sub = float(np.max(risk_features["proximity"]))
            gaze_sub = float(np.max(risk_features["gaze"]))
            x_pose_sub = float(np.max(risk_features["x_offset"]))
        else:
            proximity_sub = gaze_sub = x_pose_sub = 0.0

        # ── Visualisation ─────────────────────────────────────────────────────
        if self.visualize_image:
            annotated = vis.visualize_risk(
                cv_image, human_bboxes, depth_viz, risk_features, risk_score, max_risk_idx
            )

        # ── Publish ───────────────────────────────────────────────────────────
        header = color_msg.header

        self.score_pub.publish(_float_stamped(header, risk_score))
        self.depth_pub.publish(_float_stamped(header, proximity_sub))
        self.gaze_pub.publish(_float_stamped(header, gaze_sub))
        self.x_pose_pub.publish(_float_stamped(header, x_pose_sub))

        if self.visualize_image:
            self.annotated_img_pub.publish(self.bridge.cv2_to_imgmsg(annotated))

        # ── Diagnostics ───────────────────────────────────────────────────────
        elapsed_ms = (monotonic() - t_start) * 1000.0
        self._publish_diagnostics(
            header,
            elapsed_ms=elapsed_ms,
            n_persons=len(human_bboxes),
            n_tracks=sum(1 for t in track_ids if t is not None),
            risk_score=risk_score,
        )

    def _publish_diagnostics(self, header, elapsed_ms, n_persons, n_tracks, risk_score):
        status = DiagnosticStatus()
        status.name = "riskam"
        status.hardware_id = "riskam_node"
        status.level = DiagnosticStatus.OK
        status.message = "OK"
        status.values = [
            KeyValue(key="frame_time_ms", value=f"{elapsed_ms:.1f}"),
            KeyValue(key="n_persons", value=str(n_persons)),
            KeyValue(key="n_tracks", value=str(n_tracks)),
            KeyValue(key="risk_score", value=f"{risk_score:.4f}"),
        ]
        arr = DiagnosticArray()
        arr.header = header
        arr.status = [status]
        self.diag_pub.publish(arr)

    def destroy_node(self):
        self._stop_event.set()
        self._frame_event.set()  # unblock the wait
        self._proc_thread.join(timeout=2.0)
        super().destroy_node()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _float_stamped(header, value: float) -> FloatStamped:
    msg = FloatStamped()
    msg.header = header
    msg.score = float(value)
    return msg


def _path_proximity_score(
    bbox: list,
    image_width: int,
    image_height: int,
    cmd_vel: Twist,
) -> float:
    """Path-aware position sub-score using robot velocity.

    Projects the human's image-position onto the robot's instantaneous motion
    direction.  Falls back to centre-offset heuristic when the robot is nearly
    stationary.

    Coordinate convention (ROS REP-103):
      vx > 0 → forward → dangerous zone is image centre (norm_x ≈ 0)
      vy > 0 → left     → in the camera image, the robot's left side appears
                           on the LEFT (negative norm_x)
    """
    vx = cmd_vel.linear.x
    vy = cmd_vel.linear.y
    speed = math.hypot(vx, vy)

    cx = (bbox[0] + bbox[2]) / 2.0
    norm_x = (cx - image_width / 2.0) / (image_width / 2.0)  # −1…+1

    if speed < 0.05:
        # Robot nearly stationary — fall back to centre-offset heuristic.
        return float(1.0 - norm_x**2)

    # Lateral fraction of motion: -vy because +vy=left maps to neg norm_x.
    lat_fraction = -vy / speed  # expected dangerous zone in norm_x coords

    # Gaussian centred on lat_fraction, σ≈0.71 in normalised image space.
    delta = norm_x - lat_fraction
    score = math.exp(-(delta**2) / 0.5)
    return float(np.clip(score, 0.0, 1.0))


def main(args=None):
    rclpy.init(args=args)
    node = RiskAM()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
