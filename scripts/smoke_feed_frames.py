#!/usr/bin/env python3
"""Publish extracted dataset frames as live camera topics for node smoke tests.

Feeds real RGB + depth frames (and the run's camera intrinsics) from
``ml_datasets/<dataset>/raw_dataset/<run>/`` to ``/smoke/rgb``,
``/smoke/depth``, and ``/smoke/camera_info`` at a fixed rate, then exits.
RGB and depth frames share one freshly-stamped header per pair, so the
node's approximate-time synchroniser always matches them.

Runs inside the ROS 2 container started by
``scripts/smoke_test_node_macos.sh``; rclpy + cv2 + numpy only.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def _image_msg(stamp, frame_id: str, arr: np.ndarray, encoding: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * (arr.shape[2] if arr.ndim == 3 else 1) * arr.dtype.itemsize
    msg.data = arr.tobytes()
    return msg


class FrameFeeder(Node):
    def __init__(self, run_dir: Path, n_frames: int, hz: float):
        super().__init__("smoke_frame_feeder")
        self.rgb_pub = self.create_publisher(Image, "/smoke/rgb", 10)
        self.depth_pub = self.create_publisher(Image, "/smoke/depth", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/smoke/camera_info", 10)

        info = json.loads((run_dir / "camera_info.json").read_text())
        self.cam_info = CameraInfo()
        self.cam_info.width = int(info["width"])
        self.cam_info.height = int(info["height"])
        fx, cx = float(info["fx"]), float(info["cx"])
        self.cam_info.k = [fx, 0.0, cx, 0.0, fx, info["height"] / 2.0, 0.0, 0.0, 1.0]

        rgb_files = sorted((run_dir / "rgb").glob("*.png"), key=lambda p: float(p.stem))
        depth_files = sorted(
            (run_dir / "depth").glob("*.npy"), key=lambda p: float(p.stem)
        )
        depth_stems = [float(p.stem) for p in depth_files]
        self.pairs: list[tuple[Path, Path]] = []
        for rgb in rgb_files[: n_frames]:
            i = bisect.bisect_left(depth_stems, float(rgb.stem))
            cand = [j for j in (i - 1, i) if 0 <= j < len(depth_files)]
            j = min(cand, key=lambda j: abs(depth_stems[j] - float(rgb.stem)))
            self.pairs.append((rgb, depth_files[j]))
        self.get_logger().info(f"feeding {len(self.pairs)} frame pairs at {hz} Hz")

        self._idx = 0
        self.done = False
        self.create_timer(1.0 / hz, self._tick)

    def _tick(self) -> None:
        if self._idx >= len(self.pairs):
            self.done = True
            return
        rgb_path, depth_path = self.pairs[self._idx]
        self._idx += 1
        rgb = cv2.imread(str(rgb_path))
        depth = np.load(depth_path).astype(np.uint16)
        if rgb is None:
            return
        stamp = self.get_clock().now().to_msg()
        self.cam_info.header.stamp = stamp
        self.info_pub.publish(self.cam_info)
        self.rgb_pub.publish(_image_msg(stamp, "camera", rgb, "bgr8"))
        self.depth_pub.publish(_image_msg(stamp, "camera", depth, "16UC1"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cs_robocup_2023")
    parser.add_argument("--run", default="RB_02")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--hz", type=float, default=2.0)
    args = parser.parse_args()

    run_dir = (
        Path(__file__).resolve().parent.parent
        / "ml_datasets" / args.dataset / "raw_dataset" / args.run
    )
    rclpy.init()
    node = FrameFeeder(run_dir, args.n, args.hz)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
