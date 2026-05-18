#!/usr/bin/env python3
"""
Run this in a SEPARATE terminal before starting evaluation:
  cd ~/ws_aic/src/aic
  pixi run python3 /path/to/record_trial_videos.py

Records /center_camera/image frames, splits by trial, saves MP4s to ~/aic_results/videos/.
Press Ctrl+C after all trials finish.
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String


OUTPUT_DIR = Path(os.path.expanduser(
    os.environ.get("AIC_VIDEO_DIR", "~/aic_results/videos")
))
TOPIC = "/center_camera/image"
FPS = 10  # target FPS for output video (camera may be higher; we downsample)


class TrialVideoRecorder(Node):
    def __init__(self):
        super().__init__("trial_video_recorder")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.frames: list[tuple[float, np.ndarray]] = []  # (timestamp, frame)
        self.trial_boundaries: list[tuple[str, float, float | None]] = []
        self.current_trial: str | None = None
        self.trial_start_time: float | None = None
        self.trial_counter = 0

        # Subscribe to camera
        self.image_sub = self.create_subscription(
            RosImage, TOPIC, self._image_cb, 10
        )

        # Monitor /aic_model/status or lifecycle transitions by watching for
        # insert_cable activity via joint_commands topic timing heuristic
        from aic_control_interfaces.msg import JointMotionUpdate
        self.cmd_sub = self.create_subscription(
            JointMotionUpdate,
            "/aic_controller/joint_commands",
            self._cmd_cb,
            10,
        )

        self._last_cmd_time: float = 0.0
        self._trial_active = False
        self._idle_timer = None

        self.get_logger().info(
            f"Recording {TOPIC} → {OUTPUT_DIR}  (waiting for robot activity...)"
        )

    # ------------------------------------------------------------------
    def _image_cb(self, msg: RosImage):
        # Convert ROS Image to numpy BGR
        try:
            encoding = msg.encoding.lower()
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if encoding in ("rgb8", "bgr8"):
                frame = data.reshape((msg.height, msg.width, 3))
                if encoding == "rgb8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif encoding == "mono8":
                frame = data.reshape((msg.height, msg.width))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                return  # unsupported encoding
        except Exception as e:
            self.get_logger().warn(f"Image decode error: {e}", throttle_duration_sec=5.0)
            return

        ts = time.time()
        with self.lock:
            if self._trial_active:
                self.frames.append((ts, frame))

    # ------------------------------------------------------------------
    def _cmd_cb(self, msg):
        """Heuristic: robot is active when joint commands arrive."""
        now = time.time()
        with self.lock:
            self._last_cmd_time = now
            if not self._trial_active:
                self._trial_active = True
                self.trial_counter += 1
                trial_id = f"trial_{self.trial_counter}"
                self.current_trial = trial_id
                self.trial_start_time = now
                self.get_logger().info(f"▶  Trial {trial_id} started — recording")

        # Schedule idle check (trial ends when no commands for 2 s)
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(2.0, self._check_idle)
        self._idle_timer.start()

    def _check_idle(self):
        with self.lock:
            if not self._trial_active:
                return
            if time.time() - self._last_cmd_time < 2.0:
                return  # still active
            self._trial_active = False
            trial_id = self.current_trial
            frames = list(self.frames)
            self.frames.clear()

        if frames:
            self.get_logger().info(
                f"■  Trial {trial_id} ended — {len(frames)} frames → saving MP4"
            )
            threading.Thread(
                target=self._save_video, args=(trial_id, frames), daemon=True
            ).start()
        else:
            self.get_logger().warn(f"Trial {trial_id} ended but no frames captured")

    # ------------------------------------------------------------------
    def _save_video(self, trial_id: str, frames: list[tuple[float, np.ndarray]]):
        if not frames:
            return

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{trial_id}_{ts_str}.mp4"

        h, w = frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (w, h))

        # Downsample to target FPS based on actual timestamps
        if len(frames) > 1:
            t0 = frames[0][0]
            step = 1.0 / FPS
            next_t = t0
            for ts, frame in frames:
                if ts >= next_t:
                    writer.write(frame)
                    next_t += step
        else:
            writer.write(frames[0][1])

        writer.release()
        size_mb = out_path.stat().st_size / 1e6
        self.get_logger().info(f"✓  Saved: {out_path}  ({size_mb:.1f} MB)")


def main():
    rclpy.init()
    node = TrialVideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
