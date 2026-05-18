# Copyright 2026 The RLinf Authors / Phy-Lab-aic.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import zmq
from rclpy.node import Node

from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation as AicObservation
from aic_task_interfaces.msg import Task

# ---------------------------------------------------------------------------
# Paths — override via environment variables for Docker / different servers
# ---------------------------------------------------------------------------
_RLINF_PATH   = os.environ.get("RLINF_PATH",  "/home/graphai/project/aic/RLinf")
_RLINF_PYTHON = os.environ.get("RLINF_PYTHON", os.path.join(_RLINF_PATH, ".venv", "bin", "python"))

_BASE_MODEL_PATH = os.environ.get("GROOT_MODEL_PATH",      "/home/graphai/models/gr00t-n1.5-3b")
_CHECKPOINT_PATH = os.environ.get("GROOT_CHECKPOINT_PATH", "/home/graphai/checkpoints/gr00t-pretrained-v3/full_weights.pt")

_WORKER_SCRIPT = str(Path(__file__).parent / "gr00t_worker.py")

_INF_PORT   = int(os.environ.get("GROOT_ZMQ_PORT",   "5555"))
_READY_PORT = int(os.environ.get("GROOT_READY_PORT", "5556"))
_MODEL_LOAD_TIMEOUT = 300  # seconds to wait for model to load

# ---------------------------------------------------------------------------
# Start the worker subprocess immediately at module import time so that model
# loading overlaps with the ROS2 / engine startup delay before on_configure.
# A background thread waits for the READY signal so __init__ can return fast.
# ---------------------------------------------------------------------------

# Kill any leftover worker processes from a previous run to free GPU memory.
subprocess.run(["pkill", "-f", "gr00t_worker.py"], capture_output=True)
time.sleep(1)

_zmq_ctx      = zmq.Context()
_worker_ready = threading.Event()   # set when worker sends READY

_worker_env = os.environ.copy()
# Strip Python-path variables so the Python 3.11 worker uses only its own packages,
# not the Python 3.12 pixi site-packages inherited from the parent process.
for _k in ("PYTHONPATH", "PYTHONHOME"):
    _worker_env.pop(_k, None)
_worker_env["RLINF_PATH"]            = _RLINF_PATH
_worker_env["GROOT_MODEL_PATH"]      = _BASE_MODEL_PATH
_worker_env["GROOT_CHECKPOINT_PATH"] = _CHECKPOINT_PATH
_worker_env["GROOT_ZMQ_PORT"]        = str(_INF_PORT)
_worker_env["GROOT_READY_PORT"]      = str(_READY_PORT)

_worker_log_path = "/tmp/gr00t_worker.log"
_worker_log_file = open(_worker_log_path, "w")
_worker_proc = subprocess.Popen(
    [_RLINF_PYTHON, _WORKER_SCRIPT],
    env=_worker_env,
    stdout=_worker_log_file,
    stderr=subprocess.STDOUT,
)


def _cleanup_worker():
    try:
        _worker_proc.terminate()
        _worker_proc.wait(timeout=10)
    except Exception:
        pass
    try:
        _worker_log_file.close()
    except Exception:
        pass


atexit.register(_cleanup_worker)


def _wait_for_ready():
    ready_sock = _zmq_ctx.socket(zmq.PULL)
    ready_sock.bind(f"tcp://127.0.0.1:{_READY_PORT}")
    ready_sock.setsockopt(zmq.RCVTIMEO, _MODEL_LOAD_TIMEOUT * 1000)
    try:
        msg = ready_sock.recv()
        if msg == b"READY":
            _worker_ready.set()
    except zmq.Again:
        pass
    finally:
        ready_sock.close()


threading.Thread(target=_wait_for_ready, daemon=True).start()


class RunGR00T(Policy):
    """GR00T N1.5-3B policy served via a RLinf Python 3.11 subprocess.

    The GR00T model (and all its CUDA-compiled dependencies such as flash-attn)
    runs inside the RLinf virtual environment (Python 3.11).  This node
    communicates with that subprocess over a local ZMQ REQ/REP socket, avoiding
    any Python-version or ABI conflicts in the pixi/ROS2 environment (Python 3.12).
    """

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        # Worker subprocess and ready-watcher thread are already running.
        # Return immediately so configure completes within the engine timeout.
        self.get_logger().info("RunGR00T: worker subprocess started, model loading in background.")
        self._worker   = _worker_proc
        self._zmq_ctx  = _zmq_ctx
        self._inf_sock = _zmq_ctx.socket(zmq.REQ)
        self._inf_sock.connect(f"tcp://127.0.0.1:{_INF_PORT}")

    def __del__(self):
        try:
            self._inf_sock.send(b"SHUTDOWN")
            self._inf_sock.recv()
            self._inf_sock.close()
            self._zmq_ctx.term()
        except Exception:
            pass
        try:
            self._worker.terminate()
            self._worker.wait(timeout=10)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _ros_img_to_np(img_msg) -> np.ndarray:
        return np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )

    def _obs_to_dict(self, obs_msg: AicObservation, task_desc: str) -> dict:
        joints = np.array(obs_msg.joint_states.position[:7], dtype=np.float32)
        return {
            "center_image":    self._ros_img_to_np(obs_msg.center_image),
            "left_image":      self._ros_img_to_np(obs_msg.left_image),
            "right_image":     self._ros_img_to_np(obs_msg.right_image),
            "joint_positions": joints,
            "task_description": task_desc,
        }

    @staticmethod
    def _task_to_instruction(task: Task) -> str:
        # Reproduce the training-data instruction format from the Task fields.
        # Training data only contains the "SFP-to-SC" cable type across all 12 tasks,
        # so we use it as a fixed label rather than deriving from task.cable_name
        # (which is an instance ID like "cable_0", not the cable type).
        cable_label = "SFP-to-SC"
        plug_tip    = f"{task.plug_type}_tip" if task.plug_type else task.plug_name
        return (
            f"Insert the {cable_label} cable's {plug_tip} "
            f"into {task.port_name} on {task.target_module_name}."
        )

    # ------------------------------------------------------------------
    # Policy entry point
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        # Recreate the REQ socket each call so stale state from a prior (cancelled)
        # call never corrupts the send/recv handshake.
        try:
            self._inf_sock.close(linger=0)
        except Exception:
            pass
        self._inf_sock = _zmq_ctx.socket(zmq.REQ)
        self._inf_sock.setsockopt(zmq.RCVTIMEO, 60_000)  # 60 s max per inference
        self._inf_sock.connect(f"tcp://127.0.0.1:{_INF_PORT}")

        self.get_logger().info("RunGR00T.insert_cable() start")

        # Wait for model to finish loading before starting the task timer.
        if not _worker_ready.is_set():
            self.get_logger().info("RunGR00T: waiting for model to finish loading ...")
            deadline = time.time() + _MODEL_LOAD_TIMEOUT
            while not _worker_ready.is_set():
                if _worker_proc.poll() is not None:
                    raise RuntimeError(
                        f"GR00T worker exited (code {_worker_proc.returncode}) before becoming ready. "
                        f"Check {_worker_log_path}"
                    )
                if time.time() > deadline:
                    raise RuntimeError(
                        f"GR00T worker did not become ready within {_MODEL_LOAD_TIMEOUT}s. "
                        f"Check {_worker_log_path}"
                    )
                _worker_ready.wait(timeout=5)
            self.get_logger().info("RunGR00T: model ready, starting task.")

        start    = time.time()
        task_desc = self._task_to_instruction(task)
        self.get_logger().info(f"RunGR00T: task instruction = {task_desc!r}")
        step_dt   = 1.0 / 10.0  # 10 Hz execution (A/B: reverted from 20 Hz)

        while time.time() - start < 30.0:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warning("No observation received, skipping.")
                continue

            obs = self._obs_to_dict(obs_msg, task_desc)

            # Send observation to worker, receive action chunks
            try:
                self._inf_sock.send(pickle.dumps(obs))
                joint_actions = pickle.loads(self._inf_sock.recv())
            except zmq.Again:
                self.get_logger().error("RunGR00T: inference timed out (>60 s), aborting")
                break
            except zmq.ZMQError as e:
                self.get_logger().error(f"RunGR00T: ZMQ error during inference: {e}")
                break

            if joint_actions is None:
                self.get_logger().error("RunGR00T: worker inference error, skipping step")
                continue

            # Execute each step of the action chunk at step_dt intervals.
            # The aic_controller's JointMotionUpdate subscription expects
            # 6 arm joints (gripper is on a separate hardware/topic) and
            # requires non-empty stiffness/damping arrays to actuate.
            step_start = time.time()
            joint_msg = JointMotionUpdate(
                target_stiffness=[200.0, 200.0, 200.0, 50.0, 50.0, 50.0],
                target_damping=[40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
                trajectory_generation_mode=TrajectoryGenerationMode(
                    mode=TrajectoryGenerationMode.MODE_POSITION
                ),
            )
            for i, joint_target in enumerate(joint_actions):
                joint_msg.target_state.positions = joint_target[:6].tolist()
                move_robot(joint_motion_update=joint_msg)
                deadline = step_start + (i + 1) * step_dt
                sleep_time = deadline - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            send_feedback("GR00T in progress")

        self.get_logger().info("RunGR00T.insert_cable() done")
        return True
