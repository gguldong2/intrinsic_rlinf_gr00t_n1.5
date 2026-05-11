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

import sys
import time
from pathlib import Path

import numpy as np
import torch
from rclpy.node import Node

# Make rlinf and gr00t packages importable from the RLinf project environment.
_RLINF_PATH = "/home/elicer/project/RLinf"
if _RLINF_PATH not in sys.path:
    sys.path.insert(0, _RLINF_PATH)

from omegaconf import OmegaConf

from aic_control_interfaces.msg import JointMotionUpdate
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation as AicObservation
from aic_task_interfaces.msg import Task

from rlinf.models.embodiment.gr00t import get_model
from rlinf.models.embodiment.gr00t.simulation_io import (
    convert_aic_obs_to_gr00t_format,
    convert_to_aic_joint_actions,
)

# ---------------------------------------------------------------------------
# Paths — adjust if model / checkpoint live elsewhere
# ---------------------------------------------------------------------------
_BASE_MODEL_PATH = "/data/models/gr00t-n1.5-3b"
_CHECKPOINT_PATH = (
    "/home/elicer/project/RLinf/logs/20260507-00:06:13"
    "/ur5e_sft_gr00t/checkpoints/global_step_3000"
    "/actor/model_state_dict/full_weights.pt"
)

_RL_HEAD_CFG = {
    "add_value_head": False,
    "joint_logprob": False,
    "noise_method": "flow_sde",
    "ignore_last": False,
    "safe_get_logprob": False,
    "noise_anneal": False,
    "noise_params": [0.7, 0.3, 400],
    "noise_level": 0.5,
    "chunk_critic_input": False,
    "detach_critic_input": True,
    "disable_dropout": True,
    "use_vlm_value": False,
    "value_vlm_mode": "mean_token",
    "padding_value": 570,
}


class RunGR00T(Policy):
    """GR00T N1.5-3B policy fine-tuned on UR5e joint-space data.

    Loads the base GR00T checkpoint and overlays the SFT-fine-tuned weights,
    then runs flow-matching inference to produce 16-step joint-angle chunks.
    Actions are sent as JointMotionUpdate commands at ~10 Hz.

    Note: The model was trained on UR5e pick-and-place tasks, not cable
    insertion.  Inference will run correctly, but task quality depends on
    how well the training domain transfers to cable insertion.
    """

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"RunGR00T: loading model on {self.device} ...")

        cfg = OmegaConf.create(
            {
                "model_path": _BASE_MODEL_PATH,
                "embodiment_tag": "ur5e",
                "obs_converter_type": "aic",
                "action_dim": 7,
                "num_action_chunks": 16,
                "denoising_steps": 4,
                "dataset_path": None,
                "rl_head_config": _RL_HEAD_CFG,
            }
        )

        self.model = get_model(cfg)

        ckpt_path = Path(_CHECKPOINT_PATH)
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self.get_logger().info(
                f"Fine-tuned weights loaded: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        else:
            self.get_logger().warning(
                f"Checkpoint not found at {ckpt_path}. Running base model only."
            )

        self.model.eval()
        self.model.to(self.device)
        self.get_logger().info("RunGR00T: model ready.")

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
            "center_image": self._ros_img_to_np(obs_msg.center_image),
            "left_image": self._ros_img_to_np(obs_msg.left_image),
            "right_image": self._ros_img_to_np(obs_msg.right_image),
            "joint_positions": joints,
            "task_description": task_desc,
        }

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
        self.get_logger().info("RunGR00T.insert_cable() start")
        start = time.time()
        task_desc = "insert cable"
        chunk_size = 16
        step_dt = 1.0 / 10.0  # 10 Hz execution

        while time.time() - start < 30.0:
            loop_start = time.time()

            obs_msg = get_observation()
            if obs_msg is None:
                self.get_logger().warning("No observation received, skipping.")
                continue

            # Convert ROS observation → GR00T input dict
            aic_obs = self._obs_to_dict(obs_msg, task_desc)
            groot_obs = convert_aic_obs_to_gr00t_format(aic_obs)

            # Apply modality transform and run inference
            normalized_input = self.model.apply_transforms(groot_obs)
            normalized_action = self.model._get_action_from_normalized_input(normalized_input)
            action_dict = self.model._get_unnormalized_action(normalized_action)

            # Extract (chunk_size, 7) joint angle targets
            joint_actions = convert_to_aic_joint_actions(action_dict, chunk_size=chunk_size)

            # Execute each step of the action chunk
            for joint_target in joint_actions:
                joint_msg = JointMotionUpdate()
                joint_msg.position = joint_target.tolist()
                move_robot(joint_motion_update=joint_msg)
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, step_dt - elapsed % step_dt))

            send_feedback("GR00T in progress")

        self.get_logger().info("RunGR00T.insert_cable() done")
        return True
