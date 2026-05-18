#!/usr/bin/env python3
# GR00T inference worker — runs under RLinf Python 3.11 as a subprocess.
# Communication with the ROS2 policy node (Python 3.12) is via ZMQ REQ/REP.
#
# Environment variables (all optional, have defaults):
#   RLINF_PATH            path to RLinf repo root
#   GROOT_MODEL_PATH      path to base GR00T-N1.5-3B checkpoint directory
#   GROOT_CHECKPOINT_PATH path to fine-tuned full_weights.pt (empty = skip)
#   GROOT_ZMQ_PORT        TCP port for inference socket   (default 5555)
#   GROOT_READY_PORT      TCP port for ready-notification (default 5556)

import os
import pickle
import sys
from pathlib import Path

_RLINF_PATH  = os.environ.get("RLINF_PATH",  "/home/graphai/project/aic/RLinf")
_GROOT_PATH  = os.path.join(_RLINF_PATH, ".venv", "gr00t")
_INF_PORT    = int(os.environ.get("GROOT_ZMQ_PORT",   "5555"))
_READY_PORT  = int(os.environ.get("GROOT_READY_PORT", "5556"))

_BASE_MODEL  = os.environ.get("GROOT_MODEL_PATH",      "/home/graphai/models/gr00t-n1.5-3b")
_CKPT_PATH   = os.environ.get("GROOT_CHECKPOINT_PATH", "")

for _p in [_GROOT_PATH, _RLINF_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import traceback

import zmq
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image as _PILImage

from rlinf.models.embodiment.gr00t import get_model
from rlinf.models.embodiment.gr00t.simulation_io import (
    convert_aic_obs_to_gr00t_format,
    convert_to_aic_joint_actions,
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


def _resize_images(obs: dict, size: int = 224) -> dict:
    out = obs.copy()
    for key in ("center_image", "left_image", "right_image"):
        if key in out:
            img = out[key]  # (H, W, 3) uint8
            pil = _PILImage.fromarray(img)
            pil = pil.resize((size, size), _PILImage.BILINEAR)
            out[key] = np.array(pil)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[worker] loading model on {device} ...", flush=True)

    cfg = OmegaConf.create({
        "model_path":       _BASE_MODEL,
        "embodiment_tag":   "ur5e",
        "obs_converter_type": "aic",
        "action_dim":       7,
        "num_action_chunks": 16,
        "denoising_steps":  4,
        "dataset_path":     None,
        "rl_head_config":   _RL_HEAD_CFG,
    })

    model = get_model(cfg)

    if _CKPT_PATH:
        ckpt = Path(_CKPT_PATH)
        if ckpt.exists():
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[worker] checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        else:
            print(f"[worker] checkpoint not found at {ckpt}, running base model", flush=True)

    model.eval()
    model.to(device)
    print("[worker] model ready", flush=True)

    # Warmup inference — compiles CUDA kernels so the first real call is fast.
    print("[worker] running warmup inference ...", flush=True)
    try:
        _dummy = {
            "center_image":    np.zeros((224, 224, 3), dtype=np.uint8),
            "left_image":      np.zeros((224, 224, 3), dtype=np.uint8),
            "right_image":     np.zeros((224, 224, 3), dtype=np.uint8),
            "joint_positions": np.zeros(7, dtype=np.float32),
            "task_description": "insert cable",
        }
        with torch.no_grad():
            _g = convert_aic_obs_to_gr00t_format(_dummy)
            _n = model.apply_transforms(_g)
            _a = model._get_action_from_normalized_input(_n)
            _  = model._get_unnormalized_action(_a)
        del _dummy, _g, _n, _a, _
        print("[worker] warmup done", flush=True)
    except Exception as _e:
        print(f"[worker] warmup failed (non-fatal): {_e}", flush=True)

    ctx = zmq.Context()

    # Inference socket (REP)
    inf_sock = ctx.socket(zmq.REP)
    inf_sock.bind(f"tcp://127.0.0.1:{_INF_PORT}")

    # Ready notification socket (PUSH)
    ready_sock = ctx.socket(zmq.PUSH)
    ready_sock.connect(f"tcp://127.0.0.1:{_READY_PORT}")
    ready_sock.send(b"READY")
    ready_sock.close()

    print(f"[worker] listening on port {_INF_PORT}", flush=True)
    _debug_saved = False  # save first observation images for inspection

    while True:
        msg = inf_sock.recv()

        if msg == b"SHUTDOWN":
            inf_sock.send(b"OK")
            break

        try:
            obs = pickle.loads(msg)
            # Save first observation for camera inspection
            if not _debug_saved:
                _debug_saved = True
                try:
                    for key in ("center_image", "left_image", "right_image"):
                        if key in obs:
                            _PILImage.fromarray(obs[key]).save(f"/tmp/debug_obs_{key}_raw.png")
                    print("[worker] saved raw observation images to /tmp/debug_obs_*.png", flush=True)
                except Exception:
                    pass
            obs = _resize_images(obs)
            with torch.no_grad():
                groot_obs = convert_aic_obs_to_gr00t_format(obs)
                normalized_input  = model.apply_transforms(groot_obs)
                normalized_action = model._get_action_from_normalized_input(normalized_input)
                action_dict       = model._get_unnormalized_action(normalized_action)
                joint_actions     = convert_to_aic_joint_actions(action_dict, chunk_size=16)
            # DEBUG: log first/last action and delta vs current joint positions
            try:
                cur = obs.get("joint_positions")
                first = joint_actions[0]
                last  = joint_actions[-1]
                delta = (last - cur) if cur is not None else None
                print(f"[worker] cur={cur} first={first} last={last} delta_last={delta}", flush=True)
            except Exception:
                pass
            inf_sock.send(pickle.dumps(joint_actions))
        except Exception as e:
            print(f"[worker] inference error: {e}", flush=True)
            traceback.print_exc()
            inf_sock.send(pickle.dumps(None))  # must reply to keep REP socket valid

    inf_sock.close()
    ctx.term()
    print("[worker] shutdown complete", flush=True)


if __name__ == "__main__":
    main()
