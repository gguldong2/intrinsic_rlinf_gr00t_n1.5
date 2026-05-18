#!/usr/bin/env python3
"""Inject ur5e statistics into the GR00T base model's metadata.json.

The GR00T-N1.5-3B base ships without a 'ur5e' embodiment entry; the fine-tuned
checkpoint was trained on UR5e data with task-specific normalization, so we
need to add ur5e statistics for inference to work.

Usage:
    python inject_ur5e_stats.py \
        --model-dir /path/to/gr00t-n1.5-3b \
        --stats-json /path/to/dataset/meta/stats.json
"""
import argparse
import json
from pathlib import Path

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "gripper",
]


def split_by_joint(arr_stats):
    out = {}
    for i, name in enumerate(UR5E_JOINT_NAMES):
        out[name] = {
            "min":  [float(arr_stats["min"][i])],
            "max":  [float(arr_stats["max"][i])],
            "mean": [float(arr_stats["mean"][i])],
            "std":  [float(arr_stats["std"][i])],
            "q01":  [float(arr_stats.get("q01", arr_stats["min"])[i])],
            "q99":  [float(arr_stats.get("q99", arr_stats["max"])[i])],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="GR00T-N1.5-3B model dir (containing experiment_cfg/metadata.json)")
    ap.add_argument("--stats-json", required=True, help="LeRobot-format stats.json from training dataset")
    args = ap.parse_args()

    meta_path = Path(args.model_dir) / "experiment_cfg" / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"metadata.json not found at {meta_path}")

    with open(args.stats_json) as f:
        stats = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    state_stats = split_by_joint(stats["observation.state"])
    action_stats = split_by_joint(stats["action"])

    if "ur5e" in meta and "modalities" in meta["ur5e"]:
        modalities = meta["ur5e"]["modalities"]
    else:
        joint_modality = {
            n: {"absolute": True, "rotation_type": None, "shape": [1], "continuous": True}
            for n in UR5E_JOINT_NAMES
        }
        modalities = {
            "video": {
                "cam_center": {"resolution": [224, 224], "channels": 3, "fps": 30.0},
                "cam_left":   {"resolution": [224, 224], "channels": 3, "fps": 30.0},
                "cam_right":  {"resolution": [224, 224], "channels": 3, "fps": 30.0},
            },
            "state":  joint_modality,
            "action": joint_modality,
        }

    meta["ur5e"] = {
        "statistics": {"state": state_stats, "action": action_stats},
        "modalities": modalities,
        "embodiment_tag": "new_embodiment",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"ur5e stats injected into {meta_path}")


if __name__ == "__main__":
    main()
