"""Export an official-pipeline SB3 PPO policy to the tournament Agent format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def export(model_path: Path, output_directory: Path):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    if tuple(model.observation_space.shape) != (6, 84, 84):
        raise ValueError(
            f"Expected official model observation shape (6, 84, 84), got "
            f"{model.observation_space.shape}"
        )
    source = model.policy.state_dict()
    weights = {
        "encoder.0.weight": source["features_extractor.cnn.0.weight"],
        "encoder.0.bias": source["features_extractor.cnn.0.bias"],
        "encoder.2.weight": source["features_extractor.cnn.2.weight"],
        "encoder.2.bias": source["features_extractor.cnn.2.bias"],
        "encoder.4.weight": source["features_extractor.cnn.4.weight"],
        "encoder.4.bias": source["features_extractor.cnn.4.bias"],
        "projection.0.weight": source["features_extractor.linear.0.weight"],
        "projection.0.bias": source["features_extractor.linear.0.bias"],
        "action_head.weight": source["action_net.weight"],
        "action_head.bias": source["action_net.bias"],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output_directory / "weights.pt")
    template = Path(__file__).resolve().parents[1] / "submission" / "agent.py"
    shutil.copy2(template, output_directory / "agent.py")
    print(f"Exported tournament agent to {output_directory}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.model, args.output_directory)
