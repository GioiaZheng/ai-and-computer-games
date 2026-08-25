"""Export role-specialized PPO policies to one tournament Agent package."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def policy_weights(model, prefix: str):
    source = model.policy.state_dict()
    return {
        f"{prefix}_encoder.0.weight": source["features_extractor.cnn.0.weight"],
        f"{prefix}_encoder.0.bias": source["features_extractor.cnn.0.bias"],
        f"{prefix}_encoder.2.weight": source["features_extractor.cnn.2.weight"],
        f"{prefix}_encoder.2.bias": source["features_extractor.cnn.2.bias"],
        f"{prefix}_encoder.4.weight": source["features_extractor.cnn.4.weight"],
        f"{prefix}_encoder.4.bias": source["features_extractor.cnn.4.bias"],
        f"{prefix}_projection.0.weight": source["features_extractor.linear.0.weight"],
        f"{prefix}_projection.0.bias": source["features_extractor.linear.0.bias"],
        f"{prefix}_action_head.weight": source["action_net.weight"],
        f"{prefix}_action_head.bias": source["action_net.bias"],
    }


def export(first_model_path: Path, second_model_path: Path, output_directory: Path):
    from stable_baselines3 import PPO

    first_model = PPO.load(first_model_path, device="cpu")
    second_model = PPO.load(second_model_path, device="cpu")
    for role, model in (("first", first_model), ("second", second_model)):
        if tuple(model.observation_space.shape) != (6, 84, 84):
            raise ValueError(
                f"{role} model observation shape is not official: "
                f"{model.observation_space.shape}"
            )
    weights = policy_weights(first_model, "first")
    weights.update(policy_weights(second_model, "second"))
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output_directory / "weights.pt")
    template = Path(__file__).resolve().parents[1] / "submission" / "dual_agent.py"
    shutil.copy2(template, output_directory / "agent.py")
    print(f"Exported role-specialized tournament agent to {output_directory}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-model", type=Path, required=True)
    parser.add_argument("--second-model", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.first_model, args.second_model, args.output_directory)
