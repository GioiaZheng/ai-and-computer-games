"""Match exported tournament weights to their source PPO checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO


EXPORT_TO_PPO = {
    "encoder.0.weight": "features_extractor.cnn.0.weight",
    "encoder.0.bias": "features_extractor.cnn.0.bias",
    "encoder.2.weight": "features_extractor.cnn.2.weight",
    "encoder.2.bias": "features_extractor.cnn.2.bias",
    "encoder.4.weight": "features_extractor.cnn.4.weight",
    "encoder.4.bias": "features_extractor.cnn.4.bias",
    "projection.0.weight": "features_extractor.linear.0.weight",
    "projection.0.bias": "features_extractor.linear.0.bias",
    "action_head.weight": "action_net.weight",
    "action_head.bias": "action_net.bias",
}


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def role_weights(weights: dict[str, torch.Tensor], role: str | None):
    if role is None:
        return weights
    prefix = f"{role}_"
    return {
        key.removeprefix(prefix): value
        for key, value in weights.items()
        if key.startswith(prefix)
    }


def exact_match(exported: dict[str, torch.Tensor], checkpoint: Path) -> bool:
    model = PPO.load(checkpoint, device="cpu")
    source = model.policy.state_dict()
    return all(
        torch.equal(exported[export_key], source[ppo_key])
        for export_key, ppo_key in EXPORT_TO_PPO.items()
    )


def exported_targets(paths: list[Path]) -> dict[str, dict[str, torch.Tensor]]:
    targets = {}
    for path in paths:
        weights = load_weights(path)
        if "encoder.0.weight" in weights:
            targets[str(path)] = weights
            continue
        for role in ("first", "second"):
            selected = role_weights(weights, role)
            if selected:
                targets[f"{path} [{role}]"] = selected
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, nargs="+")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    targets = exported_targets(args.weights)
    for name, exported in targets.items():
        missing = set(EXPORT_TO_PPO).difference(exported)
        if missing:
            raise ValueError(f"{name} is missing expected tensors: {sorted(missing)}")

    matches = {name: [] for name in targets}
    candidates = sorted(args.checkpoints.rglob("*.zip"))
    for index, checkpoint in enumerate(candidates, start=1):
        try:
            model = PPO.load(checkpoint, device="cpu")
            source = model.policy.state_dict()
        except (KeyError, ValueError, RuntimeError):
            continue
        for name, exported in targets.items():
            if all(
                torch.equal(exported[export_key], source[ppo_key])
                for export_key, ppo_key in EXPORT_TO_PPO.items()
            ):
                matches[name].append(checkpoint)
        if index % 25 == 0:
            print(f"checked {index}/{len(candidates)}", flush=True)

    for name, target_matches in matches.items():
        print(f"\n{name}")
        if target_matches:
            for match in target_matches:
                print(f"  {match}")
        else:
            print("  no exact source checkpoint found")


if __name__ == "__main__":
    main()
