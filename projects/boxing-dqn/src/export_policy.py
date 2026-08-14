"""Export only the policy state dict from a resumable training checkpoint."""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain model_state_dict")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"Exported policy weights: {args.output}")


if __name__ == "__main__":
    main()
