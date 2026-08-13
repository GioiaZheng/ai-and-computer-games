"""Evaluate a Day 4 Boxing checkpoint on fixed seeds / 固定种子评估 Boxing。"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from dqn_boxing import DQN, evaluate


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(opponent, returns):
    values = np.asarray(returns, dtype=np.float32)
    return {
        "opponent": opponent,
        "episodes": len(returns),
        "mean_return": round(float(values.mean()), 4),
        "std_return": round(float(values.std()), 4),
        "min_return": round(float(values.min()), 4),
        "max_return": round(float(values.max()), 4),
        "win_rate": round(float(np.mean(values > 0)), 4),
        "draw_rate": round(float(np.mean(values == 0)), 4),
    }


def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Expected a Day 4 format-version 2 checkpoint.")

    args.frame_stack = int(checkpoint["frame_stack"])
    n_actions = int(checkpoint["n_actions"])
    policy_net = DQN(args.frame_stack, n_actions).to(device)
    opponent_net = DQN(args.frame_stack, n_actions).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    opponent_net.load_state_dict(checkpoint["opponent_state_dict"])
    policy_net.eval()
    opponent_net.eval()

    rows = []
    opponents = ["random"]
    if checkpoint.get("opponent_ready", False):
        opponents.append("snapshot")
    for opponent in opponents:
        _, _, returns = evaluate(
            policy_net,
            opponent_net,
            args,
            device,
            n_actions,
            opponent,
            args.eval_seed,
        )
        row = summarize(opponent, returns)
        rows.append(row)
        print(
            f"opponent={opponent:8s} mean={row['mean_return']:7.2f} "
            f"std={row['std_return']:6.2f} win_rate={row['win_rate']:.2%}"
        )

    write_rows(Path(args.output), rows)
    print(f"Evaluation CSV / 评估结果: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Day 4 Boxing agent.")
    parser.add_argument("--checkpoint", default="checkpoints/day4_boxing_best.pt")
    parser.add_argument("--output", default="results/day4_boxing_final_evaluation.csv")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20000)
    parser.add_argument("--eval-opponent-epsilon", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
