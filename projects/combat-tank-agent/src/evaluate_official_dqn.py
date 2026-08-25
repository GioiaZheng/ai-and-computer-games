"""Evaluate a six-channel Nature-DQN checkpoint in the official environment."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .official_dqn import load_official_dqn
from .official_environment import AGENTS, create_official_environment


def evaluate_role(model, role: str, args):
    opponent = AGENTS[1] if role == AGENTS[0] else AGENTS[0]
    scores = []
    counts: Counter[int] = Counter()
    for game in range(args.games_per_role):
        env = create_official_environment(render_mode=None)
        observations, _ = env.reset(seed=args.seed + game)
        score = 0.0
        for _ in range(args.max_steps):
            action = model.get_action(observations[role])
            counts[action] += 1
            actions = {
                role: action,
                opponent: int(env.action_space(opponent).sample()),
            }
            observations, rewards, terminations, truncations, _ = env.step(actions)
            score += float(rewards.get(role, 0.0))
            if terminations.get(role, False) or truncations.get(role, False):
                break
        env.close()
        scores.append(score)
    return scores, counts


def main(args):
    model = load_official_dqn(args.model, device=args.device)
    combined = []
    total_counts: Counter[int] = Counter()
    for role in AGENTS:
        scores, counts = evaluate_role(model, role, args)
        combined.extend(scores)
        total_counts.update(counts)
        print(
            f"{role}: {np.mean(scores):+.2f} +/- {np.std(scores):.2f}; "
            f"scores={scores}"
        )
    print(f"combined: {np.mean(combined):+.2f} +/- {np.std(combined):.2f}")
    print(
        f"action coverage: {len(total_counts)}/18; "
        f"counts={dict(sorted(total_counts.items()))}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--games-per-role", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=82_326)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
