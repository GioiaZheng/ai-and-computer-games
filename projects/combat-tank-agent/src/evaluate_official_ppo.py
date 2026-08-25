"""Evaluate an SB3 PPO checkpoint on the exact tournament observation pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .official_environment import (
    AGENTS,
    OfficialSingleAgentCombatEnv,
    official_tank_center,
)
from .official_opponents import OfficialPPOOpponent


def evaluate_role(model, role: str, args):
    scores = []
    cell_counts = []
    idle_fractions = []
    action_counts: Counter[int] = Counter()
    for game in range(args.games_per_role):
        opponent_policy = None
        if args.opponent_model is not None:
            opponent_policy = OfficialPPOOpponent(
                args.opponent_model,
                device=args.opponent_device,
                deterministic=args.opponent_deterministic,
            )
        env = OfficialSingleAgentCombatEnv(
            opponent_policy=opponent_policy,
            fixed_role=role,
            seed=args.seed + game,
        )
        observation, _ = env.reset(seed=args.seed + game)
        score = 0.0
        cells: set[tuple[int, int]] = set()
        previous_center = official_tank_center(observation, role)
        idle_steps = 0
        executed_steps = 0
        for _ in range(args.max_steps):
            action, _ = model.predict(
                observation,
                deterministic=args.deterministic,
            )
            action = int(np.asarray(action).item())
            action_counts[action] += 1
            observation, reward, terminated, truncated, _ = env.step(action)
            score += reward
            executed_steps += 1
            center = official_tank_center(observation, role)
            if center is not None:
                cells.add((int(center[0] // 7), int(center[1] // 7)))
                if previous_center is not None:
                    movement = abs(center[0] - previous_center[0]) + abs(
                        center[1] - previous_center[1]
                    )
                    idle_steps += int(movement < 0.5)
                previous_center = center
            if terminated or truncated:
                break
        env.close()
        scores.append(score)
        cell_counts.append(len(cells))
        idle_fractions.append(idle_steps / max(executed_steps, 1))
    return scores, action_counts, cell_counts, idle_fractions


def main(args):
    from stable_baselines3 import PPO

    model = PPO.load(args.model, device=args.device)
    if tuple(model.observation_space.shape) != (6, 84, 84):
        raise ValueError(
            "This checkpoint was not trained on the official (84, 84, 6) pipeline: "
            f"model space is {model.observation_space}"
        )
    all_counts: Counter[int] = Counter()
    combined = []
    roles = (args.role,) if args.role is not None else AGENTS
    for role in roles:
        scores, counts, cell_counts, idle_fractions = evaluate_role(model, role, args)
        combined.extend(scores)
        all_counts.update(counts)
        print(
            f"{role}: {np.mean(scores):+.2f} +/- {np.std(scores):.2f}; "
            f"scores={scores}; cells={np.mean(cell_counts):.1f}; "
            f"idle={100 * np.mean(idle_fractions):.1f}%"
        )
    print(f"combined: {np.mean(combined):+.2f} +/- {np.std(combined):.2f}")
    print(f"action coverage: {len(all_counts)}/18; counts={dict(sorted(all_counts.items()))}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--games-per-role", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=82_126)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--role", choices=AGENTS)
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument("--opponent-deterministic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
