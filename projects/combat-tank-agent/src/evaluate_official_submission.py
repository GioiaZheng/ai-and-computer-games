"""Evaluate an exported tournament Agent on the exact official pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .official_environment import (
    AGENTS,
    OfficialSingleAgentCombatEnv,
    create_official_environment,
    official_tank_center,
)
from .verify_official_submission import load_agent_class


def evaluate_role(agent, role: str, args):
    scores = []
    cells_per_game = []
    idle_fractions = []
    action_counts: Counter[int] = Counter()
    for game in range(args.games_per_role):
        game_seed = args.seed + game
        torch.manual_seed(game_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(game_seed)
        env = OfficialSingleAgentCombatEnv(fixed_role=role, seed=args.seed + game)
        observation, _ = env.reset(seed=args.seed + game)
        score = 0.0
        cells: set[tuple[int, int]] = set()
        previous_center = official_tank_center(observation, role)
        idle_steps = 0
        executed_steps = 0
        for _ in range(args.max_steps):
            action = int(agent.get_action(observation))
            action_counts[action] += 1
            observation, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
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
        cells_per_game.append(len(cells))
        idle_fractions.append(idle_steps / max(executed_steps, 1))
    return scores, cells_per_game, idle_fractions, action_counts


def main(args):
    Agent = load_agent_class(args.agent_directory / "agent.py")
    probe_env = create_official_environment(render_mode=None)
    agent = Agent(probe_env)
    if hasattr(agent, "temperature"):
        agent.temperature = args.temperature
    probe_env.close()
    all_scores = []
    all_counts: Counter[int] = Counter()
    for role in AGENTS:
        scores, cells, idle, counts = evaluate_role(agent, role, args)
        all_scores.extend(scores)
        all_counts.update(counts)
        print(
            f"{role}: {np.mean(scores):+.2f} +/- {np.std(scores):.2f}; "
            f"scores={scores}; cells={np.mean(cells):.1f}; "
            f"idle={100 * np.mean(idle):.1f}%"
        )
    print(f"combined: {np.mean(all_scores):+.2f} +/- {np.std(all_scores):.2f}")
    print(f"action coverage: {len(all_counts)}/18; counts={dict(sorted(all_counts.items()))}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-directory", type=Path, required=True)
    parser.add_argument("--games-per-role", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=82_126)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
