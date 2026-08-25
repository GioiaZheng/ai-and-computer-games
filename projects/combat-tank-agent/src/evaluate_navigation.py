"""Measure maze coverage without changing the official Combat Tank game."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .environment import ACTION_SETS, AGENTS, SingleAgentCombatEnv


def evaluate(
    model,
    role: str,
    games: int,
    seed: int,
    deterministic: bool,
    action_set: str,
    render: bool,
):
    coverages = []
    idle_fractions = []
    action_counts = Counter()
    scores = []
    for game in range(games):
        env = SingleAgentCombatEnv(
            fixed_role=role,
            learner_actions=ACTION_SETS[action_set],
            # Enable position bookkeeping while a zero scale guarantees that
            # the returned reward remains exactly the official game reward.
            reward_shaping=True,
            shaping_scale=0.0,
            seed=seed + game,
            render_mode="human" if render else None,
        )
        observation, _ = env.reset(seed=seed + game)
        cells = set()
        idle_steps = 0
        steps = 0
        score = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            policy_action, _ = model.predict(
                observation, deterministic=deterministic
            )
            policy_action = int(np.asarray(policy_action).item())
            game_action = ACTION_SETS[action_set][policy_action]
            action_counts[game_action] += 1
            observation, reward, terminated, truncated, info = env.step(
                policy_action
            )
            if "position_cell" in info:
                cells.add(tuple(info["position_cell"]))
            idle_steps += int(info.get("idle_steps", 0) > 0)
            score += float(info.get("official_reward", reward))
            steps += 1
        env.close()
        coverages.append(len(cells))
        idle_fractions.append(idle_steps / max(steps, 1))
        scores.append(score)
    return np.asarray(coverages), np.asarray(idle_fractions), action_counts, scores


def main():
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--games-per-role", type=int, default=3)
    parser.add_argument("--seed", type=int, default=200828)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--action-set", choices=ACTION_SETS, default="all")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    model = PPO.load(args.model, device=args.device)
    total_actions = Counter()
    for role in AGENTS:
        coverage, idle, actions, scores = evaluate(
            model,
            role,
            args.games_per_role,
            args.seed,
            args.deterministic,
            args.action_set,
            args.render,
        )
        total_actions.update(actions)
        print(
            f"{role}: cells={coverage.mean():.1f}+/-{coverage.std():.1f}; "
            f"idle={idle.mean():.1%}; official_score={np.mean(scores):.2f}"
        )
    print(
        f"action coverage={len(total_actions)}/18; allowed={ACTION_SETS[args.action_set]}; "
        f"counts={dict(sorted(total_actions.items()))}"
    )


if __name__ == "__main__":
    main()
