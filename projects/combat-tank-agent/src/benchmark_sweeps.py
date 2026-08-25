"""Compare post-navigation firing patterns against random opponents."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .environment import AGENTS, create_environment
from .scripted_agent import SpawnHunterAgent


PATTERNS = {
    "constant_11": ((11,), 96),
    "constant_12": ((12,), 96),
    "alternate_24": ((11, 12), 24),
    "alternate_48": ((11, 12), 48),
    "alternate_96": ((11, 12), 96),
    "alternate_192": ((11, 12), 192),
    "bias_11": ((11, 11, 12), 96),
    "bias_12": ((11, 12, 12), 96),
}


def run_episode(task):
    name, role, seed = task
    sweep_actions, sweep_block = PATTERNS[name]
    agent = SpawnHunterAgent(
        role=role,
        sweep_actions=sweep_actions,
        sweep_block=sweep_block,
    )
    env = create_environment()
    env.reset(seed=seed)
    opponent = AGENTS[1] if role == AGENTS[0] else AGENTS[0]
    score = 0.0
    while env.agents:
        _, rewards, _, _, _ = env.step(
            {
                role: agent.get_action(),
                opponent: int(env.action_space(opponent).sample()),
            }
        )
        reward = float(rewards.get(role, 0.0))
        agent.observe_reward(reward)
        score += reward
    env.close()
    return name, role, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-role", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=35001)
    parser.add_argument("--patterns", nargs="+", choices=PATTERNS, default=tuple(PATTERNS))
    args = parser.parse_args()

    tasks = [
        (name, role, args.seed + game)
        for name in args.patterns
        for role in AGENTS
        for game in range(args.games_per_role)
    ]
    scores = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for name, role, score in executor.map(run_episode, tasks):
            scores[(name, role)].append(score)

    for name in args.patterns:
        combined = []
        details = []
        for role in AGENTS:
            values = np.asarray(scores[(name, role)], dtype=np.float32)
            combined.extend(values.tolist())
            details.append(f"{role}={values.mean():+.2f}")
        values = np.asarray(combined, dtype=np.float32)
        print(
            f"{name:13s} {' '.join(details)} combined={values.mean():+.2f} "
            f"std={values.std():.2f} wins={np.count_nonzero(values > 0)}/{len(values)}"
        )


if __name__ == "__main__":
    main()
