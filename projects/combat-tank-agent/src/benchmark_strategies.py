"""Benchmark simple legal action strategies against a random opponent."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .environment import AGENTS, create_environment


ACTION_SETS = {
    "uniform_all": tuple(range(18)),
    "fire_only": (1, 10, 11, 12, 13, 14, 15, 16, 17),
    "fire_cardinal": (10, 11, 12, 13),
    "fire_diagonal": (14, 15, 16, 17),
    "move_only": (2, 3, 4, 5, 6, 7, 8, 9),
}


def run_episode(task):
    strategy, role, seed = task
    env = create_environment()
    env.reset(seed=seed)
    opponent = AGENTS[1] if role == AGENTS[0] else AGENTS[0]
    rng = np.random.default_rng(seed + 50_000)
    action_set = ACTION_SETS[strategy]
    score = 0.0
    counts = Counter()
    while env.agents:
        learner_action = int(rng.choice(action_set))
        counts[learner_action] += 1
        actions = {
            role: learner_action,
            opponent: int(env.action_space(opponent).sample()),
        }
        _, rewards, _, _, _ = env.step(actions)
        score += float(rewards.get(role, 0.0))
    env.close()
    return strategy, role, score, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-role", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=6001)
    args = parser.parse_args()

    tasks = [
        (strategy, role, args.seed + game)
        for strategy in ACTION_SETS
        for role in AGENTS
        for game in range(args.games_per_role)
    ]
    grouped = defaultdict(list)
    action_counts = defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for strategy, role, score, counts in executor.map(run_episode, tasks):
            grouped[(strategy, role)].append(score)
            action_counts[strategy].update(counts)

    for strategy in ACTION_SETS:
        combined = []
        role_text = []
        for role in AGENTS:
            values = np.asarray(grouped[(strategy, role)], dtype=np.float32)
            combined.extend(values.tolist())
            role_text.append(f"{role}={values.mean():+.2f}")
        values = np.asarray(combined, dtype=np.float32)
        print(
            f"{strategy:14s} {' '.join(role_text)} combined={values.mean():+.2f} "
            f"std={values.std():.2f} coverage={len(action_counts[strategy])}/18"
        )


if __name__ == "__main__":
    main()
