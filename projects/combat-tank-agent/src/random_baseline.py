"""Reproduce Combat Tank and measure random-versus-random scores."""

from __future__ import annotations

import argparse

import numpy as np

from .environment import AGENTS, create_environment


def run_episode(seed: int, render: bool = False) -> dict[str, float]:
    env = create_environment(render_mode="human" if render else None)
    _, _ = env.reset(seed=seed)
    returns = {agent: 0.0 for agent in AGENTS}
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)
        for agent, reward in rewards.items():
            returns[agent] += float(reward)
    env.close()
    return returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    results = [run_episode(args.seed + i, args.render) for i in range(args.episodes)]
    for agent in AGENTS:
        values = np.asarray([result[agent] for result in results])
        print(f"{agent}: {values.mean():.2f} +/- {values.std():.2f}")


if __name__ == "__main__":
    main()
