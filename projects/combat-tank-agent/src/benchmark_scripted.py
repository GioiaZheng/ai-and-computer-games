"""Benchmark scripted maze-entry curricula against a random opponent."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .environment import AGENTS, create_environment
from .scripted_agent import MazeRushAgent, SpawnHunterAgent


def run_episode(task):
    strategy, role, seed, vertical_frames = task
    env = create_environment()
    env.reset(seed=seed)
    opponent = AGENTS[1] if role == AGENTS[0] else AGENTS[0]
    agent_class = MazeRushAgent if strategy == "rush" else SpawnHunterAgent
    learner = agent_class(role=role, vertical_frames=vertical_frames)
    score = 0.0
    steps = 0
    while env.agents:
        actions = {
            role: learner.get_action(),
            opponent: int(env.action_space(opponent).sample()),
        }
        _, rewards, _, _, _ = env.step(actions)
        reward = float(rewards.get(role, 0.0))
        if isinstance(learner, SpawnHunterAgent):
            learner.observe_reward(reward)
        score += reward
        steps += 1
    env.close()
    return strategy, vertical_frames, role, score, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-role", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=8001)
    parser.add_argument("--strategies", nargs="+", choices=("rush", "hunter"), default=("hunter",))
    parser.add_argument(
        "--vertical-frames",
        type=int,
        nargs="+",
        default=(380, 400, 420, 440),
    )
    args = parser.parse_args()

    tasks = [
        (strategy, role, args.seed + game, vertical_frames)
        for strategy in args.strategies
        for vertical_frames in args.vertical_frames
        for role in AGENTS
        for game in range(args.games_per_role)
    ]
    grouped = {
        (strategy, frames, role): []
        for strategy in args.strategies
        for frames in args.vertical_frames
        for role in AGENTS
    }
    lengths = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for strategy, frames, role, score, steps in executor.map(run_episode, tasks):
            grouped[(strategy, frames, role)].append(score)
            lengths.append(steps)

    print(f"episode_steps={min(lengths)}..{max(lengths)}")
    for strategy in args.strategies:
        for frames in args.vertical_frames:
            combined = []
            role_results = []
            for role in AGENTS:
                scores = np.asarray(grouped[(strategy, frames, role)], dtype=np.float32)
                combined.extend(scores.tolist())
                role_results.append(
                    f"{role}={scores.mean():+.2f} wins={np.count_nonzero(scores > 0)}/{len(scores)}"
                )
            scores = np.asarray(combined, dtype=np.float32)
            print(
                f"{strategy:6s} vertical={frames:3d} {' '.join(role_results)} "
                f"combined={scores.mean():+.2f} std={scores.std():.2f}"
            )


if __name__ == "__main__":
    main()
