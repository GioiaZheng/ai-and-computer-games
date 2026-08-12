"""Trace trained Simple Spread policies with W&B Weave.

This script loads the course IQL or centralized Q-learning checkpoints and
creates one parent evaluation trace with one child trace per episode. It does
not call an LLM and does not retrain the policies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import weave
from mpe2 import simple_spread_v3

from dqn import QNetwork


AGENTS = ("agent_0", "agent_1")
DEFAULT_CHECKPOINTS = {
    "iql": (
        Path("checkpoints/spread_iql_agent_0.pth"),
        Path("checkpoints/spread_iql_agent_1.pth"),
    ),
    "cql": (Path("checkpoints/spread_cql.pth"),),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create W&B Weave traces for trained Simple Spread policies."
    )
    parser.add_argument("--algorithm", choices=("iql", "cql"), default="iql")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument(
        "--weave-project",
        default="gioiazheng/ai-and-computer-games",
        help="W&B entity/project used by weave.init().",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        help="Override checkpoint path. Pass twice for IQL and once for CQL.",
    )
    parser.add_argument(
        "--max-trace-steps",
        type=int,
        default=25,
        help="Maximum number of per-step records returned in each trace.",
    )
    return parser.parse_args()


def make_env():
    return simple_spread_v3.parallel_env(
        N=2,
        local_ratio=0.0,
        max_cycles=25,
        continuous_actions=False,
    )


def load_networks(algorithm, checkpoint_paths, device):
    env = make_env()
    observation_sizes = [env.observation_space(agent).shape[0] for agent in AGENTS]
    action_sizes = [env.action_space(agent).n for agent in AGENTS]
    env.close()

    if algorithm == "iql":
        if len(checkpoint_paths) != 2:
            raise ValueError("IQL requires two --checkpoint paths")
        networks = [
            QNetwork(observation_sizes[i], action_sizes[i]).to(device)
            for i in range(2)
        ]
    else:
        if len(checkpoint_paths) != 1:
            raise ValueError("CQL requires one --checkpoint path")
        networks = [
            QNetwork(sum(observation_sizes), int(np.prod(action_sizes))).to(device)
        ]

    for network, checkpoint_path in zip(networks, checkpoint_paths):
        network.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        network.eval()
    return networks, action_sizes


@weave.op()
def evaluate_episode(
    algorithm: str,
    seed: int,
    max_trace_steps: int,
) -> dict:
    """Run one greedy episode and return a compact, inspectable trajectory."""
    env = make_env()
    observations, _ = env.reset(seed=seed)
    episode_return = 0.0
    trajectory = []
    step = 0

    try:
        while env.agents:
            if algorithm == "iql":
                actions = {
                    agent: NETWORKS[i].action(observations[agent], DEVICE)
                    for i, agent in enumerate(AGENTS)
                }
            else:
                joint_observation = np.concatenate(
                    [observations[agent] for agent in AGENTS]
                )
                joint_action = NETWORKS[0].action(joint_observation, DEVICE)
                actions = {
                    "agent_0": joint_action // ACTION_SIZES[1],
                    "agent_1": joint_action % ACTION_SIZES[1],
                }

            observations_before = {
                agent: np.asarray(observations[agent]).round(5).tolist()
                for agent in AGENTS
            }
            observations, rewards, terminations, truncations, _ = env.step(actions)
            shared_reward = float(rewards["agent_0"])
            episode_return += shared_reward

            if step < max_trace_steps:
                trajectory.append(
                    {
                        "step": step,
                        "observations": observations_before,
                        "actions": {agent: int(actions[agent]) for agent in AGENTS},
                        "shared_reward": shared_reward,
                        "cumulative_return": episode_return,
                        "terminated": bool(any(terminations.values())),
                        "truncated": bool(any(truncations.values())),
                    }
                )
            step += 1
    finally:
        env.close()

    return {
        "algorithm": algorithm,
        "seed": seed,
        "episode_steps": step,
        "team_return": episode_return,
        "trajectory": trajectory,
    }


@weave.op()
def evaluate_policy(
    algorithm: str,
    episodes: int,
    first_seed: int,
    max_trace_steps: int,
    checkpoint_paths: list[str],
) -> dict:
    """Create one parent trace containing one child trace per evaluation episode."""
    episode_results = [
        evaluate_episode(algorithm, first_seed + index, max_trace_steps)
        for index in range(episodes)
    ]
    returns = np.asarray(
        [result["team_return"] for result in episode_results],
        dtype=np.float64,
    )
    return {
        "algorithm": algorithm,
        "episodes": episodes,
        "first_seed": first_seed,
        "checkpoint_paths": checkpoint_paths,
        "mean_team_return": float(np.mean(returns)),
        "std_team_return": float(np.std(returns)),
        "min_team_return": float(np.min(returns)),
        "max_team_return": float(np.max(returns)),
        "episode_results": episode_results,
    }


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_trace_steps <= 0:
        raise ValueError("--max-trace-steps must be positive")
    checkpoint_paths = tuple(args.checkpoint or DEFAULT_CHECKPOINTS[args.algorithm])
    missing = [path for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s): " + ", ".join(str(path) for path in missing)
        )

    global DEVICE, NETWORKS, ACTION_SIZES
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NETWORKS, ACTION_SIZES = load_networks(
        args.algorithm,
        checkpoint_paths,
        DEVICE,
    )

    weave.init(args.weave_project)
    result = evaluate_policy(
        algorithm=args.algorithm,
        episodes=args.episodes,
        first_seed=args.seed,
        max_trace_steps=args.max_trace_steps,
        checkpoint_paths=[str(path) for path in checkpoint_paths],
    )
    print(f"Device: {DEVICE}")
    print(
        f"{args.algorithm.upper()} over {args.episodes} traced episodes: "
        f"{result['mean_team_return']:.3f} +/- "
        f"{result['std_team_return']:.3f}"
    )


if __name__ == "__main__":
    main()
