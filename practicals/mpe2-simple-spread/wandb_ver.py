"""Weights & Biases version of the Simple Spread IQL/CQL practical.

The original course scripts remain unchanged. This file adds experiment
tracking, command-line configuration, evaluation, and model artifacts while
using the same environment and DQN components.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import wandb
from mpe2 import simple_spread_v3

from dqn import QNetwork, ReplayBuffer, train_dqn, update_target


AGENTS = ("agent_0", "agent_1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Simple Spread IQL or centralized Q-learning with W&B."
    )
    parser.add_argument(
        "--algorithm",
        choices=("iql", "cql"),
        default="iql",
        help="IQL uses one learner per agent; CQL uses one joint-action learner.",
    )
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--target-update-frequency", type=int, default=500)
    parser.add_argument("--start-epsilon", type=float, default=1.0)
    parser.add_argument("--end-epsilon", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10_000)
    parser.add_argument("--evaluation-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--wandb-project", default="ai-and-computer-games")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Use offline for local logging without an internet connection.",
    )
    return parser.parse_args()


def make_env():
    return simple_spread_v3.parallel_env(
        N=2,
        local_ratio=0.0,
        max_cycles=25,
        continuous_actions=False,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_at(step, args):
    fraction = min(step / max(args.epsilon_decay_steps, 1), 1.0)
    return args.start_epsilon + fraction * (
        args.end_epsilon - args.start_epsilon
    )


def random_or_greedy(network, observation, action_space, epsilon, device):
    if random.random() < epsilon:
        return action_space.sample()
    return network.action(observation, device)


def evaluate_iql(networks, games, seed, device):
    env = make_env()
    returns = []
    try:
        for game in range(games):
            observations, _ = env.reset(seed=seed + game)
            episode_return = 0.0
            while env.agents:
                actions = {
                    agent: networks[index].action(observations[agent], device)
                    for index, agent in enumerate(AGENTS)
                }
                observations, rewards, _, _, _ = env.step(actions)
                episode_return += rewards["agent_0"]
            returns.append(episode_return)
    finally:
        env.close()
    return np.asarray(returns, dtype=np.float64)


def evaluate_cql(network, games, seed, device):
    env = make_env()
    returns = []
    action_size_1 = env.action_space("agent_1").n
    try:
        for game in range(games):
            observations, _ = env.reset(seed=seed + game)
            episode_return = 0.0
            while env.agents:
                joint_observation = np.concatenate(
                    [observations[agent] for agent in AGENTS]
                )
                joint_action = network.action(joint_observation, device)
                actions = {
                    "agent_0": joint_action // action_size_1,
                    "agent_1": joint_action % action_size_1,
                }
                observations, rewards, _, _, _ = env.step(actions)
                episode_return += rewards["agent_0"]
            returns.append(episode_return)
    finally:
        env.close()
    return np.asarray(returns, dtype=np.float64)


def train_iql(args, run, device):
    env = make_env()
    observations, _ = env.reset(seed=args.seed)

    observation_sizes = [env.observation_space(agent).shape[0] for agent in AGENTS]
    action_sizes = [env.action_space(agent).n for agent in AGENTS]
    networks = [
        QNetwork(observation_sizes[i], action_sizes[i]).to(device)
        for i in range(2)
    ]
    targets = [
        QNetwork(observation_sizes[i], action_sizes[i]).to(device)
        for i in range(2)
    ]
    for network, target in zip(networks, targets):
        update_target(network, target)

    optimizers = [
        torch.optim.Adam(network.parameters(), lr=args.learning_rate)
        for network in networks
    ]
    buffers = [
        ReplayBuffer(
            args.buffer_size,
            observation_sizes[i],
            device,
            seed=args.seed + i,
        )
        for i in range(2)
    ]

    episode = 0
    episode_return = 0.0
    recent_returns = deque(maxlen=100)
    last_losses = [np.nan, np.nan]

    try:
        for step in range(args.total_timesteps):
            if not env.agents:
                episode += 1
                recent_returns.append(episode_return)
                metrics = {
                    "global_step": step,
                    "train_episode": episode,
                    "train_episode_return": episode_return,
                    "train_mean_return_100": float(np.mean(recent_returns)),
                }
                if episode % args.log_every == 0:
                    print(
                        f"Episode {episode:4d} | "
                        f"mean return {np.mean(recent_returns):7.3f}"
                    )
                run.log(metrics)
                observations, _ = env.reset(seed=args.seed + episode)
                episode_return = 0.0

            epsilon = epsilon_at(step, args)
            actions = {
                agent: random_or_greedy(
                    networks[i],
                    observations[agent],
                    env.action_space(agent),
                    epsilon,
                    device,
                )
                for i, agent in enumerate(AGENTS)
            }
            old_observations = [observations[agent] for agent in AGENTS]
            observations, rewards, terminations, truncations, _ = env.step(actions)
            episode_return += rewards["agent_0"]

            for i, agent in enumerate(AGENTS):
                done = terminations[agent] or truncations[agent]
                next_observation = (
                    np.zeros(observation_sizes[i], dtype=np.float32)
                    if done
                    else observations[agent]
                )
                buffers[i].add(
                    old_observations[i],
                    actions[agent],
                    rewards[agent],
                    next_observation,
                    done,
                )

            if step >= args.learning_starts and len(buffers[0]) >= args.batch_size:
                for i in range(2):
                    last_losses[i] = train_dqn(
                        networks[i],
                        targets[i],
                        buffers[i],
                        optimizers[i],
                        args.batch_size,
                        args.gamma,
                    )

            if (
                step >= args.learning_starts
                and step % args.target_update_frequency == 0
            ):
                for network, target in zip(networks, targets):
                    update_target(network, target)

            if step % args.log_every == 0:
                run.log(
                    {
                        "global_step": step,
                        "train_epsilon": epsilon,
                        "train_loss_agent_0": last_losses[0],
                        "train_loss_agent_1": last_losses[1],
                        "train_loss_mean": float(np.nanmean(last_losses))
                        if not np.all(np.isnan(last_losses))
                        else np.nan,
                        "train_replay_size": len(buffers[0]),
                    }
                )
    finally:
        env.close()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, network in enumerate(networks):
        path = args.checkpoint_dir / f"spread_iql_wandb_agent_{i}.pth"
        torch.save(network.state_dict(), path)
        paths.append(path)

    returns = evaluate_iql(
        networks,
        args.evaluation_games,
        seed=args.seed + 10_000,
        device=device,
    )
    return paths, returns


def train_cql(args, run, device):
    env = make_env()
    observations, _ = env.reset(seed=args.seed)

    observation_sizes = [env.observation_space(agent).shape[0] for agent in AGENTS]
    action_sizes = [env.action_space(agent).n for agent in AGENTS]
    joint_observation_size = sum(observation_sizes)
    joint_action_size = int(np.prod(action_sizes))

    network = QNetwork(joint_observation_size, joint_action_size).to(device)
    target = QNetwork(joint_observation_size, joint_action_size).to(device)
    update_target(network, target)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate)
    buffer = ReplayBuffer(
        args.buffer_size,
        joint_observation_size,
        device,
        seed=args.seed,
    )

    episode = 0
    episode_return = 0.0
    recent_returns = deque(maxlen=100)
    last_loss = np.nan

    try:
        for step in range(args.total_timesteps):
            if not env.agents:
                episode += 1
                recent_returns.append(episode_return)
                run.log(
                    {
                        "global_step": step,
                        "train_episode": episode,
                        "train_episode_return": episode_return,
                        "train_mean_return_100": float(np.mean(recent_returns)),
                    }
                )
                if episode % args.log_every == 0:
                    print(
                        f"Episode {episode:4d} | "
                        f"mean return {np.mean(recent_returns):7.3f}"
                    )
                observations, _ = env.reset(seed=args.seed + episode)
                episode_return = 0.0

            epsilon = epsilon_at(step, args)
            joint_observation = np.concatenate(
                [observations[agent] for agent in AGENTS]
            )
            if random.random() < epsilon:
                actions = {
                    agent: env.action_space(agent).sample() for agent in AGENTS
                }
                joint_action = actions["agent_0"] * action_sizes[1] + actions["agent_1"]
            else:
                joint_action = network.action(joint_observation, device)
                actions = {
                    "agent_0": joint_action // action_sizes[1],
                    "agent_1": joint_action % action_sizes[1],
                }

            observations, rewards, terminations, truncations, _ = env.step(actions)
            reward = rewards["agent_0"]
            episode_return += reward
            done = terminations["agent_0"] or truncations["agent_0"]
            next_joint_observation = (
                np.zeros(joint_observation_size, dtype=np.float32)
                if done
                else np.concatenate([observations[agent] for agent in AGENTS])
            )
            buffer.add(
                joint_observation,
                joint_action,
                reward,
                next_joint_observation,
                done,
            )

            if step >= args.learning_starts and len(buffer) >= args.batch_size:
                last_loss = train_dqn(
                    network,
                    target,
                    buffer,
                    optimizer,
                    args.batch_size,
                    args.gamma,
                )

            if (
                step >= args.learning_starts
                and step % args.target_update_frequency == 0
            ):
                update_target(network, target)

            if step % args.log_every == 0:
                run.log(
                    {
                        "global_step": step,
                        "train_epsilon": epsilon,
                        "train_loss": last_loss,
                        "train_replay_size": len(buffer),
                    }
                )
    finally:
        env.close()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = args.checkpoint_dir / "spread_cql_wandb.pth"
    torch.save(network.state_dict(), path)
    returns = evaluate_cql(
        network,
        args.evaluation_games,
        seed=args.seed + 10_000,
        device=device,
    )
    return [path], returns


def main():
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.evaluation_games <= 0:
        raise ValueError("--evaluation-games must be positive")
    if args.batch_size <= 0 or args.buffer_size < args.batch_size:
        raise ValueError("--buffer-size must be at least --batch-size")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(args).copy()
    config["checkpoint_dir"] = str(args.checkpoint_dir)
    config["device"] = str(device)
    config["environment"] = "MPE2 simple_spread_v3"
    config["agents"] = 2
    config["actions_per_agent"] = 5

    run_name = args.wandb_run_name or f"simple-spread-{args.algorithm}-seed-{args.seed}"
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=f"simple-spread-{args.algorithm}",
        job_type="train",
        config=config,
        mode=args.wandb_mode,
        tags=["mpe2", "simple-spread", "marl", args.algorithm],
    ) as run:
        run.define_metric("global_step")
        run.define_metric("train_*", step_metric="global_step")

        if args.algorithm == "iql":
            checkpoint_paths, evaluation_returns = train_iql(args, run, device)
        else:
            checkpoint_paths, evaluation_returns = train_cql(args, run, device)

        mean_return = float(np.mean(evaluation_returns))
        std_return = float(np.std(evaluation_returns))
        run.log(
            {
                "evaluation_mean_return": mean_return,
                "evaluation_std_return": std_return,
                "evaluation_games": args.evaluation_games,
            }
        )
        run.summary["evaluation_mean_return"] = mean_return
        run.summary["evaluation_std_return"] = std_return

        table = wandb.Table(columns=["episode", "team_return"])
        for episode, episode_return in enumerate(evaluation_returns):
            table.add_data(episode, float(episode_return))
        run.log({"evaluation_returns": table})

        if args.wandb_mode != "disabled":
            artifact = wandb.Artifact(
                name=f"simple-spread-{args.algorithm}-model",
                type="model",
                metadata={
                    "algorithm": args.algorithm,
                    "seed": args.seed,
                    "evaluation_mean_return": mean_return,
                },
            )
            for checkpoint_path in checkpoint_paths:
                artifact.add_file(str(checkpoint_path))
            run.log_artifact(artifact)

        print(f"Device: {device}")
        print(
            f"Evaluation over {args.evaluation_games} games: "
            f"{mean_return:.3f} +/- {std_return:.3f}"
        )
        print("Checkpoints:")
        for checkpoint_path in checkpoint_paths:
            print(f"  {checkpoint_path}")


if __name__ == "__main__":
    main()
