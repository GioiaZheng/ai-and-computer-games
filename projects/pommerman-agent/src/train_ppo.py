"""PPO fine-tuning for one shared Pommerman FFA policy."""

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from .encoding import alive_enemy_count, observation_position
from .environment import TorchPolicyAgent, make_ffa_environment
from .model import ActorCritic, load_checkpoint, save_checkpoint
from .train_bc import resolve_device


@dataclass
class Transition:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_probability: float
    value: float
    reward: float
    terminal: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--opponent", choices=("random", "simple", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--load", default="checkpoints/bc_policy.pt")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_policy.pt")
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


def shaped_reward(
    previous: Dict,
    current: Dict,
    external_reward: float,
    visited: Set[Tuple[int, int]],
    slot: int,
) -> float:
    """Add small learning signals without replacing the official outcome."""

    reward = float(external_reward)
    position = observation_position(current)
    if position not in visited:
        reward += 0.005
        visited.add(position)

    reward += 0.02 * max(0, int(current.get("ammo", 0)) - int(previous.get("ammo", 0)))
    reward += 0.02 * max(
        0,
        int(current.get("blast_strength", 0))
        - int(previous.get("blast_strength", 0)),
    )
    if bool(current.get("can_kick", False)) and not bool(
        previous.get("can_kick", False)
    ):
        reward += 0.03

    self_value = 10 + slot
    enemy_reduction = alive_enemy_count(previous, self_value) - alive_enemy_count(
        current, self_value
    )
    if self_value in current.get("alive", ()):
        reward += 0.15 * max(0, enemy_reduction)
    if int(current.get("ammo", 0)) < int(previous.get("ammo", 0)):
        reward += 0.002
    return reward


def advantages_and_returns(
    transitions: List[Transition], gamma: float, gae_lambda: float
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(transitions), dtype=np.float32)
    last_advantage = 0.0
    next_value = 0.0
    for index in reversed(range(len(transitions))):
        transition = transitions[index]
        nonterminal = 0.0 if transition.terminal else 1.0
        delta = transition.reward + gamma * next_value * nonterminal - transition.value
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
        next_value = transition.value
    values = np.asarray([transition.value for transition in transitions], dtype=np.float32)
    return advantages, advantages + values


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    observations = torch.from_numpy(
        np.stack([transition.observation for transition in transitions])
    ).to(device)
    masks = torch.from_numpy(
        np.stack([transition.action_mask for transition in transitions])
    ).to(device)
    actions = torch.tensor(
        [transition.action for transition in transitions], device=device
    )
    old_log_probabilities = torch.tensor(
        [transition.old_log_probability for transition in transitions],
        dtype=torch.float32,
        device=device,
    )
    advantages, returns = advantages_and_returns(
        transitions, args.gamma, args.gae_lambda
    )
    advantages = torch.from_numpy(advantages).to(device)
    returns = torch.from_numpy(returns).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics = []
    sample_count = len(transitions)
    for _ in range(args.ppo_epochs):
        for indices in torch.randperm(sample_count, device=device).split(args.batch_size):
            distribution, values = model.distribution(
                observations[indices], masks[indices]
            )
            log_probabilities = distribution.log_prob(actions[indices])
            ratio = (log_probabilities - old_log_probabilities[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(
                ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
            ) * advantages[indices]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[indices] - values).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + args.value_coefficient * value_loss
                - args.entropy_coefficient * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            metrics.append(
                (
                    float(policy_loss.item()),
                    float(value_loss.item()),
                    float(entropy.item()),
                )
            )

    metric_array = np.asarray(metrics, dtype=np.float32)
    return {
        "policy_loss": float(metric_array[:, 0].mean()),
        "value_loss": float(metric_array[:, 1].mean()),
        "entropy": float(metric_array[:, 2].mean()),
    }


def collect_episode(
    model: ActorCritic,
    device: torch.device,
    episode: int,
    args: argparse.Namespace,
) -> Tuple[List[Transition], Dict]:
    slot = (episode - 1) % 4
    policy_agent = TorchPolicyAgent(model, device, deterministic=False)
    environment = make_ffa_environment(
        policy_agent,
        slot,
        args.opponent,
        args.seed + episode,
    )
    observations = environment.reset()
    visited = {observation_position(observations[slot])}
    transitions: List[Transition] = []
    action_counts = Counter()
    external_return = 0.0
    shaped_return = 0.0
    result = "incomplete"

    for step in range(1, args.max_steps + 1):
        actions = environment.act(observations)
        decision = policy_agent.last_decision
        if decision is None:
            break
        action_counts[decision.action] += 1
        next_observations, rewards, done, info = environment.step(actions)
        external_reward = float(rewards[slot])
        external_return += external_reward
        current = next_observations[slot]
        self_alive = 10 + slot in current.get("alive", ())
        terminal = bool(done or not self_alive or step == args.max_steps)
        reward = shaped_reward(
            observations[slot], current, external_reward, visited, slot
        )
        shaped_return += reward
        transitions.append(
            Transition(
                observation=decision.observation,
                action_mask=decision.action_mask,
                action=decision.action,
                old_log_probability=decision.log_probability,
                value=decision.value,
                reward=reward,
                terminal=terminal,
            )
        )
        observations = next_observations
        if terminal:
            result = str(info.get("result", "terminal"))
            break

    environment.close()
    return transitions, {
        "slot": slot,
        "steps": len(transitions),
        "external_return": external_return,
        "shaped_return": shaped_return,
        "result": result,
        "visited": len(visited),
        "actions": dict(action_counts),
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.load:
        load_checkpoint(args.load, model, device=str(device))
        print("loaded {}".format(args.load))

    print("PPO fine-tuning against {} opponents".format(args.opponent))
    print("device={} episodes={}".format(device, args.episodes))
    pending: List[Transition] = []
    recent_returns: List[float] = []
    recent_external: List[float] = []
    recent_lengths: List[int] = []
    latest_metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    for episode in range(1, args.episodes + 1):
        transitions, summary = collect_episode(model, device, episode, args)
        pending.extend(transitions)
        recent_returns.append(summary["shaped_return"])
        recent_external.append(summary["external_return"])
        recent_lengths.append(summary["steps"])

        if episode % args.episodes_per_update == 0 and pending:
            latest_metrics = ppo_update(model, optimizer, pending, args, device)
            pending.clear()

        print(
            "episode={:05d} slot={} steps={:03d} official={:+.2f} shaped={:+.3f} "
            "cells={} loss={:.4f}/{:.4f} entropy={:.3f} actions={}".format(
                episode,
                summary["slot"],
                summary["steps"],
                summary["external_return"],
                summary["shaped_return"],
                summary["visited"],
                latest_metrics["policy_loss"],
                latest_metrics["value_loss"],
                latest_metrics["entropy"],
                summary["actions"],
            )
        )

        if args.save_every > 0 and episode % args.save_every == 0:
            checkpoint = Path(args.checkpoint)
            numbered = checkpoint.with_name(
                "{}_{}{}".format(checkpoint.stem, episode, checkpoint.suffix)
            )
            save_checkpoint(
                str(numbered),
                model,
                optimizer,
                {
                    "stage": "ppo",
                    "episode": episode,
                    "mean_official_return_20": float(np.mean(recent_external[-20:])),
                    "mean_shaped_return_20": float(np.mean(recent_returns[-20:])),
                    "mean_length_20": float(np.mean(recent_lengths[-20:])),
                    "seed": args.seed,
                },
            )

    if pending:
        latest_metrics = ppo_update(model, optimizer, pending, args, device)
    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        {
            "stage": "ppo",
            "episode": args.episodes,
            "mean_official_return_20": float(np.mean(recent_external[-20:])),
            "mean_shaped_return_20": float(np.mean(recent_returns[-20:])),
            "mean_length_20": float(np.mean(recent_lengths[-20:])),
            "seed": args.seed,
            **latest_metrics,
        },
    )
    print("saved {}".format(args.checkpoint))


if __name__ == "__main__":
    train(parse_args())
