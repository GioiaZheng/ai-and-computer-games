"""Pretrain a PPO policy from the reliable scripted maze-navigation curriculum."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .environment import ACTION_SETS, AGENTS, SingleAgentCombatEnv
from .scripted_agent import (
    ScriptedOpponentPolicy,
    SpawnHunterAgent,
    WaypointCoverageAgent,
)


def update_policy(model, observations, actions, entropy_coefficient: float):
    observation_tensor, _ = model.policy.obs_to_tensor(np.stack(observations))
    action_tensor = torch.as_tensor(actions, device=model.device, dtype=torch.long)
    distribution = model.policy.get_distribution(observation_tensor)
    log_probability = distribution.log_prob(action_tensor)
    entropy = distribution.entropy()
    loss = -log_probability.mean() - entropy_coefficient * entropy.mean()

    model.policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
    model.policy.optimizer.step()
    accuracy = (distribution.get_actions(deterministic=True) == action_tensor).float().mean()
    return float(loss.item()), float(accuracy.item())


def main():
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-role", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int, default=2500)
    parser.add_argument("--samples-per-action", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--opponent", choices=["random", "scripted"], default="scripted")
    parser.add_argument(
        "--expert",
        choices=["spawn-hunter", "waypoint"],
        default="spawn-hunter",
    )
    parser.add_argument("--action-set", choices=ACTION_SETS, default="all")
    parser.add_argument("--scripted-prefix-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=14001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/bc/model"))
    args = parser.parse_args()

    learner_actions = ACTION_SETS[args.action_set]
    action_to_policy = {
        game_action: policy_action
        for policy_action, game_action in enumerate(learner_actions)
    }
    bootstrap_env = SingleAgentCombatEnv(
        fixed_role=AGENTS[0],
        learner_actions=learner_actions,
        seed=args.seed,
    )
    if args.load_model is not None:
        model = PPO.load(args.load_model, env=bootstrap_env, device=args.device)
        for parameter_group in model.policy.optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
    else:
        model = PPO(
            "CnnPolicy",
            bootstrap_env,
            learning_rate=args.learning_rate,
            n_steps=1024,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            verbose=0,
        )

    rng = np.random.default_rng(args.seed)
    reservoirs = defaultdict(list)
    seen_actions = Counter()
    action_counts = Counter()
    total_samples = 0
    for role_index, role in enumerate(AGENTS):
        for episode in range(args.episodes_per_role):
            episode_seed = args.seed + role_index * 10_000 + episode
            opponent_policy = (
                ScriptedOpponentPolicy() if args.opponent == "scripted" else None
            )
            env = SingleAgentCombatEnv(
                opponent_policy=opponent_policy,
                fixed_role=role,
                learner_actions=learner_actions,
                seed=episode_seed,
            )
            observation, _ = env.reset(seed=episode_seed)
            expert = (
                WaypointCoverageAgent(
                    role=role,
                    route="top" if episode % 2 == 0 else "bottom",
                )
                if args.expert == "waypoint"
                else SpawnHunterAgent(role=role)
            )
            for _ in range(args.scripted_prefix_steps):
                game_action = expert.get_action(env.raw_observations[role])
                observation, reward, terminated, truncated, _ = env.step_game_action(
                    game_action
                )
                expert.observe_reward(reward)
                if terminated or truncated:
                    break
            for _ in range(args.steps_per_episode):
                game_action = expert.get_action(env.raw_observations[role])
                if game_action not in action_to_policy:
                    continue
                action = action_to_policy[game_action]
                action_counts[game_action] += 1
                total_samples += 1
                seen_actions[action] += 1
                candidates = reservoirs[action]
                if len(candidates) < args.samples_per_action:
                    candidates.append(observation.copy())
                else:
                    replacement = int(rng.integers(0, seen_actions[action]))
                    if replacement < args.samples_per_action:
                        candidates[replacement] = observation.copy()
                observation, reward, terminated, truncated, _ = env.step_game_action(
                    game_action
                )
                expert.observe_reward(reward)
                if terminated or truncated:
                    break
            env.close()
            print(
                f"role={role} episode={episode + 1}/{args.episodes_per_role} "
                f"samples={total_samples} actions={dict(sorted(action_counts.items()))}"
            )

    observations = np.stack(
        [observation for action in sorted(reservoirs) for observation in reservoirs[action]]
    )
    actions = np.asarray(
        [action for action in sorted(reservoirs) for _ in reservoirs[action]],
        dtype=np.int64,
    )
    losses = []
    accuracies = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(actions))
        epoch_losses = []
        epoch_accuracies = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            loss, accuracy = update_policy(
                model,
                observations[indices],
                actions[indices],
                args.entropy_coefficient,
            )
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        losses.extend(epoch_losses)
        accuracies.extend(epoch_accuracies)
        print(
            f"epoch={epoch + 1:02d}/{args.epochs} loss={np.mean(epoch_losses):.4f} "
            f"accuracy={np.mean(epoch_accuracies):.3f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    bootstrap_env.close()
    balanced_counts = {action: len(values) for action, values in sorted(reservoirs.items())}
    print(
        f"saved={args.output}.zip collected={total_samples} "
        f"balanced={balanced_counts} source_actions={dict(sorted(action_counts.items()))}"
    )


if __name__ == "__main__":
    main()
