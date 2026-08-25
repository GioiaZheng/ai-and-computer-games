"""Evaluate a trained policy against random actions from both roles."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .environment import ACTION_SETS, AGENTS, SingleAgentCombatEnv
from .opponents import PPOOpponentPolicy
from .scripted_agent import ScriptedOpponentPolicy, SpawnHunterAgent


def evaluate_role(
    model,
    role: str,
    games: int,
    seed: int,
    deterministic: bool,
    action_set: str,
    scripted_prefix_steps: int,
    opponent: str,
    opponent_model,
    opponent_action_set: str,
    opponent_prefix_steps: int,
    opponent_device: str,
    opponent_stochastic: bool,
):
    learner_actions = ACTION_SETS[action_set]
    envs = [
        SingleAgentCombatEnv(
            opponent_policy=(
                ScriptedOpponentPolicy()
                if opponent == "scripted"
                else PPOOpponentPolicy(
                    opponent_model,
                    action_set=opponent_action_set,
                    scripted_prefix_steps=opponent_prefix_steps,
                    device=opponent_device,
                    deterministic=not opponent_stochastic,
                )
                if opponent == "model"
                else None
            ),
            fixed_role=role,
            learner_actions=ACTION_SETS["all"],
            seed=seed + game,
        )
        for game in range(games)
    ]
    observations = []
    for game, env in enumerate(envs):
        observation, _ = env.reset(seed=seed + game)
        observations.append(observation)

    scores = np.zeros(games, dtype=np.float32)
    finished = np.zeros(games, dtype=bool)
    actions = Counter()
    hits = np.zeros(games, dtype=np.int32)
    received_hits = np.zeros(games, dtype=np.int32)
    experts = [SpawnHunterAgent(role=role) for _ in range(games)]
    episode_steps = np.zeros(games, dtype=np.int32)
    while not bool(finished.all()):
        active = np.flatnonzero(~finished)
        policy_indices = [index for index in active if episode_steps[index] >= scripted_prefix_steps]
        predicted_actions = {}
        if policy_indices:
            batch = np.stack([observations[index] for index in policy_indices])
            predicted, _ = model.predict(batch, deterministic=deterministic)
            predicted_actions = {
                index: int(action)
                for index, action in zip(
                    policy_indices, np.asarray(predicted).reshape(-1), strict=True
                )
            }
        for index in active:
            if episode_steps[index] < scripted_prefix_steps:
                game_action = experts[index].get_action(observations[index])
            else:
                game_action = learner_actions[predicted_actions[index]]
            actions[game_action] += 1
            observation, reward, terminated, truncated, info = envs[
                index
            ].step_game_action(game_action)
            official_reward = float(info.get("official_reward", reward))
            if official_reward > 0:
                hits[index] += 1
            elif official_reward < 0:
                received_hits[index] += 1
            experts[index].observe_reward(reward)
            observations[index] = observation
            scores[index] += reward
            finished[index] = terminated or truncated
            episode_steps[index] += 1

    for env in envs:
        env.close()
    return scores, actions, hits, received_hits


def main():
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--opponent", choices=["random", "scripted", "model"], default="random"
    )
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument("--opponent-action-set", choices=ACTION_SETS, default="fire")
    parser.add_argument("--opponent-prefix-steps", type=int, default=2024)
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument("--opponent-stochastic", action="store_true")
    parser.add_argument("--games-per-role", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--action-set", choices=ACTION_SETS, default="all")
    parser.add_argument("--scripted-prefix-steps", type=int, default=0)
    args = parser.parse_args()
    if args.opponent == "model" and args.opponent_model is None:
        parser.error("--opponent-model is required when --opponent model")

    model = PPO.load(args.model, device=args.device)
    all_scores = []
    all_actions = Counter()
    for role in AGENTS:
        scores, actions, hits, received_hits = evaluate_role(
            model,
            role,
            args.games_per_role,
            args.seed,
            args.deterministic,
            args.action_set,
            args.scripted_prefix_steps,
            args.opponent,
            args.opponent_model,
            args.opponent_action_set,
            args.opponent_prefix_steps,
            args.opponent_device,
            args.opponent_stochastic,
        )
        all_scores.extend(scores.tolist())
        all_actions.update(actions)
        wins = int(np.count_nonzero(scores > 0))
        draws = int(np.count_nonzero(scores == 0))
        print(
            f"{role}: {scores.mean():.2f} +/- {scores.std():.2f}; "
            f"wins={wins}/{len(scores)} draws={draws}; "
            f"hits={int(hits.sum())} received={int(received_hits.sum())}"
        )
    combined = np.asarray(all_scores, dtype=np.float32)
    print(f"combined: {combined.mean():.2f} +/- {combined.std():.2f}")
    print(
        f"action coverage: {len(all_actions)}/{18}; "
        f"allowed={ACTION_SETS[args.action_set]}; counts={dict(sorted(all_actions.items()))}"
    )


if __name__ == "__main__":
    main()
