"""Watch one agent play the unchanged official Combat Tank environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .environment import ACTION_SETS, AGENTS, SingleAgentCombatEnv, create_environment
from .opponents import PPOOpponentPolicy
from .scripted_agent import ScriptedOpponentPolicy, SpawnHunterAgent


def watch_scripted_selfplay(seed: int):
    """Render both role-specific scripted policies in the official game."""
    env = create_environment(render_mode="human")
    observations, _ = env.reset(seed=seed)
    agents = {role: SpawnHunterAgent(role=role) for role in AGENTS}
    scores = {role: 0.0 for role in AGENTS}
    steps = 0
    try:
        while env.agents:
            actions = {
                role: agents[role].get_action(observations.get(role))
                for role in env.agents
            }
            observations, rewards, _, _, _ = env.step(actions)
            for role, reward in rewards.items():
                agents[role].observe_reward(float(reward))
                scores[role] += float(reward)
            steps += 1
    finally:
        env.close()
    print(f"scores={scores} steps={steps}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["scripted", "ppo"], default="scripted")
    parser.add_argument(
        "--opponent", choices=["random", "scripted", "model"], default="random"
    )
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument("--opponent-action-set", choices=ACTION_SETS, default="fire")
    parser.add_argument("--opponent-prefix-steps", type=int, default=2024)
    parser.add_argument("--opponent-device", default="cuda")
    parser.add_argument("--opponent-stochastic", action="store_true")
    parser.add_argument("--max-action-repeat", type=int, default=24)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--role", choices=AGENTS, default=AGENTS[0])
    parser.add_argument("--seed", type=int, default=31891)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--action-set", choices=ACTION_SETS, default="sweep")
    parser.add_argument("--scripted-prefix-steps", type=int, default=2024)
    args = parser.parse_args()

    if args.opponent == "scripted" and args.agent == "scripted":
        watch_scripted_selfplay(args.seed)
        return
    if args.opponent == "model" and args.opponent_model is None:
        parser.error("--opponent-model is required when --opponent model")

    learner_policy = None
    if args.agent == "ppo":
        if args.model is None:
            parser.error("--model is required when --agent ppo is selected")
        learner_policy = PPOOpponentPolicy(
            args.model,
            action_set=args.action_set,
            scripted_prefix_steps=0,
            device=args.device,
            deterministic=not args.stochastic,
            max_action_repeat=args.max_action_repeat,
        )

    env = SingleAgentCombatEnv(
        opponent_policy=(
            ScriptedOpponentPolicy()
            if args.opponent == "scripted"
            else PPOOpponentPolicy(
                args.opponent_model,
                action_set=args.opponent_action_set,
                scripted_prefix_steps=args.opponent_prefix_steps,
                device=args.opponent_device,
                deterministic=not args.opponent_stochastic,
                max_action_repeat=args.max_action_repeat,
            )
            if args.opponent == "model"
            else None
        ),
        fixed_role=args.role,
        learner_actions=ACTION_SETS["all"],
        seed=args.seed,
        render_mode="human",
    )
    observation, _ = env.reset(seed=args.seed)
    expert = SpawnHunterAgent(role=args.role)
    score = 0.0
    step = 0
    try:
        while True:
            if args.agent == "scripted" or step < args.scripted_prefix_steps:
                game_action = expert.get_action(observation)
            else:
                game_action = learner_policy(observation, args.role)
            observation, reward, terminated, truncated, _ = env.step_game_action(
                game_action
            )
            expert.observe_reward(reward)
            if learner_policy is not None:
                learner_policy.observe_reward(reward, args.role)
            score += reward
            step += 1
            if terminated or truncated:
                break
    finally:
        env.close()

    print(f"role={args.role} score={score:.2f} steps={step}")


if __name__ == "__main__":
    main()
