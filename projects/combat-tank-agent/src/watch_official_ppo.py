"""Watch a PPO checkpoint play in the unmodified official human renderer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .official_environment import AGENTS, create_official_environment


def main(args):
    from stable_baselines3 import PPO

    model = PPO.load(args.model, device=args.device)
    if tuple(model.observation_space.shape) != (6, 84, 84):
        raise ValueError(
            "The selected model is not compatible with the official pipeline: "
            f"{model.observation_space}"
        )
    learner = args.role
    opponent = AGENTS[1] if learner == AGENTS[0] else AGENTS[0]
    env = create_official_environment(render_mode="human")
    observations, _ = env.reset(seed=args.seed)
    scores = {agent: 0.0 for agent in AGENTS}
    action_counts = {agent: np.zeros(18, dtype=np.int64) for agent in AGENTS}
    steps = 0
    try:
        while env.agents and steps < args.max_steps:
            learner_action, _ = model.predict(
                observations[learner], deterministic=args.deterministic
            )
            learner_action = int(np.asarray(learner_action).item())
            opponent_action = int(env.action_space(opponent).sample())
            actions = {learner: learner_action, opponent: opponent_action}
            for agent, action in actions.items():
                action_counts[agent][action] += 1
            observations, rewards, terminations, truncations, _ = env.step(actions)
            for agent, reward in rewards.items():
                scores[agent] += float(reward)
            steps += 1
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
            if all(
                terminations.get(agent, False) or truncations.get(agent, False)
                for agent in AGENTS
            ):
                break
    finally:
        env.close()
    print(
        f"steps={steps} learner={learner} score={scores[learner]:+.0f} "
        f"opponent={scores[opponent]:+.0f}"
    )
    print(
        "learner action counts="
        f"{dict(enumerate(action_counts[learner].tolist()))}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--role", choices=AGENTS, default="first_0")
    parser.add_argument("--seed", type=int, default=82_226)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Limit playback speed for viewing; 0 keeps maximum simulation speed.",
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if args.fps < 0:
        parser.error("--fps must be non-negative")
    return args


if __name__ == "__main__":
    main(parse_args())
