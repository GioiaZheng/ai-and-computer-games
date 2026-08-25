"""Record a match from the unchanged official RGB renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .environment import ACTION_SETS, AGENTS, SingleAgentCombatEnv
from .opponents import PPOOpponentPolicy
from .scripted_agent import SpawnHunterAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--opponent-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=AGENTS, default=AGENTS[0])
    parser.add_argument("--seed", type=int, default=201101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-set", choices=ACTION_SETS, default="fire")
    parser.add_argument("--opponent-action-set", choices=ACTION_SETS, default="fire")
    parser.add_argument("--scripted-prefix-steps", type=int, default=2024)
    parser.add_argument("--max-steps", type=int, default=3224)
    parser.add_argument("--frame-stride", type=int, default=4)
    args = parser.parse_args()

    learner = PPOOpponentPolicy(
        args.model,
        action_set=args.action_set,
        scripted_prefix_steps=0,
        device=args.device,
        deterministic=False,
    )
    opponent = PPOOpponentPolicy(
        args.opponent_model,
        action_set=args.opponent_action_set,
        scripted_prefix_steps=args.scripted_prefix_steps,
        device=args.device,
        deterministic=False,
    )
    env = SingleAgentCombatEnv(
        opponent_policy=opponent,
        fixed_role=args.role,
        learner_actions=ACTION_SETS["all"],
        seed=args.seed,
        render_mode="rgb_array",
    )
    observation, _ = env.reset(seed=args.seed)
    expert = SpawnHunterAgent(role=args.role)
    frames: list[Image.Image] = []
    score = 0.0

    try:
        for step in range(args.max_steps):
            if step < args.scripted_prefix_steps:
                game_action = expert.get_action(observation)
            else:
                game_action = learner(observation, args.role)
            observation, reward, terminated, truncated, _ = env.step_game_action(
                game_action
            )
            expert.observe_reward(reward)
            learner.observe_reward(reward, args.role)
            score += reward
            if step % args.frame_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(
                        Image.fromarray(np.asarray(frame, dtype=np.uint8)).quantize(
                            colors=64
                        )
                    )
            if terminated or truncated:
                break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("The official RGB renderer returned no frames")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 * args.frame_stride / 60)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(
        f"saved={args.output} frames={len(frames)} score={score:.2f} "
        f"steps={step + 1}"
    )


if __name__ == "__main__":
    main()
