"""Record a PPO match from the exact instructor-specified environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .official_environment import AGENTS, create_official_environment


def main(args):
    from stable_baselines3 import PPO

    model = PPO.load(args.model, device=args.device)
    if tuple(model.observation_space.shape) != (6, 84, 84):
        raise ValueError(f"Model is not official-pipeline compatible: {model.observation_space}")

    learner = args.role
    opponent = AGENTS[1] if learner == AGENTS[0] else AGENTS[0]
    env = create_official_environment(render_mode="rgb_array")
    observations, _ = env.reset(seed=args.seed)
    frames: list[Image.Image] = []
    scores = {agent: 0.0 for agent in AGENTS}
    steps = 0
    try:
        while env.agents and steps < args.max_steps:
            learner_action, _ = model.predict(
                observations[learner], deterministic=False
            )
            actions = {
                learner: int(np.asarray(learner_action).item()),
                opponent: int(env.action_space(opponent).sample()),
            }
            observations, rewards, terminations, truncations, _ = env.step(actions)
            for role, reward in rewards.items():
                scores[role] += float(reward)
            if steps % args.frame_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(
                        Image.fromarray(np.asarray(frame, dtype=np.uint8)).quantize(
                            colors=64
                        )
                    )
            steps += 1
            if all(
                terminations.get(role, False) or truncations.get(role, False)
                for role in AGENTS
            ):
                break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("The official RGB renderer returned no frames")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 * args.frame_stride * 4 / 60)
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
        f"saved={args.output} frames={len(frames)} steps={steps} "
        f"learner_score={scores[learner]:+.0f} opponent_score={scores[opponent]:+.0f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=AGENTS, required=True)
    parser.add_argument("--seed", type=int, default=510826)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
