"""Evaluate a trained Pommerman policy in every FFA spawn slot."""

import argparse
from collections import Counter

import numpy as np
import torch

from .environment import TorchPolicyAgent, make_ffa_environment
from .model import ActorCritic, load_checkpoint
from .train_bc import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/ppo_policy.pt")
    parser.add_argument("--opponent", choices=("random", "simple", "mixed"), default="simple")
    parser.add_argument("--games-per-role", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model = ActorCritic().to(device)
    metadata = load_checkpoint(args.model, model, device=str(device))
    model.eval()
    all_returns = []
    all_lengths = []
    all_actions = Counter()

    for slot in range(4):
        role_returns = []
        role_lengths = []
        for game in range(args.games_per_role):
            agent = TorchPolicyAgent(model, device, deterministic=True)
            environment = make_ffa_environment(
                agent,
                slot,
                args.opponent,
                args.seed + slot * 1000 + game,
                render_mode="human" if args.render else "rgb_array",
            )
            observations = environment.reset()
            episode_return = 0.0
            for step in range(1, args.max_steps + 1):
                if args.render:
                    environment.render()
                actions = environment.act(observations)
                all_actions[int(actions[slot])] += 1
                observations, rewards, done, _ = environment.step(actions)
                episode_return += float(rewards[slot])
                if done or 10 + slot not in observations[slot].get("alive", ()):
                    break
            environment.close()
            role_returns.append(episode_return)
            role_lengths.append(step)
        all_returns.extend(role_returns)
        all_lengths.extend(role_lengths)
        print(
            "slot={} return={:+.3f}+/-{:.3f} length={:.1f}".format(
                slot,
                float(np.mean(role_returns)),
                float(np.std(role_returns)),
                float(np.mean(role_lengths)),
            )
        )

    print(
        "combined return={:+.3f}+/-{:.3f} length={:.1f}".format(
            float(np.mean(all_returns)),
            float(np.std(all_returns)),
            float(np.mean(all_lengths)),
        )
    )
    print("actions={}".format(dict(sorted(all_actions.items()))))
    print("checkpoint_metadata={}".format(metadata))


if __name__ == "__main__":
    evaluate(parse_args())
