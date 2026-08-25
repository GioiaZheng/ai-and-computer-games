"""Verify each role-specialized exported policy against its PPO source."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .official_environment import create_official_environment
from .verify_official_submission import load_agent_class


def main(args):
    from stable_baselines3 import PPO

    env = create_official_environment(render_mode=None)
    observations, _ = env.reset(seed=args.seed)
    sources = {
        "first_0": PPO.load(args.first_model, device=args.device),
        "second_0": PPO.load(args.second_model, device=args.device),
    }
    Agent = load_agent_class(args.agent_directory / "agent.py")
    exported = Agent(env)
    for role, observation in observations.items():
        source_observation, _ = sources[role].policy.obs_to_tensor(observation)
        source_distribution = sources[role].policy.get_distribution(source_observation)
        source_probabilities = source_distribution.distribution.probs.detach().cpu().numpy()
        exported_probabilities = (
            torch.softmax(exported.forward(observation), dim=-1).detach().cpu().numpy()
        )
        np.testing.assert_allclose(
            source_probabilities, exported_probabilities, rtol=1e-5, atol=1e-6
        )
        sampled_actions = [exported.get_action(observation) for _ in range(32)]
        if not all(0 <= action < 18 for action in sampled_actions):
            raise AssertionError(f"{role}: invalid sampled actions {sampled_actions}")
        print(f"{role}: probabilities match; sampled={sorted(set(sampled_actions))}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-model", type=Path, required=True)
    parser.add_argument("--second-model", type=Path, required=True)
    parser.add_argument("--agent-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=82_026)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
