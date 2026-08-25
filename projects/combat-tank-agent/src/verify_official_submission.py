"""Check exported Agent logits against the source SB3 policy."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from .official_environment import create_official_environment


def load_agent_class(agent_file: Path):
    spec = importlib.util.spec_from_file_location("submitted_agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load {agent_file}")
    spec.loader.exec_module(module)
    return module.Agent


def main(args):
    from stable_baselines3 import PPO

    env = create_official_environment(render_mode=None)
    observations, _ = env.reset(seed=args.seed)
    source = PPO.load(args.model, device=args.device)
    Agent = load_agent_class(args.agent_directory / "agent.py")
    exported = Agent(env)
    for agent_name, observation in observations.items():
        source_observation, _ = source.policy.obs_to_tensor(observation)
        source_distribution = source.policy.get_distribution(source_observation)
        source_probabilities = (
            source_distribution.distribution.probs.detach().cpu().numpy()
        )
        exported_logits = exported.forward(observation)
        exported_probabilities = (
            torch.softmax(exported_logits, dim=-1).detach().cpu().numpy()
        )
        np.testing.assert_allclose(
            source_probabilities,
            exported_probabilities,
            rtol=1e-5,
            atol=1e-6,
        )
        sampled_actions = [exported.get_action(observation) for _ in range(32)]
        if not all(0 <= action < 18 for action in sampled_actions):
            raise AssertionError(f"{agent_name}: invalid sampled actions {sampled_actions}")
        print(
            f"{agent_name}: logits match; sampled actions="
            f"{sorted(set(sampled_actions))}"
        )
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--agent-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=82_026)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
