"""Pommerman FFA environment helpers and the PyTorch policy agent."""

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import pommerman
from pommerman import agents

from .encoding import encode_observation, valid_action_mask
from .model import ActorCritic


@dataclass
class Decision:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    log_probability: float
    value: float


class TorchPolicyAgent(agents.BaseAgent):
    """Adapter from an ActorCritic model to Pommerman's BaseAgent contract."""

    def __init__(
        self,
        model: ActorCritic,
        device: torch.device,
        deterministic: bool = False,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.deterministic = deterministic
        self.last_decision: Optional[Decision] = None

    def act(self, observation, action_space):
        encoded = encode_observation(observation)
        mask = valid_action_mask(observation)
        observations = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        masks = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            distribution, values = self.model.distribution(observations, masks)
            if self.deterministic:
                action_tensor = distribution.logits.argmax(dim=-1)
            else:
                action_tensor = distribution.sample()
            log_probability = distribution.log_prob(action_tensor)
        action = int(action_tensor.item())
        self.last_decision = Decision(
            observation=encoded,
            action_mask=mask,
            action=action,
            log_probability=float(log_probability.item()),
            value=float(values.item()),
        )
        return action


def make_ffa_environment(
    policy_agent: TorchPolicyAgent,
    train_slot: int,
    opponent_mode: str,
    seed: int,
    render_mode: str = "rgb_array",
):
    """Create one FFA game with the learner in the requested spawn slot."""

    if train_slot not in range(4):
        raise ValueError("train_slot must be between 0 and 3")
    if opponent_mode not in {"random", "simple", "mixed"}:
        raise ValueError("opponent_mode must be random, simple, or mixed")

    generator = random.Random(seed)
    agent_list = []
    for slot in range(4):
        if slot == train_slot:
            agent_list.append(policy_agent)
        elif opponent_mode == "random":
            agent_list.append(agents.RandomAgent())
        elif opponent_mode == "simple":
            agent_list.append(agents.SimpleAgent())
        else:
            use_simple = generator.random() < 0.7
            agent_list.append(agents.SimpleAgent() if use_simple else agents.RandomAgent())

    environment = pommerman.make(
        "PommeFFACompetition-v0",
        agent_list,
        render_mode=render_mode,
    )
    environment.seed(seed)
    return environment


def make_expert_environment(seed: int, render_mode: str = "rgb_array"):
    """Create four SimpleAgents for behavioral-cloning demonstrations."""

    agent_list = [agents.SimpleAgent() for _ in range(4)]
    environment = pommerman.make(
        "PommeFFACompetition-v0",
        agent_list,
        render_mode=render_mode,
    )
    environment.seed(seed)
    return environment
