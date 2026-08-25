"""Compact convolutional actor-critic used by BC and PPO."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoding import NUM_ACTIONS, NUM_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class ActorCritic(nn.Module):
    def __init__(self, input_channels: int = NUM_CHANNELS, actions: int = NUM_ACTIONS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def distribution(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> Tuple[Categorical, torch.Tensor]:
        logits, values = self(observations)
        masked_logits = logits.masked_fill(~action_masks.bool(), -1e9)
        return Categorical(logits=masked_logits), values


def save_checkpoint(
    path: str,
    model: ActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[Dict] = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, output)


def load_checkpoint(
    path: str,
    model: ActorCritic,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return dict(payload.get("metadata", {}))
