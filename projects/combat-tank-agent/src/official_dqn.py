"""Load Nature-DQN checkpoints that use the official six-channel pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class OfficialDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 18),
        )

    def forward(self, observation):
        if isinstance(observation, torch.Tensor):
            tensor = observation
        else:
            tensor = torch.as_tensor(np.asarray(observation))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] == 6:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] != 6:
            raise ValueError(f"Expected six observation channels, got {tensor.shape}")
        tensor = tensor.to(next(self.parameters()).device, dtype=torch.float32) / 255.0
        return self.fc(self.conv(tensor).flatten(start_dim=1))

    @torch.no_grad()
    def get_action(self, observation) -> int:
        return int(torch.argmax(self(observation), dim=-1).item())


def load_official_dqn(path: str | Path, device: str = "cpu") -> OfficialDQN:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    state_dict = checkpoint.get("policy_net", checkpoint.get("state_dict", checkpoint))
    model = OfficialDQN()
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    return model
