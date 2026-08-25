"""Role-specialized tournament agent for the instructor-provided template."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_encoder():
    return nn.Sequential(
        nn.Conv2d(6, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    )


def make_projection():
    return nn.Sequential(nn.Linear(3136, 512), nn.ReLU())


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.first_encoder = make_encoder()
        self.first_projection = make_projection()
        self.first_action_head = nn.Linear(512, 18)
        self.second_encoder = make_encoder()
        self.second_projection = make_projection()
        self.second_action_head = nn.Linear(512, 18)

        weights_path = Path(__file__).with_name("weights.pt")
        try:
            weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(weights)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = 1.0
        self.to(self.device)
        self.eval()

    def _prepare_state(self, state):
        if isinstance(state, torch.Tensor):
            tensor = state
        else:
            tensor = torch.as_tensor(np.asarray(state))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected a 3D or 4D observation, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == 6:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.shape[1] != 6:
            raise ValueError(
                "Expected official observations in HWC or CHW format with 6 channels, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.to(self.device, dtype=torch.float32) / 255.0

    def _first_logits(self, tensor):
        features = self.first_encoder(tensor)
        return self.first_action_head(self.first_projection(features))

    def _second_logits(self, tensor):
        features = self.second_encoder(tensor)
        return self.second_action_head(self.second_projection(features))

    def forward(self, state):
        tensor = self._prepare_state(state)
        # SuperSuit appends one constant indicator plane per possible agent.
        first_role = tensor[:, 4].mean(dim=(1, 2)) > tensor[:, 5].mean(dim=(1, 2))
        if tensor.shape[0] == 1:
            return self._first_logits(tensor) if bool(first_role.item()) else self._second_logits(tensor)
        first_logits = self._first_logits(tensor)
        second_logits = self._second_logits(tensor)
        return torch.where(first_role[:, None], first_logits, second_logits)

    @torch.no_grad()
    def get_action(self, state=None) -> int:
        if state is None:
            raise ValueError("state is required")
        logits = self.forward(state) / max(float(self.temperature), 1e-6)
        action = torch.distributions.Categorical(logits=logits).sample()
        return int(action.item())
