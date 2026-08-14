"""Tournament agent backed by an exported PPO policy."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


IN_CHANNELS = 6
NUM_ACTIONS = 18
WEIGHTS_FILE = "policy_weights.pt"


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, NUM_ACTIONS),
        )

    def forward(self, observation):
        observation = observation.permute(0, 3, 1, 2).float() / 255.0
        features = self.conv(observation).reshape(observation.shape[0], -1)
        return self.fc(features)


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork().to(self.device)
        weights_path = Path(__file__).with_name(WEIGHTS_FILE)
        try:
            weights = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            weights = torch.load(weights_path, map_location=self.device)
        self.policy.load_state_dict(weights)
        self.policy.eval()

    def get_action(self, state=None):
        if state is None:
            return 0
        observation = np.asarray(state, dtype=np.uint8)
        if observation.shape != (84, 84, IN_CHANNELS):
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits = self.policy(tensor)
            probabilities = torch.softmax(logits, dim=1)
            action = torch.multinomial(probabilities, num_samples=1)
        return int(action.item())
