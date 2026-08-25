"""Tournament agent using a Double-Dueling DQN with an anti-stall controller."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


IN_CHANNELS = 6
NUM_ACTIONS = 18
REPEAT_THRESHOLD = 60
ESCAPE_STEPS = 8
OFFICIAL_EPISODE_STEPS = 1785
WEIGHTS_FILE = "policy_weights.pt"


class DQN(nn.Module):
    def __init__(self, frame_stack, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.shared = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
        self.value_head = nn.Linear(512, 1)
        self.advantage_head = nn.Linear(512, n_actions)

    def forward(self, state):
        hidden = self.shared(self.features(state.float() / 255.0))
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DQN(IN_CHANNELS, NUM_ACTIONS).to(self.device)
        weights_path = Path(__file__).with_name(WEIGHTS_FILE)
        try:
            weights = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            weights = torch.load(weights_path, map_location=self.device)
        self.net.load_state_dict(weights)
        self.net.eval()
        self._reset_controller()

    def _reset_controller(self):
        self.previous_greedy = None
        self.repeat_count = 0
        self.escape_remaining = 0
        self.escape_index = 0
        self.recent_actions = []
        self.episode_step = 0

    @staticmethod
    def _to_training_format(state):
        observation = np.asarray(state, dtype=np.uint8)
        if observation.shape != (84, 84, IN_CHANNELS):
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
        return np.ascontiguousarray(observation.transpose(2, 0, 1))

    def get_action(self, state=None) -> int:
        if state is None:
            self._reset_controller()
            return 0
        if self.episode_step >= OFFICIAL_EPISODE_STEPS:
            self._reset_controller()

        observation = self._to_training_format(state)
        tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            q_values = self.net(tensor)[0]
        ranking = torch.argsort(q_values, descending=True).tolist()
        greedy = int(ranking[0])

        if greedy == self.previous_greedy:
            self.repeat_count += 1
        else:
            self.previous_greedy = greedy
            self.repeat_count = 1

        if self.repeat_count >= REPEAT_THRESHOLD and self.escape_remaining == 0:
            self.escape_remaining = ESCAPE_STEPS
            self.escape_index = 0
            self.repeat_count = 0

        selected = greedy
        if self.escape_remaining:
            recent = set(self.recent_actions[-ESCAPE_STEPS:])
            alternatives = [action for action in ranking if action not in recent]
            if not alternatives:
                alternatives = ranking
            selected = int(alternatives[self.escape_index % len(alternatives)])
            self.escape_index += 1
            self.escape_remaining -= 1

        self.recent_actions.append(selected)
        if len(self.recent_actions) > ESCAPE_STEPS:
            self.recent_actions.pop(0)
        self.episode_step += 1
        return selected
