import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, observation):
        return self.network(observation.float())

    @torch.no_grad()
    def action(self, observation, device):
        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        q_values = self(observation)
        return int(q_values.argmax(dim=1).item())


class ReplayBuffer:
    def __init__(self, size, observation_size, device, seed):
        self.size = size
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.observations = np.empty(
            (size, observation_size), dtype=np.float32
        )
        self.actions = np.empty(size, dtype=np.int64)
        self.rewards = np.empty(size, dtype=np.float32)
        self.next_observations = np.empty(
            (size, observation_size), dtype=np.float32
        )
        self.dones = np.empty(size, dtype=np.float32)

        self.position = 0
        self.length = 0

    def add(self, observation, action, reward, next_observation, done):
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size):
        indices = self.rng.integers(0, self.length, size=batch_size)

        observations = torch.as_tensor(
            self.observations[indices], device=self.device
        )
        actions = torch.as_tensor(
            self.actions[indices], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            self.rewards[indices], device=self.device
        )
        next_observations = torch.as_tensor(
            self.next_observations[indices], device=self.device
        )
        dones = torch.as_tensor(
            self.dones[indices], device=self.device
        )

        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        return self.length


def train_dqn(
    q_network,
    target_network,
    replay_buffer,
    optimizer,
    batch_size,
    gamma,
):
    observations, actions, rewards, next_observations, dones = (
        replay_buffer.sample(batch_size)
    )

    with torch.no_grad():
        next_q_values = target_network(next_observations)
        next_values = next_q_values.max(dim=1).values
        targets = rewards + gamma * next_values * (1.0 - dones)

    q_values = q_network(observations)
    chosen_q_values = q_values.gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)

    loss = F.smooth_l1_loss(chosen_q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def update_target(q_network, target_network):
    target_network.load_state_dict(q_network.state_dict())
