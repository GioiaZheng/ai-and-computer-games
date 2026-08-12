import random
from collections import deque

import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm


# ---------------------------------------------------------
# Replay memory
# ---------------------------------------------------------


class ReplayBuffer:
    def __init__(self, max_size=10_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # Store one experience: (s, a, r, s', done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        # Random experiences break correlations between consecutive frames.
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states).to(device),
            torch.tensor(actions, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.stack(next_states).to(device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------
# DQN agent
# ---------------------------------------------------------


class BreakoutAgent(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.NUM_ACTIONS = env.action_space.n
        self.OBS_SHAPE = env.observation_space.shape

        # Q-network: state -> Q(s,a) for every possible action.
        self.q_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(210 * 160 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_ACTIONS),
        )

    def forward(self, state):
        # Predict one Q-value for each possible action.
        return self.q_network(state)

    def choose_action(self, state, epsilon):
        # Epsilon-greedy exploration: sometimes choose a random action.
        if random.random() < epsilon:
            return random.randint(0, self.NUM_ACTIONS - 1)

        # Otherwise choose the action with the largest Q-value.
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))

        return q_values.argmax(dim=1).item()


# ---------------------------------------------------------
# Convert Atari image to a tensor
# ---------------------------------------------------------


def to_tensor(obs, device):
    # uint8 [0,255] -> float [0,1]
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.0


# ---------------------------------------------------------
# One DQN learning step
# ---------------------------------------------------------


def train(agent, memory, optimizer, device, batch_size=32, gamma=0.99):
    # Get random experiences from replay memory.
    states, actions, rewards, next_states, dones = memory.sample(batch_size, device)

    # -----------------------------------------------------
    # 1. What does our network currently predict?
    #
    # Q(s, a)
    # -----------------------------------------------------

    all_q_values = agent(states)

    # We only want the Q-value of the action that was taken.
    q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # -----------------------------------------------------
    # 2. What SHOULD the Q-value be?
    #
    # target = reward + gamma * max Q(s', a')
    # -----------------------------------------------------

    with torch.no_grad():
        next_q_values = agent(next_states).max(dim=1).values

        # If the episode ended, there is no future reward.
        targets = rewards + gamma * next_q_values * (1 - dones)

    # -----------------------------------------------------
    # 3. Make our prediction closer to the target.
    # -----------------------------------------------------

    loss = nn.functional.mse_loss(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ---------------------------------------------------------
# Main training loop
# ---------------------------------------------------------


if __name__ == "__main__":
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    agent = BreakoutAgent(env).to(device)

    memory = ReplayBuffer()
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    EPISODES = 100
    BATCH_SIZE = 32
    EPSILON = 0.1

    for episode in tqdm(range(EPISODES)):
        obs, info = env.reset()
        state = to_tensor(obs, device)

        done = False
        total_reward = 0

        while not done:
            # 1. Choose action
            action = agent.choose_action(state, EPSILON)

            # 2. Perform action
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            next_state = to_tensor(next_obs, device)

            # 3. Remember what happened
            memory.add(state, action, reward, next_state, done)

            # 4. Learn from old experiences
            if len(memory) >= BATCH_SIZE:
                loss = train(agent, memory, optimizer, device, BATCH_SIZE)

            state = next_state
            total_reward += reward

        print(f"Episode {episode}: reward = {total_reward}")

    env.close()
