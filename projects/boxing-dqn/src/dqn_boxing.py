import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from pettingzoo.atari import boxing_v2
from torch import nn


TRAIN_AGENT = "first_0"
FRAME_SIZE = 84


def preprocess_frame(observation):
    """Convert an Atari RGB frame to a small grayscale uint8 frame."""
    gray = (
        0.299 * observation[:, :, 0]
        + 0.587 * observation[:, :, 1]
        + 0.114 * observation[:, :, 2]
    ).astype(np.uint8)
    y_idx = np.linspace(0, gray.shape[0] - 1, FRAME_SIZE).astype(np.int32)
    x_idx = np.linspace(0, gray.shape[1] - 1, FRAME_SIZE).astype(np.int32)
    return gray[y_idx][:, x_idx]


def stacked_initial_state(observation, frame_stack):
    frame = preprocess_frame(observation)
    frames = deque([frame.copy() for _ in range(frame_stack)], maxlen=frame_stack)
    return np.stack(frames, axis=0), frames


def append_frame(frames, observation):
    frames.append(preprocess_frame(observation))
    return np.stack(frames, axis=0)


class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.states[idx], device=device)
        actions = torch.as_tensor(self.actions[idx], device=device)
        rewards = torch.as_tensor(self.rewards[idx], device=device)
        next_states = torch.as_tensor(self.next_states[idx], device=device)
        dones = torch.as_tensor(self.dones[idx], device=device)
        return states, actions, rewards, next_states, dones


class DQN(nn.Module):
    def __init__(self, frame_stack, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(frame_stack, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, frame_stack, FRAME_SIZE, FRAME_SIZE)
            feature_dim = self.features(sample).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.head(self.features(x))


def linear_epsilon(step, start, end, decay_steps):
    if decay_steps <= 0:
        return end
    progress = min(step / decay_steps, 1.0)
    return start + progress * (end - start)


def select_action(policy_net, state, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state[None, ...], device=device)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def optimize(policy_net, target_net, replay, optimizer, args, device):
    if len(replay) < max(args.learning_starts, args.batch_size):
        return None

    states, actions, rewards, next_states, dones = replay.sample(args.batch_size, device)
    q_values = policy_net(states).gather(1, actions[:, None]).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values
        targets = rewards + args.gamma * next_q_values * (1.0 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss.item())


def save_checkpoint(path, policy_net, optimizer, steps, episodes, args, n_actions):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": steps,
            "episodes": episodes,
            "frame_stack": args.frame_stack,
            "n_actions": n_actions,
            "args": vars(args),
        },
        path,
    )


def write_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    env = boxing_v2.parallel_env(render_mode=None)
    obs, _ = env.reset(seed=args.seed)
    n_actions = env.action_space(TRAIN_AGENT).n
    state_shape = (args.frame_stack, FRAME_SIZE, FRAME_SIZE)

    policy_net = DQN(args.frame_stack, n_actions).to(device)
    target_net = DQN(args.frame_stack, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size, state_shape)
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.results)

    total_steps = 0
    print(f"Training {TRAIN_AGENT} with DQN against a random opponent.")
    print(f"Device: {device}; actions: {n_actions}; replay size: {args.replay_size}")

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        state, frames = stacked_initial_state(obs[TRAIN_AGENT], args.frame_stack)
        episode_reward = 0.0
        losses = []

        for episode_step in range(1, args.max_steps + 1):
            epsilon = linear_epsilon(total_steps, args.eps_start, args.eps_end, args.eps_decay_steps)
            action = select_action(policy_net, state, epsilon, n_actions, device)

            actions = {}
            for agent in env.agents:
                if agent == TRAIN_AGENT:
                    actions[agent] = action
                else:
                    actions[agent] = env.action_space(agent).sample()

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            reward = float(rewards.get(TRAIN_AGENT, 0.0))
            done = bool(
                terminations.get(TRAIN_AGENT, False)
                or truncations.get(TRAIN_AGENT, False)
                or TRAIN_AGENT not in env.agents
            )

            if TRAIN_AGENT in next_obs:
                next_state = append_frame(frames, next_obs[TRAIN_AGENT])
            else:
                next_state = state.copy()

            replay.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % args.train_freq == 0:
                loss = optimize(policy_net, target_net, replay, optimizer, args, device)
                if loss is not None:
                    losses.append(loss)

            if total_steps % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        mean_loss = float(np.mean(losses)) if losses else 0.0
        row = {
            "episode": episode,
            "total_steps": total_steps,
            "episode_steps": episode_step,
            "reward": round(episode_reward, 4),
            "epsilon": round(epsilon, 4),
            "mean_loss": round(mean_loss, 6),
            "replay_size": len(replay),
        }
        write_result(result_path, row)
        print(
            f"episode={episode:04d} steps={episode_step:04d} "
            f"reward={episode_reward:7.2f} eps={epsilon:.3f} loss={mean_loss:.5f}"
        )

        if episode % args.save_every == 0:
            save_checkpoint(checkpoint_path, policy_net, optimizer, total_steps, episode, args, n_actions)
            print(f"saved checkpoint: {checkpoint_path}")

    save_checkpoint(checkpoint_path, policy_net, optimizer, total_steps, args.episodes, args, n_actions)
    env.close()
    print(f"Training finished. Final checkpoint: {checkpoint_path}")
    print(f"Results CSV: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight DQN baseline for PettingZoo Atari Boxing.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--replay-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.1)
    parser.add_argument("--eps-decay-steps", type=int, default=20000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint", default="checkpoints/dqn_boxing.pt")
    parser.add_argument("--results", default="results/dqn_boxing_training.csv")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
