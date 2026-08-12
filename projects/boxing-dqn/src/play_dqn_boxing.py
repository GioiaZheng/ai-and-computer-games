import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from pettingzoo.atari import boxing_v2

from dqn_boxing import DQN, TRAIN_AGENT, append_frame, preprocess_frame


def choose_action(policy_net, state, n_actions, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state[None, ...], device=device)
        return int(policy_net(state_tensor).argmax(dim=1).item())


def play(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    frame_stack = int(checkpoint["frame_stack"])
    n_actions = int(checkpoint["n_actions"])

    device = torch.device(args.device)
    policy_net = DQN(frame_stack, n_actions).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()

    env = boxing_v2.parallel_env(render_mode="human")

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode)
        frame = preprocess_frame(obs[TRAIN_AGENT])
        frames = deque([frame.copy() for _ in range(frame_stack)], maxlen=frame_stack)
        state = np.stack(frames, axis=0)
        total_reward = 0.0

        try:
            while len(env.agents) > 0:
                actions = {}
                for agent in env.agents:
                    if agent == TRAIN_AGENT:
                        actions[agent] = choose_action(
                            policy_net, state, n_actions, args.epsilon, device
                        )
                    else:
                        actions[agent] = env.action_space(agent).sample()

                obs, rewards, terminations, truncations, _ = env.step(actions)
                total_reward += float(rewards.get(TRAIN_AGENT, 0.0))

                done = bool(
                    terminations.get(TRAIN_AGENT, False)
                    or truncations.get(TRAIN_AGENT, False)
                    or TRAIN_AGENT not in env.agents
                )
                if done:
                    break

                state = append_frame(frames, obs[TRAIN_AGENT])
                time.sleep(args.sleep)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break

        print(f"episode={episode} reward={total_reward:.2f}")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN play Atari Boxing.")
    parser.add_argument("--checkpoint", default="checkpoints/dqn_boxing.pt")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sleep", type=float, default=1 / 60)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    play(parse_args())
