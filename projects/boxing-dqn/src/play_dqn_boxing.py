"""Render a trained Boxing agent / 可视化训练后的 Boxing 智能体。"""

import argparse
import random
import time
from pathlib import Path

import torch
from pettingzoo.atari import boxing_v2

from dqn_boxing import DQN, TRAIN_AGENT, append_frame, select_action, stacked_initial_state


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def play(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    frame_stack = int(checkpoint["frame_stack"])
    n_actions = int(checkpoint["n_actions"])

    policy_net = DQN(frame_stack, n_actions).to(device)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    policy_net.eval()

    opponent_net = DQN(frame_stack, n_actions).to(device)
    opponent_state = checkpoint.get("opponent_state_dict", checkpoint["model_state_dict"])
    opponent_net.load_state_dict(opponent_state)
    opponent_net.eval()

    env = boxing_v2.parallel_env(render_mode="human")
    rng = random.Random(args.seed)

    for episode in range(1, args.episodes + 1):
        observations, _ = env.reset(seed=args.seed + episode)
        states = {}
        frame_queues = {}
        for agent in env.agents:
            states[agent], frame_queues[agent] = stacked_initial_state(
                observations[agent], frame_stack
            )
        total_reward = 0.0

        try:
            while env.agents:
                actions = {}
                for agent in env.agents:
                    if agent == TRAIN_AGENT:
                        actions[agent] = select_action(
                            policy_net,
                            states[agent],
                            args.epsilon,
                            n_actions,
                            device,
                            rng,
                        )
                    elif args.opponent == "snapshot":
                        actions[agent] = select_action(
                            opponent_net,
                            states[agent],
                            args.opponent_epsilon,
                            n_actions,
                            device,
                            rng,
                        )
                    else:
                        actions[agent] = rng.randrange(n_actions)

                next_observations, rewards, terminations, truncations, _ = env.step(actions)
                total_reward += float(rewards.get(TRAIN_AGENT, 0.0))
                done = bool(
                    terminations.get(TRAIN_AGENT, False)
                    or truncations.get(TRAIN_AGENT, False)
                    or TRAIN_AGENT not in env.agents
                )
                if done:
                    break
                for agent in env.agents:
                    states[agent] = append_frame(frame_queues[agent], next_observations[agent])
                time.sleep(args.sleep)

        except KeyboardInterrupt:
            print("Stopped by user / 用户停止。")
            break

        print(f"episode={episode} opponent={args.opponent} reward={total_reward:.2f}")

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN play Atari Boxing.")
    parser.add_argument("--checkpoint", default="checkpoints/day4_boxing_best.pt")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--opponent", choices=["random", "snapshot"], default="random")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--opponent-epsilon", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=1 / 60)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    play(parse_args())
