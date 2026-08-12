import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from mpe2 import simple_spread_v3

from dqn import QNetwork, ReplayBuffer, train_dqn, update_target


SEED = 42
TOTAL_TIMESTEPS = 50_000

LEARNING_RATE = 1e-3
GAMMA = 0.95

BUFFER_SIZE = 10_000
BATCH_SIZE = 64
LEARNING_STARTS = 1_000
TARGET_UPDATE_FREQUENCY = 500

START_EPSILON = 1.0
END_EPSILON = 0.05
EPSILON_DECAY_STEPS = 10_000


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


env = simple_spread_v3.parallel_env(
    N=2,
    local_ratio=0.0,
    max_cycles=25,
    continuous_actions=False,
)

observations, infos = env.reset(seed=SEED)


# Agent 0
observation_size_0 = env.observation_space("agent_0").shape[0]
action_size_0 = env.action_space("agent_0").n

q_network_0 = QNetwork(
    observation_size_0,
    action_size_0,
).to(device)

target_network_0 = QNetwork(
    observation_size_0,
    action_size_0,
).to(device)

update_target(q_network_0, target_network_0)

optimizer_0 = torch.optim.Adam(
    q_network_0.parameters(),
    lr=LEARNING_RATE,
)

replay_buffer_0 = ReplayBuffer(
    BUFFER_SIZE,
    observation_size_0,
    device,
    seed=SEED,
)


# Agent 1
observation_size_1 = env.observation_space("agent_1").shape[0]
action_size_1 = env.action_space("agent_1").n

q_network_1 = QNetwork(
    observation_size_1,
    action_size_1,
).to(device)

target_network_1 = QNetwork(
    observation_size_1,
    action_size_1,
).to(device)

update_target(q_network_1, target_network_1)

optimizer_1 = torch.optim.Adam(
    q_network_1.parameters(),
    lr=LEARNING_RATE,
)

replay_buffer_1 = ReplayBuffer(
    BUFFER_SIZE,
    observation_size_1,
    device,
    seed=SEED + 1,
)


game = 0
game_reward = 0.0
recent_rewards = deque(maxlen=100)


try:
    for timestep in range(TOTAL_TIMESTEPS):

        if not env.agents:
            game += 1
            recent_rewards.append(game_reward)

            if game % 100 == 0:
                print(
                    f"Game {game:4d} | "
                    f"mean reward {np.mean(recent_rewards):7.3f}"
                )

            observations, infos = env.reset(
                seed=SEED + game
            )

            game_reward = 0.0


        epsilon = max(
            END_EPSILON,
            START_EPSILON
            - (START_EPSILON - END_EPSILON)
            * timestep
            / EPSILON_DECAY_STEPS,
        )


        # Agent 0 chooses its own action.
        if random.random() < epsilon:
            action_0 = env.action_space("agent_0").sample()
        else:
            action_0 = q_network_0.action(
                observations["agent_0"],
                device,
            )


        # Agent 1 chooses its own action.
        if random.random() < epsilon:
            action_1 = env.action_space("agent_1").sample()
        else:
            action_1 = q_network_1.action(
                observations["agent_1"],
                device,
            )


        actions = {
            "agent_0": action_0,
            "agent_1": action_1,
        }


        old_observation_0 = observations["agent_0"]
        old_observation_1 = observations["agent_1"]


        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)


        # With local_ratio=0.0, both agents receive the team reward.
        game_reward += rewards["agent_0"]


        done_0 = (
            terminations["agent_0"]
            or truncations["agent_0"]
        )

        done_1 = (
            terminations["agent_1"]
            or truncations["agent_1"]
        )


        if done_0:
            next_observation_0 = np.zeros(
                observation_size_0,
                dtype=np.float32,
            )
        else:
            next_observation_0 = observations["agent_0"]


        if done_1:
            next_observation_1 = np.zeros(
                observation_size_1,
                dtype=np.float32,
            )
        else:
            next_observation_1 = observations["agent_1"]


        replay_buffer_0.add(
            old_observation_0,
            action_0,
            rewards["agent_0"],
            next_observation_0,
            done_0,
        )


        replay_buffer_1.add(
            old_observation_1,
            action_1,
            rewards["agent_1"],
            next_observation_1,
            done_1,
        )


        if (
            timestep >= LEARNING_STARTS
            and len(replay_buffer_0) >= BATCH_SIZE
        ):
            train_dqn(
                q_network_0,
                target_network_0,
                replay_buffer_0,
                optimizer_0,
                BATCH_SIZE,
                GAMMA,
            )

            train_dqn(
                q_network_1,
                target_network_1,
                replay_buffer_1,
                optimizer_1,
                BATCH_SIZE,
                GAMMA,
            )


        if (
            timestep >= LEARNING_STARTS
            and timestep % TARGET_UPDATE_FREQUENCY == 0
        ):
            update_target(
                q_network_0,
                target_network_0,
            )

            update_target(
                q_network_1,
                target_network_1,
            )


finally:
    env.close()


Path("checkpoints").mkdir(exist_ok=True)

torch.save(
    q_network_0.state_dict(),
    "checkpoints/spread_iql_agent_0.pth",
)

torch.save(
    q_network_1.state_dict(),
    "checkpoints/spread_iql_agent_1.pth",
)

print("Training complete.")
