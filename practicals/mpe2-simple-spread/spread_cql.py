import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from mpe2 import simple_spread_v3

from dqn import QNetwork, ReplayBuffer, train_dqn, update_target


# Training settings
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


# Two agents and two landmarks.
# local_ratio=0.0 means both agents use the global team reward.
env = simple_spread_v3.parallel_env(
    N=2,
    local_ratio=0.0,
    max_cycles=25,
    continuous_actions=False,
)

observations, infos = env.reset(seed=SEED)


# ------------------------------------------------------------
# Centralized Q-Learning
#
# Instead of one Q-network per agent, there is one Q-network.
#
# Input:
#   observation of agent_0 + observation of agent_1
#
# Output:
#   Q-value for every joint action
#
# Each agent has 5 actions, so there are:
#   5 x 5 = 25 joint actions
# ------------------------------------------------------------

observation_size_0 = env.observation_space("agent_0").shape[0]
observation_size_1 = env.observation_space("agent_1").shape[0]

action_size_0 = env.action_space("agent_0").n
action_size_1 = env.action_space("agent_1").n

joint_observation_size = (
    observation_size_0 + observation_size_1
)

joint_action_size = action_size_0 * action_size_1


q_network = QNetwork(
    joint_observation_size,
    joint_action_size,
).to(device)

target_network = QNetwork(
    joint_observation_size,
    joint_action_size,
).to(device)

update_target(q_network, target_network)

optimizer = torch.optim.Adam(
    q_network.parameters(),
    lr=LEARNING_RATE,
)

replay_buffer = ReplayBuffer(
    BUFFER_SIZE,
    joint_observation_size,
    device,
    seed=SEED,
)


game = 0
game_reward = 0.0
recent_rewards = deque(maxlen=100)


try:
    for timestep in range(TOTAL_TIMESTEPS):

        # Start a new game after the previous one ends.
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


        # Epsilon decreases from 1.0 to 0.05.
        epsilon = max(
            END_EPSILON,
            START_EPSILON
            - (START_EPSILON - END_EPSILON)
            * timestep
            / EPSILON_DECAY_STEPS,
        )


        # Combine both agents' observations into one joint state.
        joint_observation = np.concatenate(
            [
                observations["agent_0"],
                observations["agent_1"],
            ]
        )


        # Choose one joint action.
        if random.random() < epsilon:
            action_0 = env.action_space("agent_0").sample()
            action_1 = env.action_space("agent_1").sample()

            joint_action = (
                action_0 * action_size_1
                + action_1
            )

        else:
            joint_action = q_network.action(
                joint_observation,
                device,
            )

            # Convert the joint action back into the two
            # individual agent actions.
            action_0 = joint_action // action_size_1
            action_1 = joint_action % action_size_1


        actions = {
            "agent_0": action_0,
            "agent_1": action_1,
        }


        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)


        # Both agents receive the same team reward.
        reward = rewards["agent_0"]
        game_reward += reward


        done = (
            terminations["agent_0"]
            or truncations["agent_0"]
        )


        if done:
            next_joint_observation = np.zeros(
                joint_observation_size,
                dtype=np.float32,
            )
        else:
            next_joint_observation = np.concatenate(
                [
                    observations["agent_0"],
                    observations["agent_1"],
                ]
            )


        # Store one centralized transition.
        replay_buffer.add(
            joint_observation,
            joint_action,
            reward,
            next_joint_observation,
            done,
        )


        # Train the centralized Q-network.
        if (
            timestep >= LEARNING_STARTS
            and len(replay_buffer) >= BATCH_SIZE
        ):
            train_dqn(
                q_network,
                target_network,
                replay_buffer,
                optimizer,
                BATCH_SIZE,
                GAMMA,
            )


        # Periodically copy the Q-network to the target network.
        if (
            timestep >= LEARNING_STARTS
            and timestep % TARGET_UPDATE_FREQUENCY == 0
        ):
            update_target(
                q_network,
                target_network,
            )


finally:
    env.close()


# Save the centralized Q-network.
Path("checkpoints").mkdir(exist_ok=True)

torch.save(
    q_network.state_dict(),
    "checkpoints/spread_cql.pth",
)

print("Training complete.")
