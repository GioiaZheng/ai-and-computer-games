import time

import numpy as np
import pygame
import torch
from mpe2 import simple_spread_v3

from dqn import QNetwork


EVALUATION_GAMES = 100
WATCH_GAMES = 5
SEED = 10_000
FPS = 10


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def make_env(render_mode=None):
    return simple_spread_v3.parallel_env(
        render_mode=render_mode,
        N=2,
        local_ratio=0.0,
        max_cycles=25,
        continuous_actions=False,
    )


# ------------------------------------------------------------
# Load IQL and CQL networks.
# ------------------------------------------------------------

env = make_env()

observation_size_0 = env.observation_space("agent_0").shape[0]
observation_size_1 = env.observation_space("agent_1").shape[0]

action_size_0 = env.action_space("agent_0").n
action_size_1 = env.action_space("agent_1").n


# IQL: one Q-network per agent.
iql_network_0 = QNetwork(
    observation_size_0,
    action_size_0,
).to(device)

iql_network_1 = QNetwork(
    observation_size_1,
    action_size_1,
).to(device)

iql_network_0.load_state_dict(
    torch.load(
        "checkpoints/spread_iql_agent_0.pth",
        map_location=device,
        weights_only=True,
    )
)

iql_network_1.load_state_dict(
    torch.load(
        "checkpoints/spread_iql_agent_1.pth",
        map_location=device,
        weights_only=True,
    )
)

iql_network_0.eval()
iql_network_1.eval()


# CQL: one centralized Q-network.
joint_observation_size = (
    observation_size_0 + observation_size_1
)

joint_action_size = action_size_0 * action_size_1

cql_network = QNetwork(
    joint_observation_size,
    joint_action_size,
).to(device)

cql_network.load_state_dict(
    torch.load(
        "checkpoints/spread_cql.pth",
        map_location=device,
        weights_only=True,
    )
)

cql_network.eval()

env.close()


# ------------------------------------------------------------
# Policy helpers.
# ------------------------------------------------------------

def random_actions(env):
    return {
        "agent_0": env.action_space("agent_0").sample(),
        "agent_1": env.action_space("agent_1").sample(),
    }


def iql_actions(observations):
    action_0 = iql_network_0.action(
        observations["agent_0"],
        device,
    )

    action_1 = iql_network_1.action(
        observations["agent_1"],
        device,
    )

    return {
        "agent_0": action_0,
        "agent_1": action_1,
    }


def cql_actions(observations):
    joint_observation = np.concatenate(
        [
            observations["agent_0"],
            observations["agent_1"],
        ]
    )

    joint_action = cql_network.action(
        joint_observation,
        device,
    )

    action_0 = joint_action // action_size_1
    action_1 = joint_action % action_size_1

    return {
        "agent_0": action_0,
        "agent_1": action_1,
    }


# ------------------------------------------------------------
# Play one complete game without rendering.
# ------------------------------------------------------------

def play_random_game(env, seed):
    observations, infos = env.reset(seed=seed)

    env.action_space("agent_0").seed(seed)
    env.action_space("agent_1").seed(seed + 1)

    total_reward = 0.0

    while env.agents:
        actions = random_actions(env)

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        total_reward += rewards["agent_0"]

    return total_reward


def play_iql_game(env, seed):
    observations, infos = env.reset(seed=seed)

    total_reward = 0.0

    while env.agents:
        actions = iql_actions(observations)

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        total_reward += rewards["agent_0"]

    return total_reward


def play_cql_game(env, seed):
    observations, infos = env.reset(seed=seed)

    total_reward = 0.0

    while env.agents:
        actions = cql_actions(observations)

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = env.step(actions)

        total_reward += rewards["agent_0"]

    return total_reward


# ============================================================
# 1. NUMERICAL COMPARISON
# ============================================================

env = make_env()

random_rewards = []
iql_rewards = []
cql_rewards = []

try:
    for game in range(EVALUATION_GAMES):
        seed = SEED + game

        random_rewards.append(
            play_random_game(env, seed)
        )

        iql_rewards.append(
            play_iql_game(env, seed)
        )

        cql_rewards.append(
            play_cql_game(env, seed)
        )

finally:
    env.close()


random_mean = np.mean(random_rewards)
random_std = np.std(random_rewards)

iql_mean = np.mean(iql_rewards)
iql_std = np.std(iql_rewards)

cql_mean = np.mean(cql_rewards)
cql_std = np.std(cql_rewards)


print(
    f"Evaluation over {EVALUATION_GAMES} unseen games"
)
print()

print(
    f"Random: {random_mean:7.3f} "
    f"+/- {random_std:.3f}"
)

print(
    f"IQL:    {iql_mean:7.3f} "
    f"+/- {iql_std:.3f}"
)

print(
    f"CQL:    {cql_mean:7.3f} "
    f"+/- {cql_std:.3f}"
)

print()

print(
    f"IQL improvement over random: "
    f"{iql_mean - random_mean:.3f}"
)

print(
    f"CQL improvement over random: "
    f"{cql_mean - random_mean:.3f}"
)

print(
    f"CQL difference from IQL:      "
    f"{cql_mean - iql_mean:.3f}"
)


# ============================================================
# 2. SIDE-BY-SIDE VISUAL COMPARISON
# ============================================================

print()
print("Showing Random, IQL, and CQL side by side...")


random_env = make_env(render_mode="rgb_array")
iql_env = make_env(render_mode="rgb_array")
cql_env = make_env(render_mode="rgb_array")


pygame.init()
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

window = None
running = True


try:
    for game in range(WATCH_GAMES):
        if not running:
            break

        seed = SEED + EVALUATION_GAMES + game

        random_observations, _ = random_env.reset(seed=seed)
        iql_observations, _ = iql_env.reset(seed=seed)
        cql_observations, _ = cql_env.reset(seed=seed)

        random_env.action_space("agent_0").seed(seed)
        random_env.action_space("agent_1").seed(seed + 1)

        random_reward = 0.0
        iql_reward = 0.0
        cql_reward = 0.0


        while (
            random_env.agents
            and iql_env.agents
            and cql_env.agents
            and running
        ):

            # Random policy.
            random_action = random_actions(random_env)

            (
                random_observations,
                random_rewards_step,
                _,
                _,
                _,
            ) = random_env.step(random_action)

            random_reward += random_rewards_step["agent_0"]


            # IQL policy.
            iql_action = iql_actions(iql_observations)

            (
                iql_observations,
                iql_rewards_step,
                _,
                _,
                _,
            ) = iql_env.step(iql_action)

            iql_reward += iql_rewards_step["agent_0"]


            # CQL policy.
            cql_action = cql_actions(cql_observations)

            (
                cql_observations,
                cql_rewards_step,
                _,
                _,
                _,
            ) = cql_env.step(cql_action)

            cql_reward += cql_rewards_step["agent_0"]


            # Render all three environments.
            random_frame = random_env.render()
            iql_frame = iql_env.render()
            cql_frame = cql_env.render()


            # All three environments have the same frame size.
            height, width, _ = random_frame.shape
            label_height = 80

            if window is None:
                window = pygame.display.set_mode(
                    (
                        width * 3,
                        height + label_height,
                    )
                )

                pygame.display.set_caption(
                    "Simple Spread: Random vs IQL vs CQL"
                )


            # Convert NumPy RGB arrays to Pygame surfaces.
            random_surface = pygame.surfarray.make_surface(
                np.transpose(random_frame, (1, 0, 2))
            )

            iql_surface = pygame.surfarray.make_surface(
                np.transpose(iql_frame, (1, 0, 2))
            )

            cql_surface = pygame.surfarray.make_surface(
                np.transpose(cql_frame, (1, 0, 2))
            )


            window.fill((255, 255, 255))

            window.blit(
                random_surface,
                (0, label_height),
            )

            window.blit(
                iql_surface,
                (width, label_height),
            )

            window.blit(
                cql_surface,
                (width * 2, label_height),
            )


            # Game counter, centered across the whole window.
            game_text = font.render(
                f"Game {game + 1} / {WATCH_GAMES}",
                True,
                (0, 0, 0),
            )

            window.blit(
                game_text,
                (
                    (width * 3) // 2
                    - game_text.get_width() // 2,
                    10,
                ),
            )

            # Method labels, one above each environment.
            random_text = font.render(
                "RANDOM",
                True,
                (0, 0, 0),
            )

            iql_text = font.render(
                "IQL",
                True,
                (0, 0, 0),
            )

            cql_text = font.render(
                "CQL",
                True,
                (0, 0, 0),
            )

            window.blit(
                random_text,
                (
                    width // 2
                    - random_text.get_width() // 2,
                    45,
                ),
            )

            window.blit(
                iql_text,
                (
                    width
                    + width // 2
                    - iql_text.get_width() // 2,
                    45,
                ),
            )

            window.blit(
                cql_text,
                (
                    width * 2
                    + width // 2
                    - cql_text.get_width() // 2,
                    45,
                ),
            )


            pygame.display.flip()


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False


            clock.tick(FPS)


        print(
            f"Game {game + 1} | "
            f"Random {random_reward:7.3f} | "
            f"IQL {iql_reward:7.3f} | "
            f"CQL {cql_reward:7.3f}"
        )


finally:
    random_env.close()
    iql_env.close()
    cql_env.close()
    pygame.quit()
