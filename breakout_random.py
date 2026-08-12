import time

import ale_py
import gymnasium as gym


# Make the Atari environments available to Gymnasium.
gym.register_envs(ale_py)

# Create the Breakout environment.
env = gym.make(
    "ALE/Breakout-v5",
    render_mode="human",
)

# Start a new game.
observation, info = env.reset(seed=42)

# The game has not finished yet.
game_finished = False

try:
    # Continue until the game finishes.
    while not game_finished:

        # Select a random action.
        action = env.action_space.sample()

        # Send the action to the game.
        observation, reward, terminated, truncated, info = env.step(action)

        # Check whether the game has finished.
        game_finished = terminated or truncated

        # Slow the program down so that the game can be watched.
        time.sleep(1 / 60)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    # Close the game window.
    env.close()

print("The game has finished.")
