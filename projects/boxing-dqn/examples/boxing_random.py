import time

from pettingzoo.atari import boxing_v2


# Create the Boxing environment.
env = boxing_v2.parallel_env(render_mode="human")

# Start a new game.
env.reset(seed=42)

try:
    # Continue while the game has active players.
    while len(env.agents) > 0:

        # Create an empty dictionary for the players' actions.
        actions = {}

        # Select a random action for each player.
        for agent in env.agents:
            random_action = env.action_space(agent).sample()
            actions[agent] = random_action

        # Send the actions to the game.
        env.step(actions)

        # Slow the program down so that the game can be watched.
        time.sleep(1 / 60)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    # Close the game window after the game finishes or is stopped.
    env.close()

print("The game has finished.")
