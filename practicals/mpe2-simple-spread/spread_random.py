import time

from mpe2 import simple_spread_v3


# Create the environment.
env = simple_spread_v3.parallel_env(
    render_mode="human",
    N=2,
    local_ratio=0.0,
    max_cycles=25,
    continuous_actions=False,
)

# Start a new game.
observations, infos = env.reset(seed=42)

team_reward = 0.0

# Continue while the game has active agents.
while len(env.agents) > 0:

    # Agent 0 selects a random action.
    action_0 = env.action_space("agent_0").sample()

    # Agent 1 selects a random action.
    action_1 = env.action_space("agent_1").sample()

    # Both agents act at the same environment step.
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

    # Both agents receive the shared team reward.
    team_reward += rewards["agent_0"]

    # Slow down the game so that it is easy to watch.
    time.sleep(1 / 10)

env.close()

print(f"Team reward: {team_reward:.3f}")
