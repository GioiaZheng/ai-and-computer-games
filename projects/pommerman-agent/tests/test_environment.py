from pommerman import agents

import pommerman


def test_official_ffa_environment_runs_one_step():
    agent_list = [agents.RandomAgent() for _ in range(4)]
    environment = pommerman.make(
        "PommeFFACompetition-v0", agent_list, render_mode="rgb_array"
    )
    observations = environment.reset()
    actions = environment.act(observations)
    next_observations, rewards, done, info = environment.step(actions)
    environment.close()
    assert len(next_observations) == 4
    assert len(rewards) == 4
    assert all(0 <= action < 6 for action in actions)
    assert not done
    assert "result" in info
