"""Training wrappers for staged Combat Tank curricula."""

from __future__ import annotations

import gymnasium as gym

from .scripted_agent import SpawnHunterAgent


class ScriptedPrefixWrapper(gym.Wrapper):
    """Start each episode after a deterministic maze-navigation demonstration."""

    def __init__(self, env, prefix_steps: int):
        if prefix_steps < 0:
            raise ValueError("prefix_steps must be non-negative")
        super().__init__(env)
        self.prefix_steps = prefix_steps

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        expert = SpawnHunterAgent(role=self.env.learner)
        prefix_score = 0.0
        completed_steps = 0
        for _ in range(self.prefix_steps):
            action = expert.get_action(observation)
            observation, _, terminated, truncated, step_info = self.env.step_game_action(action)
            official_reward = float(step_info.get("official_reward", 0.0))
            expert.observe_reward(official_reward)
            prefix_score += official_reward
            completed_steps += 1
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
                expert = SpawnHunterAgent(role=self.env.learner)
                prefix_score = 0.0
                completed_steps = 0
        info = dict(info)
        info["prefix_steps"] = completed_steps
        info["prefix_official_score"] = prefix_score
        return observation, info
