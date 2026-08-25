"""Adapters for the instructor-provided Combat Tank evaluation pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import supersuit as ss
from pettingzoo.atari import combat_tank_v2


AGENTS = ("first_0", "second_0")
NUM_ACTIONS = 18
TANK_VALUES = {"first_0": 225, "second_0": 193}


def official_tank_center(
    observation: np.ndarray, role: str
) -> tuple[float, float] | None:
    """Locate a tank in the newest official grayscale frame."""
    frame = np.asarray(observation)
    if frame.shape != (84, 84, 6):
        return None
    latest_frame = frame[..., 3]
    mask = latest_frame == TANK_VALUES[role]
    mask[:12] = False
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return None
    return float(np.median(xs)), float(np.median(ys))


def create_official_environment(render_mode=None):
    """Build the exact environment and wrapper order supplied by the instructor."""
    env = combat_tank_v2.parallel_env(
        render_mode=render_mode,
        has_maze=True,
        is_invisible=False,
        billiard_hit=False,
    )
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    return env


OpponentPolicy = Callable[[np.ndarray, str], int]


class OfficialSingleAgentCombatEnv(gym.Env):
    """Expose one official-pipeline player while a policy controls the opponent."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        opponent_policy: OpponentPolicy | None = None,
        fixed_role: str | None = None,
        seed: int = 0,
        render_mode=None,
        exploration_bonus_scale: float = 0.0,
        idle_penalty_scale: float = 0.0,
    ):
        super().__init__()
        if fixed_role not in (None, *AGENTS):
            raise ValueError(f"Unknown role: {fixed_role}")
        self.env = create_official_environment(render_mode=render_mode)
        self.opponent_policy = opponent_policy
        self.fixed_role = fixed_role
        self.initial_seed = seed
        self.exploration_bonus_scale = exploration_bonus_scale
        self.idle_penalty_scale = idle_penalty_scale
        self.episode_index = 0
        self.learner = AGENTS[0]
        self.opponent = AGENTS[1]
        self.observations: dict[str, np.ndarray] = {}
        self.episode_return = 0.0
        self.episode_steps = 0
        self.position_visits: Counter[tuple[int, int]] = Counter()
        self.lifetime_position_visits: Counter[tuple[str, int, int]] = Counter()
        self.previous_center: tuple[float, float] | None = None
        self.idle_steps = 0
        self.observation_space = self.env.observation_space(AGENTS[0])
        self.action_space = self.env.action_space(AGENTS[0])
        if self.action_space.n != NUM_ACTIONS:
            raise RuntimeError(
                f"Official Combat Tank action space changed: {self.action_space}"
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else self.initial_seed + self.episode_index
        observations, infos = self.env.reset(seed=actual_seed)
        if self.fixed_role is None:
            self.learner = AGENTS[int(self.np_random.integers(0, len(AGENTS)))]
        else:
            self.learner = self.fixed_role
        self.opponent = AGENTS[1] if self.learner == AGENTS[0] else AGENTS[0]
        if self.opponent_policy is not None and hasattr(self.opponent_policy, "reset"):
            self.opponent_policy.reset(self.opponent)
        self.observations = {
            agent: np.asarray(observation).copy()
            for agent, observation in observations.items()
        }
        self.episode_return = 0.0
        self.episode_steps = 0
        self.position_visits.clear()
        self.previous_center = official_tank_center(
            self.observations[self.learner], self.learner
        )
        self.idle_steps = 0
        if self.previous_center is not None:
            self.position_visits[self._position_cell(self.previous_center)] += 1
        self.episode_index += 1
        info = dict(infos.get(self.learner, {}))
        info["role"] = self.learner
        return self.observations[self.learner], info

    @staticmethod
    def _position_cell(center: tuple[float, float]) -> tuple[int, int]:
        return int(center[0] // 7), int(center[1] // 7)

    def step(self, action):
        learner_action = int(action)
        if not self.action_space.contains(learner_action):
            raise ValueError(f"Learner returned invalid action {learner_action}")
        if self.opponent_policy is None:
            opponent_action = int(self.env.action_space(self.opponent).sample())
        else:
            opponent_action = int(
                self.opponent_policy(self.observations[self.opponent], self.opponent)
            )
        if not self.env.action_space(self.opponent).contains(opponent_action):
            raise ValueError(f"Opponent returned invalid action {opponent_action}")

        observations, rewards, terminations, truncations, infos = self.env.step(
            {self.learner: learner_action, self.opponent: opponent_action}
        )
        if self.opponent_policy is not None and hasattr(
            self.opponent_policy, "observe_reward"
        ):
            self.opponent_policy.observe_reward(
                float(rewards.get(self.opponent, 0.0)), self.opponent
            )
        for agent, observation in observations.items():
            self.observations[agent] = np.asarray(observation).copy()

        official_reward = float(rewards.get(self.learner, 0.0))
        reward = official_reward
        terminated = bool(terminations.get(self.learner, False))
        truncated = bool(truncations.get(self.learner, False))
        self.episode_return += reward
        self.episode_steps += 1
        learner_observation = self.observations.get(
            self.learner,
            np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype),
        )
        center = official_tank_center(learner_observation, self.learner)
        if center is not None:
            cell = self._position_cell(center)
            if self.position_visits[cell] == 0 and self.exploration_bonus_scale > 0:
                lifetime_key = (self.learner, cell[0], cell[1])
                self.lifetime_position_visits[lifetime_key] += 1
                reward += self.exploration_bonus_scale / np.sqrt(
                    self.lifetime_position_visits[lifetime_key]
                )
            self.position_visits[cell] += 1
            if self.previous_center is not None:
                movement = abs(center[0] - self.previous_center[0]) + abs(
                    center[1] - self.previous_center[1]
                )
                if movement < 0.5:
                    self.idle_steps += 1
                    if self.idle_steps > 12 and self.idle_penalty_scale > 0:
                        reward -= self.idle_penalty_scale * min(
                            (self.idle_steps - 12) / 60.0, 1.0
                        )
                else:
                    self.idle_steps = 0
            self.previous_center = center
        self.episode_return += reward - official_reward
        info = dict(infos.get(self.learner, {}))
        info.update(
            role=self.learner,
            official_reward=official_reward,
            episode_return=self.episode_return,
            episode_steps=self.episode_steps,
            episode_unique_cells=len(self.position_visits),
            idle_steps=self.idle_steps,
        )
        return learner_observation, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def describe_official_environment() -> dict[str, object]:
    """Return a small compatibility report used before training and packaging."""
    env = create_official_environment(render_mode=None)
    try:
        observations, _ = env.reset(seed=82_026)
        return {
            "agents": tuple(observations),
            "observations": {
                agent: {
                    "shape": tuple(np.asarray(observation).shape),
                    "dtype": str(np.asarray(observation).dtype),
                    "space": str(env.observation_space(agent)),
                }
                for agent, observation in observations.items()
            },
            "actions": {
                agent: str(env.action_space(agent)) for agent in observations
            },
        }
    finally:
        env.close()


if __name__ == "__main__":
    print(describe_official_environment())
