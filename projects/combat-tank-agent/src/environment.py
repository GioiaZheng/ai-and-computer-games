"""PettingZoo Combat Tank adapters used by training and evaluation."""

from __future__ import annotations

from collections import Counter, deque
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.atari import combat_tank_v2
from PIL import Image


FRAME_SIZE = 84
FRAME_STACK = 4
FRAME_CHANNELS = 3
REFERENCE_FRAME_WIDTH = 160
REFERENCE_FRAME_HEIGHT = 256
NUM_ACTIONS = 18
AGENTS = ("first_0", "second_0")
ACTION_SETS = {
    "all": tuple(range(NUM_ACTIONS)),
    "fire": (1, 10, 11, 12, 13, 14, 15, 16, 17),
    "fire_cardinal": (10, 11, 12, 13),
    "fire_diagonal": (14, 15, 16, 17),
    "sweep": (11, 12),
}
OFFICIAL_ENVIRONMENT = "atari/combat_tank-v2"
OFFICIAL_ENV_KWARGS = {
    "has_maze": True,
    "is_invisible": False,
    "billiard_hit": False,
}
TANK_COLORS = {
    "first_0": np.asarray((111, 111, 225), dtype=np.uint8),
    "second_0": np.asarray((198, 111, 193), dtype=np.uint8),
}
# The raw Combat Tank frame is 160x256. The narrow horizontal passages
# through the side walls are centred near y=91 and y=186.
LEFT_GATES = ((28.0, 91.0), (28.0, 186.0))
RIGHT_GATES = ((132.0, 91.0), (132.0, 186.0))


def create_environment(render_mode=None):
    """Create the parallel training form of the official scoring game."""
    return combat_tank_v2.parallel_env(
        render_mode=render_mode,
        **OFFICIAL_ENV_KWARGS,
    )


def create_scoring_environment(render_mode=None):
    """Reproduce the instructor-provided AEC scoring configuration exactly."""
    return combat_tank_v2.env(
        render_mode=render_mode,
        **OFFICIAL_ENV_KWARGS,
    )


def preprocess_frame(observation: np.ndarray) -> np.ndarray:
    """Resize one RGB game frame while preserving player-identifying colors."""
    frame = np.asarray(observation, dtype=np.uint8)
    image = Image.fromarray(frame)
    resized = image.resize((FRAME_SIZE, FRAME_SIZE), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _tank_pixels(
    observation: np.ndarray, agent: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    """Return tank-colored pixel coordinates below the score display."""
    frame = np.asarray(observation, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    # A stacked, resized training observation stores the newest RGB frame in
    # its final three channels. Return coordinates in the original 160x256
    # reference system so navigation code behaves identically for both forms.
    if frame.shape[-1] > 3:
        frame = frame[..., -3:]
    color_distance = np.linalg.norm(
        frame.astype(np.int16) - TANK_COLORS[agent].astype(np.int16), axis=-1
    )
    mask = color_distance <= 24
    score_cutoff = max(
        int(round(35 * frame.shape[0] / REFERENCE_FRAME_HEIGHT)), 1
    )
    mask[:score_cutoff] = False
    ys, xs = np.where(mask)
    minimum_pixels = max(
        int(
            round(
                20
                * frame.shape[0]
                * frame.shape[1]
                / (REFERENCE_FRAME_HEIGHT * REFERENCE_FRAME_WIDTH)
            )
        ),
        4,
    )
    if len(xs) < minimum_pixels:
        return None
    return xs, ys, frame.shape[:2]


def tank_center(observation: np.ndarray, agent: str) -> tuple[float, float] | None:
    """Estimate a tank center while ignoring same-colored score digits and bullets."""
    pixels = _tank_pixels(observation, agent)
    if pixels is None:
        return None
    xs, ys, frame_shape = pixels
    frame_height, frame_width = frame_shape
    x = float(np.median(xs)) * REFERENCE_FRAME_WIDTH / frame_width
    y = float(np.median(ys)) * REFERENCE_FRAME_HEIGHT / frame_height
    return x, y


def tank_heading(observation: np.ndarray, agent: str) -> tuple[float, float] | None:
    """Estimate the continuous barrel direction from the tank sprite geometry."""
    pixels = _tank_pixels(observation, agent)
    if pixels is None:
        return None
    xs, ys, frame_shape = pixels
    frame_height, frame_width = frame_shape
    median_x = float(np.median(xs))
    median_y = float(np.median(ys))

    # Bullets share the tank color. Keep only the connected-looking sprite
    # neighbourhood around the robust median before inspecting its edges.
    x_radius = max(12.0 * frame_width / REFERENCE_FRAME_WIDTH, 2.0)
    y_radius = max(18.0 * frame_height / REFERENCE_FRAME_HEIGHT, 2.0)
    nearby = (np.abs(xs - median_x) <= x_radius) & (
        np.abs(ys - median_y) <= y_radius
    )
    local_xs = xs[nearby]
    local_ys = ys[nearby]
    if len(local_xs) < 4:
        return None

    points = np.column_stack(
        (
            local_xs * REFERENCE_FRAME_WIDTH / frame_width,
            local_ys * REFERENCE_FRAME_HEIGHT / frame_height,
        )
    ).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Tracks make the body longest perpendicular to the barrel, so the minor
    # principal axis is the unsigned barrel axis even at diagonal rotations.
    axis = eigenvectors[:, int(np.argmin(eigenvalues))]
    projections = centered @ axis
    positive_extent = float(projections.max())
    negative_extent = float(-projections.min())
    positive_cap = int(
        np.count_nonzero(projections >= 0.55 * max(positive_extent, 1e-6))
    )
    negative_cap = int(
        np.count_nonzero(projections <= -0.55 * max(negative_extent, 1e-6))
    )
    # The narrow barrel cap contains fewer pixels than the rear body cap. Use
    # projection length only to break a perfectly tied cap count.
    if positive_cap > negative_cap or (
        positive_cap == negative_cap and positive_extent < negative_extent
    ):
        axis = -axis
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None
    return float(axis[0] / norm), float(axis[1] / norm)


def _manhattan(first: tuple[float, float], second: tuple[float, float]) -> float:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def navigation_distance(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """Approximate shortest route around the two vertical maze walls."""
    if first[0] > second[0]:
        first, second = second, first
    first_region = "left" if first[0] < 28 else "right" if first[0] > 132 else "center"
    second_region = "left" if second[0] < 28 else "right" if second[0] > 132 else "center"
    if first_region == second_region or (first_region == "center" and second_region == "center"):
        return _manhattan(first, second)
    if first_region == "left" and second_region == "center":
        return min(_manhattan(first, gate) + _manhattan(gate, second) for gate in LEFT_GATES)
    if first_region == "center" and second_region == "right":
        return min(_manhattan(first, gate) + _manhattan(gate, second) for gate in RIGHT_GATES)
    if first_region == "left" and second_region == "right":
        return min(
            _manhattan(first, left_gate)
            + _manhattan(left_gate, right_gate)
            + _manhattan(right_gate, second)
            for left_gate, right_gate in zip(LEFT_GATES, RIGHT_GATES, strict=True)
        )
    return _manhattan(first, second)


class FrameHistory:
    """Maintain four recent frames without changing environment timing."""

    def __init__(self):
        self.frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)

    def reset(self, observation: np.ndarray) -> np.ndarray:
        frame = preprocess_frame(observation)
        self.frames.clear()
        self.frames.extend(frame.copy() for _ in range(FRAME_STACK))
        return self.value()

    def append(self, observation: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess_frame(observation))
        return self.value()

    def value(self) -> np.ndarray:
        return np.concatenate(tuple(self.frames), axis=-1)


OpponentPolicy = Callable[[np.ndarray, str], int]


class SingleAgentCombatEnv(gym.Env):
    """Expose one tank to Gymnasium while an injected policy controls the other."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        opponent_policy: OpponentPolicy | None = None,
        fixed_role: str | None = None,
        learner_actions: tuple[int, ...] = ACTION_SETS["all"],
        reward_shaping: bool = False,
        shaping_scale: float = 0.01,
        exploration_pretraining: bool = False,
        tactical_pretraining: bool = False,
        seed: int = 0,
        render_mode=None,
    ):
        super().__init__()
        if fixed_role not in (None, *AGENTS):
            raise ValueError(f"Unknown role: {fixed_role}")
        self.env = create_environment(render_mode=render_mode)
        self.opponent_policy = opponent_policy
        self.fixed_role = fixed_role
        if not learner_actions or any(not 0 <= action < NUM_ACTIONS for action in learner_actions):
            raise ValueError("learner_actions must contain valid Combat Tank actions")
        self.learner_actions = tuple(learner_actions)
        self.reward_shaping = reward_shaping
        self.shaping_scale = shaping_scale
        self.exploration_pretraining = exploration_pretraining
        self.tactical_pretraining = tactical_pretraining
        self.initial_seed = seed
        self.episode_index = 0
        self.learner = AGENTS[0]
        self.opponent = AGENTS[1]
        self.histories = {agent: FrameHistory() for agent in AGENTS}
        self.raw_observations: dict[str, np.ndarray] = {}
        self.latest_observations: dict[str, np.ndarray] = {}
        self.episode_return = 0.0
        self.episode_steps = 0
        self.previous_navigation_distance: float | None = None
        self.previous_learner_center: tuple[float, float] | None = None
        self.previous_position_cell: tuple[int, int] | None = None
        self.position_visits: dict[tuple[int, int], int] = {}
        self.lifetime_position_visits: Counter[tuple[str, int, int]] = Counter()
        self.steps_since_new_position = 0
        self.idle_steps = 0
        self.previous_game_action: int | None = None
        self.action_repeat = 0
        self.steps_since_hit = 0
        self.previous_opponent_center: tuple[float, float] | None = None
        self.escape_origin: tuple[float, float] | None = None
        self.escape_previous_distance = 0.0
        self.escape_steps_remaining = 0
        self.opponent_hit_origin: tuple[float, float] | None = None
        self.reacquire_steps_remaining = 0
        self.recent_actions: deque[int] = deque(maxlen=120)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(FRAME_SIZE, FRAME_SIZE, FRAME_STACK * FRAME_CHANNELS),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.learner_actions))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else self.initial_seed + self.episode_index
        observations, infos = self.env.reset(seed=actual_seed)
        if self.fixed_role is None:
            self.learner = AGENTS[int(self.np_random.integers(0, 2))]
        else:
            self.learner = self.fixed_role
        self.opponent = AGENTS[1] if self.learner == AGENTS[0] else AGENTS[0]
        if self.opponent_policy is not None and hasattr(self.opponent_policy, "reset"):
            self.opponent_policy.reset(self.opponent)
        self.raw_observations = {
            agent: observation.copy() for agent, observation in observations.items()
        }
        self.latest_observations = {
            agent: self.histories[agent].reset(observations[agent]) for agent in AGENTS
        }
        self.episode_return = 0.0
        self.episode_steps = 0
        centers = [tank_center(observations[agent], agent) for agent in AGENTS]
        self.previous_navigation_distance = (
            navigation_distance(centers[0], centers[1]) if all(centers) else None
        )
        learner_index = AGENTS.index(self.learner)
        self.previous_learner_center = centers[learner_index]
        opponent_index = AGENTS.index(self.opponent)
        self.previous_opponent_center = centers[opponent_index]
        self.position_visits = {}
        self.steps_since_new_position = 0
        self.previous_position_cell = None
        if self.previous_learner_center is not None:
            self.previous_position_cell = self._position_cell(
                self.previous_learner_center
            )
            self.position_visits[self.previous_position_cell] = 1
        self.idle_steps = 0
        self.previous_game_action = None
        self.action_repeat = 0
        self.steps_since_hit = 0
        self.escape_origin = None
        self.escape_previous_distance = 0.0
        self.escape_steps_remaining = 0
        self.opponent_hit_origin = None
        self.reacquire_steps_remaining = 0
        self.recent_actions.clear()
        self.episode_index += 1
        info = dict(infos.get(self.learner, {}))
        info["role"] = self.learner
        return self.latest_observations[self.learner], info

    def step(self, action):
        policy_action = int(action)
        if not self.action_space.contains(policy_action):
            raise ValueError(f"Learner returned invalid policy action {policy_action}")
        return self.step_game_action(self.learner_actions[policy_action])

    @staticmethod
    def _position_cell(center: tuple[float, float]) -> tuple[int, int]:
        """Quantize position for a small, diminishing exploration bonus."""
        return int(center[0] // 16), int(center[1] // 16)

    def step_game_action(self, game_action: int):
        """Advance one step with a raw official action, used by scripted curricula."""
        opponent_observation = self.latest_observations[self.opponent]
        if self.opponent_policy is None:
            opponent_action = int(self.env.action_space(self.opponent).sample())
        else:
            opponent_action = int(self.opponent_policy(opponent_observation, self.opponent))
        if not 0 <= opponent_action < NUM_ACTIONS:
            raise ValueError(f"Opponent returned invalid action {opponent_action}")

        game_action = int(game_action)
        if not 0 <= game_action < NUM_ACTIONS:
            raise ValueError(f"Learner returned invalid game action {game_action}")
        if game_action == self.previous_game_action:
            self.action_repeat += 1
        else:
            self.previous_game_action = game_action
            self.action_repeat = 1
        actions = {self.learner: game_action, self.opponent: opponent_action}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        if self.opponent_policy is not None and hasattr(
            self.opponent_policy, "observe_reward"
        ):
            self.opponent_policy.observe_reward(
                float(rewards.get(self.opponent, 0.0)), self.opponent
            )
        official_reward = float(rewards.get(self.learner, 0.0))
        self.recent_actions.append(game_action)
        if official_reward > 0:
            self.steps_since_hit = 0
        else:
            self.steps_since_hit += 1
        # During navigation pretraining, combat is deliberately secondary. The
        # official score remains a small signal, but cannot dominate coverage.
        reward = (
            0.05 * official_reward
            if self.exploration_pretraining
            else official_reward
        )
        terminated = bool(terminations.get(self.learner, False))
        truncated = bool(truncations.get(self.learner, False))
        self.episode_return += reward
        self.episode_steps += 1

        for agent, observation in observations.items():
            self.raw_observations[agent] = observation.copy()
            self.latest_observations[agent] = self.histories[agent].append(observation)
        if self.reward_shaping and all(agent in observations for agent in AGENTS):
            centers = [tank_center(observations[agent], agent) for agent in AGENTS]
            if all(centers):
                current_distance = navigation_distance(centers[0], centers[1])
                if (
                    not self.exploration_pretraining
                    and self.previous_navigation_distance is not None
                ):
                    reward += self.shaping_scale * (
                        self.previous_navigation_distance - current_distance
                    )
                self.previous_navigation_distance = current_distance
                learner_center = centers[AGENTS.index(self.learner)]
                opponent_center = centers[AGENTS.index(self.opponent)]
                previous_learner_center = self.previous_learner_center
                previous_opponent_center = self.previous_opponent_center
                if self.previous_learner_center is not None:
                    movement = _manhattan(
                        self.previous_learner_center, learner_center
                    )
                    if movement < 0.75:
                        self.idle_steps += 1
                        idle_limit = 8 if self.exploration_pretraining else 24
                        if self.idle_steps > idle_limit:
                            idle_severity = min(
                                (self.idle_steps - idle_limit) / 120.0, 1.0
                            )
                            reward -= self.shaping_scale * (
                                (0.25 if self.exploration_pretraining else 0.05)
                                + 0.20 * idle_severity
                            )
                    else:
                        self.idle_steps = 0
                        movement_weight = (
                            0.005 if self.exploration_pretraining else 0.03
                        )
                        reward += (
                            self.shaping_scale
                            * movement_weight
                            * min(movement, 4.0)
                        )
                    position_cell = self._position_cell(learner_center)
                    if position_cell != self.previous_position_cell:
                        episode_visits = self.position_visits.get(position_cell, 0)
                        self.position_visits[position_cell] = episode_visits + 1
                        if self.exploration_pretraining and episode_visits == 0:
                            lifetime_key = (
                                self.learner,
                                position_cell[0],
                                position_cell[1],
                            )
                            self.lifetime_position_visits[lifetime_key] += 1
                            lifetime_visits = self.lifetime_position_visits[
                                lifetime_key
                            ]
                            reward += self.shaping_scale / np.sqrt(lifetime_visits)
                            self.steps_since_new_position = 0
                        elif not self.exploration_pretraining:
                            reward += self.shaping_scale * 0.10 / np.sqrt(
                                episode_visits + 1
                            )
                        else:
                            self.steps_since_new_position += 1
                        self.previous_position_cell = position_cell
                    elif self.exploration_pretraining:
                        self.steps_since_new_position += 1
                    if (
                        self.exploration_pretraining
                        and self.steps_since_new_position > 180
                    ):
                        stagnation = min(
                            (self.steps_since_new_position - 180) / 360.0,
                            1.0,
                        )
                        reward -= self.shaping_scale * (0.05 + 0.15 * stagnation)
                self.previous_learner_center = learner_center
                self.previous_opponent_center = opponent_center

                if self.tactical_pretraining:
                    if official_reward < 0 and previous_learner_center is not None:
                        # Remember the exposed firing line before knockback moves us.
                        self.escape_origin = previous_learner_center
                        self.escape_previous_distance = _manhattan(
                            learner_center, self.escape_origin
                        )
                        self.escape_steps_remaining = 60
                    elif self.escape_steps_remaining > 0 and self.escape_origin is not None:
                        escape_distance = _manhattan(learner_center, self.escape_origin)
                        escape_progress = escape_distance - self.escape_previous_distance
                        reward += self.shaping_scale * 0.08 * np.clip(
                            escape_progress, -4.0, 4.0
                        )
                        if self._position_cell(learner_center) == self._position_cell(
                            self.escape_origin
                        ):
                            reward -= self.shaping_scale * 0.15
                        self.escape_previous_distance = escape_distance
                        self.escape_steps_remaining -= 1

                    if official_reward > 0:
                        # One short confirmation window is allowed; afterwards the
                        # policy must use new observations to reacquire the target.
                        self.opponent_hit_origin = (
                            previous_opponent_center or opponent_center
                        )
                        self.reacquire_steps_remaining = 12
                    elif self.reacquire_steps_remaining > 0:
                        self.reacquire_steps_remaining -= 1
                if (
                    not self.exploration_pretraining
                    and current_distance < 45
                    and game_action in ACTION_SETS["fire"]
                ):
                    reward += 0.0002
            repeat_limit = 32 if self.exploration_pretraining else 96
            if self.action_repeat > repeat_limit:
                reward -= self.shaping_scale * 0.02
            if self.steps_since_hit > 120 and len(self.recent_actions) >= 60:
                dominant_count = max(Counter(self.recent_actions).values())
                dominance = dominant_count / len(self.recent_actions)
                dominance_limit = 0.45 if self.exploration_pretraining else 0.60
                if dominance > dominance_limit:
                    reward -= self.shaping_scale * 0.10 * (
                        (dominance - dominance_limit) / (1.0 - dominance_limit)
                    )
        learner_observation = self.latest_observations.get(
            self.learner, np.zeros(self.observation_space.shape, dtype=np.uint8)
        )
        info = dict(infos.get(self.learner, {}))
        info["role"] = self.learner
        info["game_action"] = game_action
        info["official_reward"] = official_reward
        info["idle_steps"] = self.idle_steps
        info["steps_since_hit"] = self.steps_since_hit
        info["exploration_pretraining"] = self.exploration_pretraining
        info["escape_steps_remaining"] = self.escape_steps_remaining
        info["reacquire_steps_remaining"] = self.reacquire_steps_remaining
        info["episode_unique_cells"] = len(self.position_visits)
        info["steps_since_new_position"] = self.steps_since_new_position
        if self.previous_learner_center is not None:
            info["position_cell"] = self._position_cell(
                self.previous_learner_center
            )
        if terminated or truncated:
            info["episode_return"] = self.episode_return
            info["episode_steps"] = self.episode_steps
        return learner_observation, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
