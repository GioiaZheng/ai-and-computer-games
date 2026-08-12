"""Central Q-learning for a small cooperative MARL gridworld.

The environment is intentionally self-contained so the assignment can be
reproduced without PettingZoo or other simulator dependencies.

Two agents move on a 5x5 grid. The shared task is complete only when both
agents stand on their assigned targets at the same time. A single centralized
"super brain" observes the full joint state and chooses a joint action.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


Position = Tuple[int, int]
State = Tuple[int, int, int, int]
JointAction = Tuple[int, int]


ACTIONS: Dict[int, Position] = {
    0: (0, 0),    # stay
    1: (-1, 0),   # up
    2: (1, 0),    # down
    3: (0, -1),   # left
    4: (0, 1),    # right
}

ACTION_NAMES = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


@dataclass
class StepResult:
    state: State
    reward: float
    terminated: bool
    info: Dict[str, float]


class CooperativeGridworld:
    """Two-agent cooperative navigation game.

    Agent 0 must reach target_a and agent 1 must reach target_b. The episode
    succeeds only when both conditions hold simultaneously. This makes the
    return a shared cooperative objective.
    """

    def __init__(
        self,
        grid_size: int = 5,
        max_steps: int = 40,
        target_a: Position = (0, 4),
        target_b: Position = (4, 0),
        seed: int = 0,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.target_a = target_a
        self.target_b = target_b
        self.rng = random.Random(seed)
        self.agent_a: Position = (0, 0)
        self.agent_b: Position = (4, 4)
        self.steps = 0

    @property
    def joint_actions(self) -> List[JointAction]:
        return [(a0, a1) for a0 in ACTIONS for a1 in ACTIONS]

    def reset(self) -> State:
        cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in {self.target_a, self.target_b}
        ]
        self.agent_a = self.rng.choice(cells)
        remaining = [cell for cell in cells if cell != self.agent_a]
        self.agent_b = self.rng.choice(remaining)
        self.steps = 0
        return self.state

    @property
    def state(self) -> State:
        return (*self.agent_a, *self.agent_b)

    def _move(self, position: Position, action: int) -> Position:
        dr, dc = ACTIONS[action]
        row = min(max(position[0] + dr, 0), self.grid_size - 1)
        col = min(max(position[1] + dc, 0), self.grid_size - 1)
        return (row, col)

    def _distance_to_targets(self, pos_a: Position, pos_b: Position) -> int:
        da = abs(pos_a[0] - self.target_a[0]) + abs(pos_a[1] - self.target_a[1])
        db = abs(pos_b[0] - self.target_b[0]) + abs(pos_b[1] - self.target_b[1])
        return da + db

    def step(self, joint_action: JointAction) -> StepResult:
        old_a, old_b = self.agent_a, self.agent_b
        old_distance = self._distance_to_targets(old_a, old_b)

        next_a = self._move(old_a, joint_action[0])
        next_b = self._move(old_b, joint_action[1])

        collision = next_a == next_b or (next_a == old_b and next_b == old_a)
        if collision:
            next_a, next_b = old_a, old_b

        self.agent_a, self.agent_b = next_a, next_b
        self.steps += 1

        new_distance = self._distance_to_targets(next_a, next_b)
        distance_improvement = old_distance - new_distance

        reward = -0.02
        reward += 0.05 * distance_improvement
        if collision:
            reward -= 0.10

        success = next_a == self.target_a and next_b == self.target_b
        if success:
            reward += 10.0

        timeout = self.steps >= self.max_steps
        terminated = success or timeout
        return StepResult(
            state=self.state,
            reward=reward,
            terminated=terminated,
            info={
                "success": float(success),
                "collision": float(collision),
                "distance": float(new_distance),
            },
        )


class CentralQLearner:
    """Tabular Q-learning over joint states and joint actions."""

    def __init__(
        self,
        joint_actions: Iterable[JointAction],
        alpha: float = 0.20,
        gamma: float = 0.95,
    ) -> None:
        self.joint_actions = list(joint_actions)
        self.alpha = alpha
        self.gamma = gamma
        self.q: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.joint_actions), dtype=np.float64)
        )

    def act(self, state: State, epsilon: float, rng: random.Random) -> JointAction:
        if rng.random() < epsilon:
            return rng.choice(self.joint_actions)
        q_values = self.q[state]
        best_value = np.max(q_values)
        best_indices = np.flatnonzero(np.isclose(q_values, best_value))
        return self.joint_actions[int(rng.choice(best_indices.tolist()))]

    def update(
        self,
        state: State,
        action: JointAction,
        reward: float,
        next_state: State,
        terminated: bool,
    ) -> None:
        action_index = self.joint_actions.index(action)
        current = self.q[state][action_index]
        bootstrap = 0.0 if terminated else float(np.max(self.q[next_state]))
        target = reward + self.gamma * bootstrap
        self.q[state][action_index] = current + self.alpha * (target - current)


def epsilon_by_episode(
    episode: int,
    episodes: int,
    start: float = 1.0,
    end: float = 0.05,
    decay_fraction: float = 0.80,
) -> float:
    decay_episodes = max(1, int(episodes * decay_fraction))
    progress = min(1.0, episode / decay_episodes)
    return end + (start - end) * math.exp(-5.0 * progress)


def run_episode(
    env: CooperativeGridworld,
    learner: CentralQLearner,
    epsilon: float,
    rng: random.Random,
    train: bool,
) -> Dict[str, float]:
    state = env.reset()
    total_reward = 0.0
    collisions = 0
    success = 0

    for _ in range(env.max_steps):
        action = learner.act(state, epsilon=epsilon, rng=rng)
        result = env.step(action)
        if train:
            learner.update(state, action, result.reward, result.state, result.terminated)
        total_reward += result.reward
        collisions += int(result.info["collision"])
        success = int(result.info["success"])
        state = result.state
        if result.terminated:
            break

    return {
        "return": total_reward,
        "success": float(success),
        "steps": float(env.steps),
        "collisions": float(collisions),
    }


def train_one_seed(seed: int, episodes: int) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rng = random.Random(seed)
    env = CooperativeGridworld(seed=seed)
    learner = CentralQLearner(env.joint_actions)

    training_rows: List[Dict[str, float]] = []
    for episode in range(episodes):
        epsilon = epsilon_by_episode(episode, episodes)
        metrics = run_episode(env, learner, epsilon=epsilon, rng=rng, train=True)
        metrics.update({"seed": float(seed), "episode": float(episode), "epsilon": epsilon})
        training_rows.append(metrics)

    eval_env = CooperativeGridworld(seed=10_000 + seed)
    eval_rows = [
        run_episode(eval_env, learner, epsilon=0.0, rng=rng, train=False)
        for _ in range(300)
    ]
    summary = {
        "seed": float(seed),
        "episodes": float(episodes),
        "eval_success_rate": float(np.mean([row["success"] for row in eval_rows])),
        "eval_avg_return": float(np.mean([row["return"] for row in eval_rows])),
        "eval_avg_steps": float(np.mean([row["steps"] for row in eval_rows])),
        "eval_avg_collisions": float(np.mean([row["collisions"] for row in eval_rows])),
        "q_states_visited": float(len(learner.q)),
    }
    return training_rows, summary


def moving_average(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curve(training_rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = sorted({int(row["episode"]) for row in training_rows})
    grouped: Dict[int, List[float]] = {episode: [] for episode in episodes}
    success_grouped: Dict[int, List[float]] = {episode: [] for episode in episodes}
    for row in training_rows:
        grouped[int(row["episode"])].append(row["return"])
        success_grouped[int(row["episode"])].append(row["success"])

    mean_returns = [float(np.mean(grouped[episode])) for episode in episodes]
    mean_success = [float(np.mean(success_grouped[episode])) for episode in episodes]
    window = 100
    ma_returns = moving_average(mean_returns, window)
    ma_success = moving_average(mean_success, window)

    width, height = 1200, 850
    margin_left, margin_right = 100, 40
    title_h = 70
    panel_gap = 55
    panel_h = 320
    panel_w = width - margin_left - margin_right
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 30)
        font_label = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((margin_left, 24), "Central Q-learning training curve", fill=(20, 31, 46), font=font_title)

    def draw_panel(
        top: int,
        values: np.ndarray,
        ylabel: str,
        color: Tuple[int, int, int],
        fixed_range: Tuple[float, float] | None = None,
    ) -> None:
        left = margin_left
        right = margin_left + panel_w
        bottom = top + panel_h
        draw.rectangle((left, top, right, bottom), outline=(170, 180, 190), width=2)
        for i in range(1, 5):
            y = top + int(panel_h * i / 5)
            draw.line((left, y, right, y), fill=(225, 230, 235), width=1)
        if fixed_range:
            y_min, y_max = fixed_range
        else:
            y_min = float(np.min(values))
            y_max = float(np.max(values))
            if abs(y_max - y_min) < 1e-9:
                y_max = y_min + 1.0
        draw.text((18, top + panel_h // 2 - 10), ylabel, fill=(20, 31, 46), font=font_label)
        draw.text((left - 78, top - 8), f"{y_max:.2f}", fill=(75, 85, 99), font=font_small)
        draw.text((left - 78, bottom - 10), f"{y_min:.2f}", fill=(75, 85, 99), font=font_small)
        if len(values) < 2:
            return
        points = []
        for i, value in enumerate(values):
            x = left + int(panel_w * i / (len(values) - 1))
            y_norm = (float(value) - y_min) / (y_max - y_min)
            y = bottom - int(panel_h * y_norm)
            points.append((x, y))
        draw.line(points, fill=color, width=4, joint="curve")

    draw_panel(title_h, ma_returns, "Return", (40, 104, 166))
    draw_panel(title_h + panel_h + panel_gap, ma_success, "Success", (23, 138, 90), fixed_range=(0.0, 1.0))

    draw.text((margin_left + panel_w // 2 - 40, height - 48), "Episode", fill=(20, 31, 46), font=font_label)
    draw.text((margin_left, height - 26), f"100-episode moving average over {len(set(row['seed'] for row in training_rows))} seeds", fill=(75, 85, 99), font=font_small)

    img.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("../results"))
    args = parser.parse_args()

    all_training_rows: List[Dict[str, float]] = []
    summaries: List[Dict[str, float]] = []
    for seed in range(args.seeds):
        training_rows, summary = train_one_seed(seed=seed, episodes=args.episodes)
        all_training_rows.extend(training_rows)
        summaries.append(summary)
        print(
            f"seed={seed} success={summary['eval_success_rate']:.3f} "
            f"return={summary['eval_avg_return']:.2f} steps={summary['eval_avg_steps']:.1f}"
        )

    output_dir = args.output_dir
    write_csv(output_dir / "training_log.csv", all_training_rows)
    write_csv(output_dir / "evaluation_summary.csv", summaries)
    plot_training_curve(all_training_rows, output_dir / "training_curve.png")

    aggregate = {
        "seed": -1.0,
        "episodes": float(args.episodes),
        "eval_success_rate": float(np.mean([row["eval_success_rate"] for row in summaries])),
        "eval_avg_return": float(np.mean([row["eval_avg_return"] for row in summaries])),
        "eval_avg_steps": float(np.mean([row["eval_avg_steps"] for row in summaries])),
        "eval_avg_collisions": float(np.mean([row["eval_avg_collisions"] for row in summaries])),
        "q_states_visited": float(np.mean([row["q_states_visited"] for row in summaries])),
    }
    print("aggregate", aggregate)


if __name__ == "__main__":
    main()
