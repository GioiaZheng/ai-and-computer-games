"""PPO training against a curriculum of frozen Atari Boxing opponents."""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo.atari import boxing_v2
import supersuit as ss


FRAME_SIZE = 84
OBSERVATION_CHANNELS = 6
NUM_ACTIONS = 18


def create_environment(render_mode=None):
    """Build exactly the observation pipeline used by the tournament."""
    env = boxing_v2.parallel_env(render_mode=render_mode)
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    return env


def load_external_agent(directory: Path, env):
    """Load one teacher-compatible Agent class from a submission folder."""
    directory = Path(directory).resolve()
    candidates = (directory / "agent_template.py", directory / "agent.py")
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            f"No agent_template.py or agent.py found in {directory}"
        )

    module_name = f"boxing_opponent_{abs(hash(str(source)))}_{random.randrange(1 << 30)}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import opponent from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Agent"):
        raise AttributeError(f"{source} does not define Agent")
    return module.Agent(env)


def validated_action(agent, observation, n_actions=NUM_ACTIONS):
    action = int(agent.get_action(np.asarray(observation, dtype=np.uint8)))
    if not 0 <= action < n_actions:
        raise ValueError(f"External agent returned invalid action {action}")
    return action


class BoxingOpponentEnv(gym.Env):
    """Expose one learning boxer while a frozen policy controls the other."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        opponent_directories: list[Path],
        external_probability: float = 0.7,
        curriculum_episodes: int = 150,
        seed: int = 0,
        render_mode=None,
    ):
        super().__init__()
        if not 0.0 <= external_probability <= 1.0:
            raise ValueError("external_probability must be between 0 and 1")
        self.env = create_environment(render_mode=render_mode)
        self.opponent_directories = [Path(path) for path in opponent_directories]
        self.external_probability = external_probability
        self.curriculum_episodes = max(0, curriculum_episodes)
        self.initial_seed = seed
        self.episode_index = 0
        self.learner_agent = "first_0"
        self.opponent_agent = "second_0"
        self.opponent = None
        self.opponent_kind = "random"
        self.episode_return = 0.0
        self.episode_steps = 0

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(FRAME_SIZE, FRAME_SIZE, OBSERVATION_CHANNELS),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def _current_external_probability(self):
        if not self.opponent_directories:
            return 0.0
        if self.curriculum_episodes == 0:
            return self.external_probability
        progress = min(self.episode_index / self.curriculum_episodes, 1.0)
        # Start with easier random games, then steadily introduce frozen agents.
        return self.external_probability * (0.2 + 0.8 * progress)

    def step(self, action):
        if self.opponent is None:
            opponent_action = int(self.env.action_space(self.opponent_agent).sample())
        else:
            opponent_observation = self._last_observations[self.opponent_agent]
            opponent_action = validated_action(self.opponent, opponent_observation)

        actions = {
            self.learner_agent: int(action),
            self.opponent_agent: opponent_action,
        }
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        reward = float(rewards.get(self.learner_agent, 0.0))
        terminated = bool(terminations.get(self.learner_agent, False))
        truncated = bool(truncations.get(self.learner_agent, False))
        self.episode_return += reward
        self.episode_steps += 1

        if self.learner_agent in observations:
            observation = np.asarray(observations[self.learner_agent], dtype=np.uint8)
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._last_observations = observations

        info = dict(infos.get(self.learner_agent, {}))
        info.update({"role": self.learner_agent, "opponent": self.opponent_kind})
        if terminated or truncated:
            info["official_return"] = self.episode_return
            info["episode_steps"] = self.episode_steps
        return observation, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def reset(self, *, seed=None, options=None):
        observation, info = self._reset_impl(seed=seed)
        return observation, info

    def _reset_impl(self, seed=None):
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else self.initial_seed + self.episode_index
        observations, infos = self.env.reset(seed=actual_seed)
        self._last_observations = observations

        if int(self.np_random.integers(0, 2)) == 0:
            self.learner_agent, self.opponent_agent = "first_0", "second_0"
        else:
            self.learner_agent, self.opponent_agent = "second_0", "first_0"

        use_external = bool(
            self.opponent_directories
            and self.np_random.random() < self._current_external_probability()
        )
        if use_external:
            index = int(self.np_random.integers(0, len(self.opponent_directories)))
            opponent_directory = self.opponent_directories[index]
            self.opponent = load_external_agent(opponent_directory, self.env)
            self.opponent_kind = opponent_directory.name
        else:
            self.opponent = None
            self.opponent_kind = "random"

        self.episode_return = 0.0
        self.episode_steps = 0
        self.episode_index += 1
        info = dict(infos.get(self.learner_agent, {}))
        info.update({"role": self.learner_agent, "opponent": self.opponent_kind})
        return np.asarray(observations[self.learner_agent], dtype=np.uint8), info


def build_vector_environment(args):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    factories = []
    for rank in range(args.n_envs):
        env_seed = args.seed + rank * 10_000

        def make_env(seed=env_seed):
            return Monitor(
                BoxingOpponentEnv(
                    opponent_directories=args.external_opponent,
                    external_probability=args.external_probability,
                    curriculum_episodes=args.curriculum_episodes,
                    seed=seed,
                )
            )

        factories.append(make_env)
    return DummyVecEnv(factories)


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    random.seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = Path(args.checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    env = build_vector_environment(args)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency // args.n_envs, 1),
        save_path=str(checkpoint_directory),
        name_prefix=args.run_name,
    )
    if args.resume:
        model = PPO.load(args.resume, env=env, device=args.device)
        model.verbose = 1
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.entropy_coefficient,
            vf_coef=args.value_coefficient,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=not bool(args.resume),
    )
    model.save(output)
    env.close()
    print(f"Saved PPO model: {output}.zip")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-opponent", action="append", type=Path, default=[])
    parser.add_argument("--external-probability", type=float, default=0.7)
    parser.add_argument("--curriculum-episodes", type=int, default=150)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints/ppo"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ppo_boxing"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name", default="ppo_boxing")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.n_envs < 1:
        parser.error("--n-envs must be at least 1")
    if args.batch_size > args.n_envs * args.n_steps:
        parser.error("--batch-size cannot exceed n_envs * n_steps")
    return args


if __name__ == "__main__":
    train(parse_args())
