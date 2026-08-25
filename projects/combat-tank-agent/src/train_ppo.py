"""Train a role-balanced PPO Combat Tank policy against selectable opponents."""

from __future__ import annotations

import argparse
from pathlib import Path

from .curriculum import ScriptedPrefixWrapper
from .environment import ACTION_SETS, AGENTS, SingleAgentCombatEnv
from .opponents import MixedOpponentPolicy, PPOOpponentPolicy
from .scripted_agent import ScriptedOpponentPolicy


def build_vector_environment(args):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = []
    for rank in range(args.n_envs):
        env_seed = args.seed + rank * 10_000

        def make_env(seed=env_seed):
            if args.opponent == "scripted":
                opponent_policy = ScriptedOpponentPolicy()
            elif args.opponent == "mixed":
                opponent_policy = MixedOpponentPolicy(
                    args.opponent_model,
                    action_set=args.opponent_action_set,
                    scripted_prefix_steps=args.opponent_prefix_steps,
                    device=args.opponent_device,
                    deterministic=not args.opponent_stochastic,
                    seed=seed + 97,
                )
            elif args.opponent == "model":
                opponent_policy = PPOOpponentPolicy(
                    args.opponent_model,
                    action_set=args.opponent_action_set,
                    scripted_prefix_steps=args.opponent_prefix_steps,
                    device=args.opponent_device,
                    deterministic=not args.opponent_stochastic,
                )
            else:
                opponent_policy = None
            env = SingleAgentCombatEnv(
                opponent_policy=opponent_policy,
                fixed_role=args.fixed_role,
                seed=seed,
                learner_actions=ACTION_SETS[args.action_set],
                reward_shaping=args.reward_shaping,
                shaping_scale=args.shaping_scale,
                exploration_pretraining=args.exploration_pretraining,
                tactical_pretraining=args.tactical_pretraining,
            )
            if args.scripted_prefix_steps:
                env = ScriptedPrefixWrapper(env, args.scripted_prefix_steps)
            return Monitor(env)

        factories.append(make_env)
    if args.vec_env == "subproc":
        return SubprocVecEnv(factories, start_method=args.start_method)
    return DummyVecEnv(factories)


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

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
    if args.load_model is not None:
        from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate

        model = PPO.load(args.load_model, env=env, device=args.device)
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        update_learning_rate(model.policy.optimizer, args.learning_rate)
        model.n_epochs = args.n_epochs
        model.batch_size = args.batch_size
        model.gamma = args.gamma
        model.ent_coef = args.entropy_coefficient
        model.clip_range = get_schedule_fn(args.clip_range)
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
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=args.entropy_coefficient,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=args.load_model is None,
    )
    model.save(output)
    env.close()
    print(f"Saved PPO model: {output}.zip")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--start-method", choices=["forkserver", "spawn"], default="forkserver")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--action-set", choices=ACTION_SETS, default="all")
    parser.add_argument("--fixed-role", choices=AGENTS)
    parser.add_argument(
        "--opponent",
        choices=["random", "scripted", "model", "mixed"],
        default="random",
    )
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument("--opponent-action-set", choices=ACTION_SETS, default="fire")
    parser.add_argument("--opponent-prefix-steps", type=int, default=2024)
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument("--opponent-stochastic", action="store_true")
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--shaping-scale", type=float, default=0.01)
    parser.add_argument("--exploration-pretraining", action="store_true")
    parser.add_argument("--tactical-pretraining", action="store_true")
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints/ppo"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ppo/final_model"))
    parser.add_argument("--run-name", default="combat_tank_ppo")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--scripted-prefix-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.n_envs < 1:
        parser.error("--n-envs must be at least 1")
    if args.batch_size > args.n_envs * args.n_steps:
        parser.error("--batch-size cannot exceed n_envs * n_steps")
    if args.scripted_prefix_steps < 0:
        parser.error("--scripted-prefix-steps must be non-negative")
    if args.opponent in {"model", "mixed"} and args.opponent_model is None:
        parser.error("--opponent-model is required for model or mixed opponents")
    return args


if __name__ == "__main__":
    train(parse_args())
