"""Train PPO on the exact Combat Tank pipeline used by the tournament."""

from __future__ import annotations

import argparse
from pathlib import Path

from .official_environment import AGENTS, OfficialSingleAgentCombatEnv
from .official_opponents import OfficialMixedOpponent, OfficialPPOOpponent


def build_vector_environment(args):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    factories = []
    for rank in range(args.n_envs):
        env_seed = args.seed + rank * 10_000
        role = args.fixed_role or AGENTS[rank % len(AGENTS)]

        def make_env(seed=env_seed, fixed_role=role):
            if args.opponent == "model":
                opponent_policy = OfficialPPOOpponent(
                    args.opponent_model,
                    device=args.opponent_device,
                    deterministic=args.opponent_deterministic,
                )
            elif args.opponent == "mixed":
                opponent_policy = OfficialMixedOpponent(
                    args.opponent_model,
                    model_probability=args.opponent_model_probability,
                    device=args.opponent_device,
                    deterministic=args.opponent_deterministic,
                    seed=seed + 97,
                )
            else:
                opponent_policy = None
            return Monitor(
                OfficialSingleAgentCombatEnv(
                    opponent_policy=opponent_policy,
                    fixed_role=fixed_role,
                    seed=seed,
                    exploration_bonus_scale=args.exploration_bonus_scale,
                    idle_penalty_scale=args.idle_penalty_scale,
                )
            )

        factories.append(make_env)
    if args.vec_env == "subproc":
        return SubprocVecEnv(factories, start_method=args.start_method)
    return DummyVecEnv(factories)


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate

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

    if args.load_model is None:
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
            max_grad_norm=0.5,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        reset_num_timesteps = True
    else:
        model = PPO.load(args.load_model, env=env, device=args.device)
        loaded_shape = tuple(model.observation_space.shape)
        compatible_shapes = {(84, 84, 6), (6, 84, 84)}
        if loaded_shape not in compatible_shapes:
            raise ValueError(
                f"Checkpoint observation shape {loaded_shape} does not match "
                "the official six-channel pipeline"
            )
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        update_learning_rate(model.policy.optimizer, args.learning_rate)
        model.n_epochs = args.n_epochs
        model.batch_size = args.batch_size
        model.gamma = args.gamma
        model.gae_lambda = args.gae_lambda
        model.ent_coef = args.entropy_coefficient
        model.vf_coef = args.value_coefficient
        model.clip_range = get_schedule_fn(args.clip_range)
        model.verbose = 1
        reset_num_timesteps = False

    print(
        "Official tournament training / 官方比赛管线训练\n"
        f"observation={env.observation_space} action={env.action_space} "
        f"device={model.device} roles=balanced"
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(output)
    env.close()
    print(f"Saved official-pipeline PPO model: {output}.zip")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--start-method", choices=["forkserver", "spawn"], default="forkserver")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--entropy-coefficient", type=float, default=0.02)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--exploration-bonus-scale", type=float, default=0.0)
    parser.add_argument("--idle-penalty-scale", type=float, default=0.0)
    parser.add_argument("--fixed-role", choices=AGENTS)
    parser.add_argument(
        "--opponent", choices=["random", "model", "mixed"], default="random"
    )
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument("--opponent-device", default="cpu")
    parser.add_argument("--opponent-deterministic", action="store_true")
    parser.add_argument("--opponent-model-probability", type=float, default=0.70)
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=Path("checkpoints/official-pipeline-random"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/official-pipeline-random/final_model"),
    )
    parser.add_argument("--run-name", default="official_pipeline_random")
    parser.add_argument("--load-model", type=Path)
    parser.add_argument("--seed", type=int, default=82_026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.n_envs < 1:
        parser.error("--n-envs must be at least 1")
    if args.batch_size > args.n_envs * args.n_steps:
        parser.error("--batch-size cannot exceed n_envs * n_steps")
    if args.opponent in {"model", "mixed"} and args.opponent_model is None:
        parser.error("--opponent-model is required for model and mixed opponents")
    return args


if __name__ == "__main__":
    train(parse_args())
