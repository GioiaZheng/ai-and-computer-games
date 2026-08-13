"""Train a reproducible Double-Dueling DQN agent for PettingZoo Atari Boxing.

The implementation stays small enough for the summer-school project while
adding the stability and evaluation mechanisms needed for a meaningful run.
实现保持课堂项目可读性，同时补充稳定训练、对手课程和可重复评估。
"""

import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import supersuit as ss
import torch
from pettingzoo.atari import boxing_v2
from torch import nn


TRAIN_AGENT = "first_0"
SECOND_AGENT = "second_0"
PLAYER_AGENTS = (TRAIN_AGENT, SECOND_AGENT)
FRAME_SIZE = 84


def create_environment(render_mode=None, frame_stack=4):
    """Reproduce the instructor's tournament observation pipeline."""
    env = boxing_v2.parallel_env(render_mode=render_mode)
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=FRAME_SIZE, y_size=FRAME_SIZE)
    env = ss.frame_stack_v1(env, frame_stack)
    env = ss.agent_indicator_v0(env, type_only=False)
    return env


def observation_to_state(observation):
    """Convert official HWC uint8 observations to PyTorch CHW replay states."""
    observation = np.asarray(observation, dtype=np.uint8)
    if observation.ndim != 3 or observation.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
        raise ValueError(f"Unexpected wrapped observation shape: {observation.shape}")
    return np.ascontiguousarray(observation.transpose(2, 0, 1))


def preprocess_frame(observation):
    """Convert one RGB frame to an 84x84 grayscale uint8 frame / 灰度缩放。"""
    gray = (
        0.299 * observation[:, :, 0]
        + 0.587 * observation[:, :, 1]
        + 0.114 * observation[:, :, 2]
    ).astype(np.uint8)
    y_idx = np.linspace(0, gray.shape[0] - 1, FRAME_SIZE).astype(np.int32)
    x_idx = np.linspace(0, gray.shape[1] - 1, FRAME_SIZE).astype(np.int32)
    return gray[y_idx][:, x_idx]


def stacked_initial_state(observation, frame_stack):
    """Repeat the first frame because no earlier motion history exists / 初始化帧栈。"""
    frame = preprocess_frame(observation)
    frames = deque([frame.copy() for _ in range(frame_stack)], maxlen=frame_stack)
    return np.stack(frames, axis=0), frames


def append_frame(frames, observation):
    """Append a frame and return the current stack / 加入新帧并返回状态。"""
    frames.append(preprocess_frame(observation))
    return np.stack(frames, axis=0)


class ReplayBuffer:
    """Fixed-size CPU replay using compact uint8 states / CPU uint8 经验回放池。"""

    def __init__(self, capacity, state_shape, seed=0):
        self.capacity = capacity
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = self.rng.integers(0, self.size, size=batch_size)
        # Replay stays on CPU; only the sampled mini-batch moves to the GPU.
        # 回放池留在 CPU，只把本次 mini-batch 搬到 GPU。
        states = torch.as_tensor(self.states[indices], device=device)
        actions = torch.as_tensor(self.actions[indices], device=device)
        rewards = torch.as_tensor(self.rewards[indices], device=device)
        next_states = torch.as_tensor(self.next_states[indices], device=device)
        dones = torch.as_tensor(self.dones[indices], device=device)
        return states, actions, rewards, next_states, dones


class DQN(nn.Module):
    """Dueling convolutional Q-network / Dueling 卷积 Q 网络。"""

    def __init__(self, frame_stack, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, frame_stack, FRAME_SIZE, FRAME_SIZE)
            feature_dim = self.features(sample).shape[1]

        self.shared = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU())
        self.value_head = nn.Linear(512, 1)
        self.advantage_head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255.0
        hidden = self.shared(self.features(x))
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        # Q = V + centered advantage makes the decomposition identifiable.
        # 减去动作优势均值，避免 V 与 A 可以任意平移的问题。
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def linear_epsilon(step, start, end, decay_steps):
    """Linearly anneal exploration / 线性衰减探索率。"""
    if decay_steps <= 0:
        return end
    progress = min(step / decay_steps, 1.0)
    return start + progress * (end - start)


def select_action(policy_net, state, epsilon, n_actions, device, rng=None):
    """Select an epsilon-greedy action / epsilon-greedy 选动作。"""
    rng = rng or random
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state[None, ...], device=device)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def optimize(policy_net, target_net, replay, optimizer, args, device):
    """Run one Double-DQN update and return diagnostics / 一次 Double DQN 更新。"""
    if len(replay) < max(args.learning_starts, args.batch_size):
        return None

    states, actions, rewards, next_states, dones = replay.sample(args.batch_size, device)
    predictions = policy_net(states).gather(1, actions[:, None]).squeeze(1)

    with torch.no_grad():
        # Online network selects; target network evaluates / 在线网络选，目标网络估值。
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        targets = rewards + args.gamma * next_values * (1.0 - dones)

    td_errors = targets - predictions
    loss = nn.functional.smooth_l1_loss(predictions, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(policy_net.parameters(), args.grad_clip)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "q_mean": float(predictions.detach().mean().item()),
        "target_mean": float(targets.mean().item()),
        "td_abs_mean": float(td_errors.abs().mean().item()),
        "grad_norm": float(grad_norm.item()),
    }


def save_checkpoint(
    path,
    policy_net,
    target_net,
    opponent_net,
    optimizer,
    steps,
    episodes,
    best_score,
    opponent_ready,
    args,
    n_actions,
):
    """Save enough state to resume training / 保存可继续训练的完整状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "algorithm": "double-dueling-dqn",
            "model_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "opponent_state_dict": opponent_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": steps,
            "episodes": episodes,
            "best_score": best_score,
            "opponent_ready": opponent_ready,
            "frame_stack": args.frame_stack,
            "input_channels": policy_net.features[0].in_channels,
            "observation_pipeline": "instructor-supersuit-v1",
            "n_actions": n_actions,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, policy_net, target_net, opponent_net, optimizer, device):
    """Restore a version-3 official-pipeline checkpoint / 恢复官方输入 checkpoint。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 3:
        raise ValueError(
            "The checkpoint does not use the instructor's six-channel pipeline."
        )
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    target_net.load_state_dict(checkpoint["target_state_dict"])
    opponent_net.load_state_dict(checkpoint["opponent_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def write_result(path, row):
    """Append one stable-schema CSV row / 追加一行实验记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def choose_episode_opponent(args, opponent_ready, rng):
    """Choose one stationary opponent for an episode / 每局固定一种对手。"""
    if args.opponent == "random" or not opponent_ready:
        return "random"
    if args.opponent == "snapshot":
        return "snapshot"
    return "random" if rng.random() < args.random_opponent_prob else "snapshot"


def choose_training_agent(train_role, rng, episode=None):
    """Choose the controlled player for one episode / 每局选择一个训练角色。"""
    if train_role == "first":
        return TRAIN_AGENT
    if train_role == "second":
        return SECOND_AGENT
    if train_role == "alternate":
        if episode is None:
            raise ValueError("episode is required when train_role='alternate'")
        return PLAYER_AGENTS[episode % len(PLAYER_AGENTS)]
    return PLAYER_AGENTS[rng.randrange(len(PLAYER_AGENTS))]


def opponent_action(kind, opponent_net, state, n_actions, args, device, rng):
    if kind == "random":
        return rng.randrange(n_actions)
    return select_action(
        opponent_net,
        state,
        args.opponent_epsilon,
        n_actions,
        device,
        rng,
    )


def evaluate(
    policy_net,
    opponent_net,
    args,
    device,
    n_actions,
    opponent_kind,
    seed_base,
    controlled_agent=TRAIN_AGENT,
):
    """Evaluate greedily on fixed seeds / 在固定种子上做贪心评估。"""
    env = create_environment(render_mode=None, frame_stack=args.frame_stack)
    returns = []
    policy_net.eval()
    opponent_net.eval()
    rng = random.Random(seed_base + 99_999)

    for episode_index in range(args.eval_episodes):
        observations, _ = env.reset(seed=seed_base + episode_index)
        states = {
            agent: observation_to_state(observations[agent]) for agent in env.agents
        }

        total_reward = 0.0
        for _ in range(args.max_steps):
            actions = {}
            for agent in env.agents:
                if agent == controlled_agent:
                    actions[agent] = select_action(
                        policy_net, states[agent], 0.0, n_actions, device, rng
                    )
                elif opponent_kind == "snapshot":
                    actions[agent] = select_action(
                        opponent_net,
                        states[agent],
                        args.eval_opponent_epsilon,
                        n_actions,
                        device,
                        rng,
                    )
                else:
                    actions[agent] = rng.randrange(n_actions)

            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += float(rewards.get(controlled_agent, 0.0))
            done = bool(
                terminations.get(controlled_agent, False)
                or truncations.get(controlled_agent, False)
                or controlled_agent not in env.agents
            )
            if done:
                break
            for agent in env.agents:
                states[agent] = observation_to_state(next_observations[agent])
        returns.append(total_reward)

    env.close()
    policy_net.train()
    values = np.asarray(returns, dtype=np.float32)
    return float(values.mean()), float(values.std()), returns


def maybe_init_wandb(args):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or use --wandb-mode disabled.") from exc
    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )


def mean_metric(metrics, key):
    values = [item[key] for item in metrics]
    return float(np.mean(values)) if values else 0.0


def train(args):
    """Train, periodically evaluate, and save latest/best checkpoints / 训练主循环。"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    env = create_environment(render_mode=None, frame_stack=args.frame_stack)
    observations, _ = env.reset(seed=args.seed)
    n_actions = env.action_space(TRAIN_AGENT).n
    initial_state = observation_to_state(observations[TRAIN_AGENT])
    input_channels = initial_state.shape[0]
    policy_net = DQN(input_channels, n_actions).to(device)
    target_net = DQN(input_channels, n_actions).to(device)
    opponent_net = DQN(input_channels, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    opponent_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    opponent_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    state_shape = initial_state.shape
    replay = ReplayBuffer(args.replay_size, state_shape, seed=args.seed)
    checkpoint_path = Path(args.checkpoint)
    best_checkpoint_path = Path(args.best_checkpoint)
    result_path = Path(args.results)
    evaluation_path = Path(args.evaluation_results)

    total_steps = 0
    start_episode = 1
    best_score = -float("inf")
    opponent_ready = False
    if args.resume:
        checkpoint = load_checkpoint(
            Path(args.resume), policy_net, target_net, opponent_net, optimizer, device
        )
        total_steps = int(checkpoint["steps"])
        start_episode = int(checkpoint["episodes"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        opponent_ready = bool(checkpoint.get("opponent_ready", False))
        # The CLI learning rate intentionally controls fine-tuning after resume.
        # 续训可显式降低学习率，避免被 checkpoint 中的旧 optimizer 值覆盖。
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.lr

    rng = random.Random(args.seed + 17)
    wandb_run = maybe_init_wandb(args)
    print("Training shared-role Double-Dueling DQN / 训练双角色共享策略")
    print(
        f"device={device} actions={n_actions} opponent={args.opponent} "
        f"train_role={args.train_role}"
    )

    for episode in range(start_episode, args.episodes + 1):
        observations, _ = env.reset(seed=args.seed + episode)
        states = {
            agent: observation_to_state(observations[agent]) for agent in env.agents
        }

        opponent_kind = choose_episode_opponent(args, opponent_ready, rng)
        learner_agent = choose_training_agent(args.train_role, rng, episode)
        episode_reward = 0.0
        shaping_penalty_total = 0.0
        previous_learner_action = None
        repeated_action_steps = 0
        max_repeated_action_steps = 0
        no_reward_steps = 0
        max_no_reward_steps = 0
        update_metrics = []
        epsilon = linear_epsilon(
            total_steps, args.eps_start, args.eps_end, args.eps_decay_steps
        )

        for episode_step in range(1, args.max_steps + 1):
            epsilon = linear_epsilon(
                total_steps, args.eps_start, args.eps_end, args.eps_decay_steps
            )
            actions = {}
            for agent in env.agents:
                if agent == learner_agent:
                    actions[agent] = select_action(
                        policy_net, states[agent], epsilon, n_actions, device, rng
                    )
                else:
                    actions[agent] = opponent_action(
                        opponent_kind,
                        opponent_net,
                        states[agent],
                        n_actions,
                        args,
                        device,
                        rng,
                    )

            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            raw_reward = float(rewards.get(learner_agent, 0.0))
            base_replay_reward = (
                float(np.sign(raw_reward)) if args.reward_clip == "sign" else raw_reward
            )
            learner_action = actions[learner_agent]
            if learner_action == previous_learner_action:
                repeated_action_steps += 1
            else:
                repeated_action_steps = 1
                previous_learner_action = learner_action
            max_repeated_action_steps = max(
                max_repeated_action_steps, repeated_action_steps
            )

            if raw_reward == 0.0:
                no_reward_steps += 1
            else:
                no_reward_steps = 0
            max_no_reward_steps = max(max_no_reward_steps, no_reward_steps)

            # Reward shaping is used only by the learner. The reported episode
            # return remains the official environment score. A tiny penalty
            # teaches the Q-network to leave deterministic standing/punching
            # loops instead of exploiting a one-point lead until time expires.
            repeat_penalty = (
                args.repeat_action_penalty
                if repeated_action_steps > args.repeat_action_threshold
                else 0.0
            )
            inactivity_penalty = (
                args.inactivity_penalty
                if no_reward_steps > args.inactivity_threshold
                else 0.0
            )
            step_shaping_penalty = repeat_penalty + inactivity_penalty
            shaping_penalty_total += step_shaping_penalty
            replay_reward = base_replay_reward - step_shaping_penalty
            # A local --max-steps cutoff also ends this replay trajectory.
            # 本地时间上限同样必须关闭 bootstrap，避免跨 reset 泄漏价值。
            time_limit = episode_step >= args.max_steps
            done = bool(
                terminations.get(learner_agent, False)
                or truncations.get(learner_agent, False)
                or learner_agent not in env.agents
                or time_limit
            )
            if learner_agent in next_observations:
                next_state = observation_to_state(next_observations[learner_agent])
            else:
                next_state = states[learner_agent].copy()

            replay.push(
                states[learner_agent],
                actions[learner_agent],
                replay_reward,
                next_state,
                done,
            )
            states[learner_agent] = next_state
            if not done:
                for agent in env.agents:
                    if agent != learner_agent:
                        states[agent] = observation_to_state(next_observations[agent])

            episode_reward += raw_reward
            total_steps += 1

            if total_steps % args.train_freq == 0:
                metrics = optimize(
                    policy_net, target_net, replay, optimizer, args, device
                )
                if metrics is not None:
                    update_metrics.append(metrics)

            if total_steps % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if total_steps >= args.opponent_learning_starts and (
                total_steps % args.opponent_update == 0
            ):
                opponent_net.load_state_dict(policy_net.state_dict())
                opponent_net.eval()
                opponent_ready = True

            if done:
                break

        row = {
            "episode": episode,
            "total_steps": total_steps,
            "episode_steps": episode_step,
            "learner_agent": learner_agent,
            "opponent": opponent_kind,
            "reward": round(episode_reward, 4),
            "shaping_penalty": round(shaping_penalty_total, 4),
            "max_action_repeat": max_repeated_action_steps,
            "max_no_reward_steps": max_no_reward_steps,
            "epsilon": round(epsilon, 6),
            "loss": round(mean_metric(update_metrics, "loss"), 6),
            "q_mean": round(mean_metric(update_metrics, "q_mean"), 6),
            "target_mean": round(mean_metric(update_metrics, "target_mean"), 6),
            "td_abs_mean": round(mean_metric(update_metrics, "td_abs_mean"), 6),
            "grad_norm": round(mean_metric(update_metrics, "grad_norm"), 6),
            "replay_size": len(replay),
        }
        write_result(result_path, row)
        print(
            f"episode={episode:04d} steps={total_steps:07d} "
            f"reward={episode_reward:7.2f} eps={epsilon:.3f} "
            f"role={learner_agent:8s} opponent={opponent_kind:8s} "
            f"loss={row['loss']:.5f} shape=-{shaping_penalty_total:.2f} "
            f"repeat={max_repeated_action_steps} idle={max_no_reward_steps}"
        )
        if wandb_run is not None:
            training_log = {
                f"train/{key}": value
                for key, value in row.items()
                if key not in {"opponent", "learner_agent"}
            }
            training_log["train/opponent_snapshot"] = int(
                opponent_kind == "snapshot"
            )
            training_log["train/learner_second"] = int(learner_agent == SECOND_AGENT)
            wandb_run.log(training_log, step=total_steps)

        if episode % args.eval_every == 0:
            random_first_mean, _, random_first_returns = evaluate(
                policy_net,
                opponent_net,
                args,
                device,
                n_actions,
                "random",
                args.eval_seed,
                TRAIN_AGENT,
            )
            random_second_mean, _, random_second_returns = evaluate(
                policy_net,
                opponent_net,
                args,
                device,
                n_actions,
                "random",
                args.eval_seed,
                SECOND_AGENT,
            )
            random_values = np.asarray(
                random_first_returns + random_second_returns, dtype=np.float32
            )
            random_mean = float(random_values.mean())
            random_std = float(random_values.std())
            snapshot_mean = float("nan")
            snapshot_std = float("nan")
            snapshot_first_mean = float("nan")
            snapshot_second_mean = float("nan")
            if opponent_ready:
                snapshot_first_mean, _, snapshot_first_returns = evaluate(
                    policy_net,
                    opponent_net,
                    args,
                    device,
                    n_actions,
                    "snapshot",
                    args.eval_seed,
                    TRAIN_AGENT,
                )
                snapshot_second_mean, _, snapshot_second_returns = evaluate(
                    policy_net,
                    opponent_net,
                    args,
                    device,
                    n_actions,
                    "snapshot",
                    args.eval_seed,
                    SECOND_AGENT,
                )
                snapshot_values = np.asarray(
                    snapshot_first_returns + snapshot_second_returns, dtype=np.float32
                )
                snapshot_mean = float(snapshot_values.mean())
                snapshot_std = float(snapshot_values.std())
            score = random_mean if not opponent_ready else (random_mean + snapshot_mean) / 2
            eval_row = {
                "episode": episode,
                "total_steps": total_steps,
                "random_mean": round(random_mean, 4),
                "random_std": round(random_std, 4),
                "random_first_mean": round(random_first_mean, 4),
                "random_second_mean": round(random_second_mean, 4),
                "snapshot_mean": round(snapshot_mean, 4),
                "snapshot_std": round(snapshot_std, 4),
                "snapshot_first_mean": round(snapshot_first_mean, 4),
                "snapshot_second_mean": round(snapshot_second_mean, 4),
                "selection_score": round(score, 4),
            }
            write_result(evaluation_path, eval_row)
            print(
                f"evaluation random={random_mean:.2f}+/-{random_std:.2f} "
                f"(first={random_first_mean:.2f}, second={random_second_mean:.2f}) "
                f"snapshot={snapshot_mean:.2f}+/-{snapshot_std:.2f} "
                f"(first={snapshot_first_mean:.2f}, second={snapshot_second_mean:.2f})"
            )
            if wandb_run is not None:
                evaluation_log = {
                    f"eval/{key}": value for key, value in eval_row.items()
                }
                wandb_run.log(evaluation_log, step=total_steps)
            if score > best_score:
                best_score = score
                save_checkpoint(
                    best_checkpoint_path,
                    policy_net,
                    target_net,
                    opponent_net,
                    optimizer,
                    total_steps,
                    episode,
                    best_score,
                    opponent_ready,
                    args,
                    n_actions,
                )
                print(f"saved best checkpoint: {best_checkpoint_path}")

        if episode % args.save_every == 0:
            save_checkpoint(
                checkpoint_path,
                policy_net,
                target_net,
                opponent_net,
                optimizer,
                total_steps,
                episode,
                best_score,
                opponent_ready,
                args,
                n_actions,
            )

    save_checkpoint(
        checkpoint_path,
        policy_net,
        target_net,
        opponent_net,
        optimizer,
        total_steps,
        args.episodes,
        best_score,
        opponent_ready,
        args,
        n_actions,
    )
    env.close()

    if wandb_run is not None:
        import wandb

        artifact = wandb.Artifact("boxing-agent", type="model")
        artifact.add_file(str(checkpoint_path))
        if best_checkpoint_path.exists():
            artifact.add_file(str(best_checkpoint_path), name="best_dqn_boxing.pt")
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(f"Training finished / 训练完成: {checkpoint_path}")
    print(f"Training CSV / 训练记录: {result_path}")
    print(f"Evaluation CSV / 评估记录: {evaluation_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Double-Dueling DQN with snapshot-opponent training for Atari Boxing."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=100000)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--reward-clip", choices=["none", "sign"], default="sign")
    parser.add_argument(
        "--repeat-action-penalty",
        type=float,
        default=0.0,
        help="Replay-only penalty after repeating one action for too many steps.",
    )
    parser.add_argument("--repeat-action-threshold", type=int, default=12)
    parser.add_argument(
        "--inactivity-penalty",
        type=float,
        default=0.0,
        help="Replay-only penalty after too many consecutive zero-reward steps.",
    )
    parser.add_argument("--inactivity-threshold", type=int, default=75)
    parser.add_argument(
        "--opponent", choices=["random", "snapshot", "mixed"], default="mixed"
    )
    parser.add_argument(
        "--train-role",
        choices=["first", "second", "random", "alternate"],
        default="first",
        help="Player role controlled by the learner in each training episode.",
    )
    parser.add_argument("--random-opponent-prob", type=float, default=0.5)
    parser.add_argument("--opponent-epsilon", type=float, default=0.05)
    parser.add_argument("--opponent-learning-starts", type=int, default=10000)
    parser.add_argument("--opponent-update", type=int, default=10000)
    parser.add_argument("--eval-opponent-epsilon", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--checkpoint", default="checkpoints/day4_boxing_latest.pt")
    parser.add_argument("--best-checkpoint", default="checkpoints/day4_boxing_best.pt")
    parser.add_argument("--results", default="results/day4_boxing_training.csv")
    parser.add_argument("--evaluation-results", default="results/day4_boxing_evaluation.csv")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="ai-and-computer-games")
    parser.add_argument("--run-name", default="day4-double-dueling-dqn")
    args = parser.parse_args()
    if not 0.0 <= args.random_opponent_prob <= 1.0:
        parser.error("--random-opponent-prob must be between 0 and 1")
    return args


if __name__ == "__main__":
    train(parse_args())
