"""Opponent policy adapters for evaluation and frozen self-play."""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np

from .environment import ACTION_SETS, AGENTS
from .scripted_agent import ScriptedOpponentPolicy, SpawnHunterAgent


class PPOOpponentPolicy:
    """Run a frozen PPO checkpoint behind the environment opponent callback."""

    def __init__(
        self,
        model_path: str | Path,
        action_set: str = "fire",
        scripted_prefix_steps: int = 2024,
        device: str = "cpu",
        deterministic: bool = True,
        max_action_repeat: int = 24,
        no_hit_patience: int = 90,
        tactic_block_steps: int = 60,
    ):
        self.model_path = str(model_path)
        self.actions = ACTION_SETS[action_set]
        self.scripted_prefix_steps = scripted_prefix_steps
        self.device = device
        self.deterministic = deterministic
        self.max_action_repeat = max_action_repeat
        self.no_hit_patience = no_hit_patience
        self.tactic_block_steps = tactic_block_steps
        self.model = None
        self.steps = {role: 0 for role in AGENTS}
        self.experts = {role: SpawnHunterAgent(role=role) for role in AGENTS}
        self.last_policy_action = {role: None for role in AGENTS}
        self.action_repeats = {role: 0 for role in AGENTS}
        self.no_hit_steps = {role: 0 for role in AGENTS}
        self.recent_policy_actions = {
            role: deque(maxlen=no_hit_patience) for role in AGENTS
        }
        self.blocked_policy_action = {role: None for role in AGENTS}
        self.block_remaining = {role: 0 for role in AGENTS}

    def _load_model(self):
        if self.model is None:
            from stable_baselines3 import PPO

            self.model = PPO.load(self.model_path, device=self.device)

    def reset(self, role: str):
        self.steps[role] = 0
        self.experts[role].reset()
        self.last_policy_action[role] = None
        self.action_repeats[role] = 0
        self.no_hit_steps[role] = 0
        self.recent_policy_actions[role].clear()
        self.blocked_policy_action[role] = None
        self.block_remaining[role] = 0

    def _alternative_action(self, observation, excluded_actions: set[int]) -> int:
        observation_tensor, _ = self.model.policy.obs_to_tensor(observation)
        distribution = self.model.policy.get_distribution(observation_tensor)
        probabilities = (
            distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
        )
        for excluded_action in excluded_actions:
            probabilities[excluded_action] = 0.0
        probability_sum = probabilities.sum()
        if probability_sum <= 0:
            available = [
                action
                for action in range(len(self.actions))
                if action not in excluded_actions
            ]
            return available[0] if available else 0
        probabilities /= probability_sum
        return int(np.random.choice(len(self.actions), p=probabilities))

    def __call__(self, observation, role: str) -> int:
        step = self.steps[role]
        self.steps[role] += 1
        if step < self.scripted_prefix_steps:
            return self.experts[role].get_action(observation)
        self._load_model()
        policy_action, _ = self.model.predict(
            observation, deterministic=self.deterministic
        )
        policy_action = int(np.asarray(policy_action).item())
        recent = self.recent_policy_actions[role]
        if (
            self.no_hit_steps[role] >= self.no_hit_patience
            and len(recent) >= self.no_hit_patience // 2
        ):
            dominant_action, dominant_count = Counter(recent).most_common(1)[0]
            if dominant_count / len(recent) >= 0.50:
                self.blocked_policy_action[role] = dominant_action
                self.block_remaining[role] = self.tactic_block_steps
                self.no_hit_steps[role] = 0
                recent.clear()
        blocked_action = self.blocked_policy_action[role]
        if self.block_remaining[role] > 0:
            self.block_remaining[role] -= 1
            if policy_action == blocked_action:
                policy_action = self._alternative_action(
                    observation, {blocked_action}
                )
        else:
            self.blocked_policy_action[role] = None
        if policy_action == self.last_policy_action[role]:
            self.action_repeats[role] += 1
        else:
            self.last_policy_action[role] = policy_action
            self.action_repeats[role] = 1
        if self.action_repeats[role] > self.max_action_repeat:
            exclusions = {policy_action}
            if self.blocked_policy_action[role] is not None:
                exclusions.add(self.blocked_policy_action[role])
            policy_action = self._alternative_action(observation, exclusions)
            self.last_policy_action[role] = policy_action
            self.action_repeats[role] = 1
        recent.append(policy_action)
        return self.actions[policy_action]

    def observe_reward(self, reward: float, role: str):
        self.experts[role].observe_reward(reward)
        if reward > 0:
            self.action_repeats[role] = 0
            self.no_hit_steps[role] = 0
            self.recent_policy_actions[role].clear()
            self.blocked_policy_action[role] = None
            self.block_remaining[role] = 0
        else:
            self.no_hit_steps[role] += 1


class MixedOpponentPolicy:
    """Sample random, scripted, or frozen-model opponents per episode."""

    def __init__(
        self,
        model_path: str | Path,
        action_set: str = "fire",
        scripted_prefix_steps: int = 2024,
        device: str = "cpu",
        deterministic: bool = False,
        seed: int = 0,
        probabilities: tuple[float, float, float] = (0.15, 0.35, 0.50),
    ):
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("Mixed opponent probabilities must sum to one")
        self.rng = np.random.default_rng(seed)
        self.probabilities = probabilities
        self.selected = {role: "random" for role in AGENTS}
        self.scripted = ScriptedOpponentPolicy()
        self.model = PPOOpponentPolicy(
            model_path,
            action_set=action_set,
            scripted_prefix_steps=scripted_prefix_steps,
            device=device,
            deterministic=deterministic,
        )

    def reset(self, role: str):
        self.selected[role] = str(
            self.rng.choice(("random", "scripted", "model"), p=self.probabilities)
        )
        self.scripted.reset(role)
        self.model.reset(role)

    def __call__(self, observation, role: str) -> int:
        opponent = self.selected[role]
        if opponent == "random":
            return int(self.rng.integers(0, 18))
        if opponent == "scripted":
            return self.scripted(observation, role)
        return self.model(observation, role)

    def observe_reward(self, reward: float, role: str):
        opponent = self.selected[role]
        if opponent == "scripted":
            self.scripted.observe_reward(reward, role)
        elif opponent == "model":
            self.model.observe_reward(reward, role)
