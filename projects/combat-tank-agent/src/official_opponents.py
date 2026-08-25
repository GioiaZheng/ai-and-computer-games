"""Opponent adapters that consume official six-channel observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OfficialPPOOpponent:
    """Use a frozen official-pipeline PPO checkpoint as the opponent."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        deterministic: bool = False,
    ):
        self.model_path = str(model_path)
        self.device = device
        self.deterministic = deterministic
        self.model = None

    def _load(self):
        if self.model is None:
            from stable_baselines3 import PPO

            self.model = PPO.load(self.model_path, device=self.device)
            if tuple(self.model.observation_space.shape) != (6, 84, 84):
                raise ValueError(
                    "Opponent checkpoint is not compatible with the official pipeline: "
                    f"{self.model.observation_space}"
                )

    def reset(self, role: str):
        return None

    def __call__(self, observation, role: str) -> int:
        self._load()
        action, _ = self.model.predict(
            observation,
            deterministic=self.deterministic,
        )
        return int(np.asarray(action).item())


class OfficialMixedOpponent:
    """Select a random or frozen PPO opponent at the start of each episode."""

    def __init__(
        self,
        model_path: str | Path,
        model_probability: float = 0.70,
        device: str = "cpu",
        deterministic: bool = False,
        seed: int = 0,
    ):
        if not 0.0 <= model_probability <= 1.0:
            raise ValueError("model_probability must be between zero and one")
        self.rng = np.random.default_rng(seed)
        self.model_probability = model_probability
        self.model = OfficialPPOOpponent(
            model_path,
            device=device,
            deterministic=deterministic,
        )
        self.use_model = False

    def reset(self, role: str):
        self.use_model = bool(self.rng.random() < self.model_probability)
        self.model.reset(role)

    def __call__(self, observation, role: str) -> int:
        if self.use_model:
            return self.model(observation, role)
        return int(self.rng.integers(0, 18))
