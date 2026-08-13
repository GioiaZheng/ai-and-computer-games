"""CPU smoke tests for the Day 4 learner / Day 4 训练器 CPU 快速测试。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dqn_boxing import (  # noqa: E402
    DQN,
    ReplayBuffer,
    choose_training_agent,
    linear_epsilon,
    optimize,
)


class BoxingDQNTests(unittest.TestCase):
    def test_dueling_network_shape(self):
        model = DQN(frame_stack=4, n_actions=18)
        output = model(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))
        self.assertEqual(tuple(output.shape), (2, 18))
        self.assertTrue(torch.isfinite(output).all())

    def test_epsilon_schedule_endpoints(self):
        self.assertAlmostEqual(linear_epsilon(0, 1.0, 0.05, 100), 1.0)
        self.assertAlmostEqual(linear_epsilon(50, 1.0, 0.05, 100), 0.525)
        self.assertAlmostEqual(linear_epsilon(200, 1.0, 0.05, 100), 0.05)

    def test_training_role_selection(self):
        import random

        rng = random.Random(7)
        self.assertEqual(choose_training_agent("first", rng), "first_0")
        self.assertEqual(choose_training_agent("second", rng), "second_0")
        roles = {choose_training_agent("random", rng) for _ in range(20)}
        self.assertEqual(roles, {"first_0", "second_0"})

    def test_one_double_dqn_update(self):
        device = torch.device("cpu")
        policy = DQN(4, 3).to(device)
        target = DQN(4, 3).to(device)
        target.load_state_dict(policy.state_dict())
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        replay = ReplayBuffer(16, (4, 84, 84), seed=1)
        state = np.zeros((4, 84, 84), dtype=np.uint8)
        for index in range(8):
            replay.push(state, index % 3, float(index % 2), state, index == 7)
        args = SimpleNamespace(learning_starts=1, batch_size=4, gamma=0.99, grad_clip=10.0)
        metrics = optimize(policy, target, replay, optimizer, args, device)
        self.assertIsNotNone(metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
