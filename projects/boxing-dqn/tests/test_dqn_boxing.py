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
    external_agent_action,
    linear_epsilon,
    normalized_action_entropy,
    observation_to_state,
    optimize,
    state_to_observation,
)


class BoxingDQNTests(unittest.TestCase):
    def test_dueling_network_shape(self):
        model = DQN(frame_stack=6, n_actions=18)
        output = model(torch.zeros(2, 6, 84, 84, dtype=torch.uint8))
        self.assertEqual(tuple(output.shape), (2, 18))
        self.assertTrue(torch.isfinite(output).all())

    def test_official_observation_conversion(self):
        observation = np.zeros((84, 84, 6), dtype=np.uint8)
        observation[..., 4] = 255
        state = observation_to_state(observation)
        self.assertEqual(state.shape, (6, 84, 84))
        self.assertEqual(state.dtype, np.uint8)
        self.assertTrue(np.all(state[4] == 255))
        reconstructed = state_to_observation(state)
        self.assertEqual(reconstructed.shape, observation.shape)
        self.assertTrue(np.array_equal(reconstructed, observation))

    def test_external_agent_action_validation(self):
        class ValidAgent:
            def get_action(self, observation):
                self.shape = observation.shape
                return 17

        class InvalidAgent:
            def get_action(self, observation):
                return 18

        state = np.zeros((6, 84, 84), dtype=np.uint8)
        valid = ValidAgent()
        self.assertEqual(external_agent_action(valid, state, 18), 17)
        self.assertEqual(valid.shape, (84, 84, 6))
        with self.assertRaises(ValueError):
            external_agent_action(InvalidAgent(), state, 18)

    def test_epsilon_schedule_endpoints(self):
        self.assertAlmostEqual(linear_epsilon(0, 1.0, 0.05, 100), 1.0)
        self.assertAlmostEqual(linear_epsilon(50, 1.0, 0.05, 100), 0.525)
        self.assertAlmostEqual(linear_epsilon(200, 1.0, 0.05, 100), 0.05)

    def test_action_entropy_reports_coverage_evenness(self):
        uniform = np.ones(18, dtype=np.int64)
        collapsed = np.zeros(18, dtype=np.int64)
        collapsed[0] = 100
        self.assertAlmostEqual(normalized_action_entropy(uniform), 1.0)
        self.assertAlmostEqual(normalized_action_entropy(collapsed), 0.0)
        self.assertAlmostEqual(
            normalized_action_entropy(np.zeros(18, dtype=np.int64)), 0.0
        )

    def test_training_role_selection(self):
        import random

        rng = random.Random(7)
        self.assertEqual(choose_training_agent("first", rng), "first_0")
        self.assertEqual(choose_training_agent("second", rng), "second_0")
        self.assertEqual(choose_training_agent("alternate", rng, 2), "first_0")
        self.assertEqual(choose_training_agent("alternate", rng, 3), "second_0")
        roles = {choose_training_agent("random", rng) for _ in range(20)}
        self.assertEqual(roles, {"first_0", "second_0"})

    def test_one_double_dqn_update(self):
        device = torch.device("cpu")
        policy = DQN(6, 3).to(device)
        target = DQN(6, 3).to(device)
        target.load_state_dict(policy.state_dict())
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        replay = ReplayBuffer(16, (6, 84, 84), seed=1)
        state = np.zeros((6, 84, 84), dtype=np.uint8)
        for index in range(8):
            replay.push(state, index % 3, float(index % 2), state, index == 7)
        args = SimpleNamespace(learning_starts=1, batch_size=4, gamma=0.99, grad_clip=10.0)
        metrics = optimize(policy, target, replay, optimizer, args, device)
        self.assertIsNotNone(metrics)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
