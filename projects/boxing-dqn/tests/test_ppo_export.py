"""CPU checks for PPO export and tournament inference compatibility."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from export_ppo_policy import QNetwork  # noqa: E402
from ppo_agent_template import PolicyNetwork  # noqa: E402
from ppo_boxing import validated_action  # noqa: E402


class PPOExportTests(unittest.TestCase):
    def test_export_and_submission_networks_match(self):
        exported = QNetwork()
        submitted = PolicyNetwork()
        submitted.load_state_dict(exported.state_dict())

        generator = torch.Generator().manual_seed(4)
        observation = torch.randint(
            0, 256, (2, 84, 84, 6), dtype=torch.uint8, generator=generator
        )
        with torch.inference_mode():
            expected = exported(observation)
            actual = submitted(observation)
        self.assertEqual(tuple(actual.shape), (2, 18))
        self.assertTrue(torch.equal(actual, expected))

    def test_external_action_validation(self):
        class ValidAgent:
            def get_action(self, observation):
                self.observation_shape = observation.shape
                return 17

        class InvalidAgent:
            def get_action(self, observation):
                return -1

        observation = np.zeros((84, 84, 6), dtype=np.uint8)
        valid = ValidAgent()
        self.assertEqual(validated_action(valid, observation), 17)
        self.assertEqual(valid.observation_shape, observation.shape)
        with self.assertRaises(ValueError):
            validated_action(InvalidAgent(), observation)


if __name__ == "__main__":
    unittest.main()
