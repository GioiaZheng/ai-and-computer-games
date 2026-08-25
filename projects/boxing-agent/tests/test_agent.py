"""Smoke tests for the packaged tournament agent."""

from pathlib import Path
import unittest

import numpy as np
import torch

from sample_agent.agent_template import Agent, DQN, NUM_ACTIONS


class TournamentAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent(env=None)

    def test_packaged_weights_exist(self):
        weights = Path(__file__).parents[1] / "sample_agent" / "policy_weights.pt"
        self.assertTrue(weights.is_file())
        self.assertGreater(weights.stat().st_size, 0)

    def test_network_parameter_count(self):
        network = DQN(frame_stack=6, n_actions=NUM_ACTIONS)
        parameter_count = sum(parameter.numel() for parameter in network.parameters())
        self.assertEqual(parameter_count, 1_697_971)

    def test_forward_output_shape_and_values(self):
        state = torch.zeros((1, 6, 84, 84), dtype=torch.uint8)
        with torch.inference_mode():
            q_values = self.agent.net(state.to(self.agent.device)).cpu()
        self.assertEqual(tuple(q_values.shape), (1, NUM_ACTIONS))
        self.assertTrue(torch.isfinite(q_values).all())

    def test_action_contract(self):
        state = np.zeros((84, 84, 6), dtype=np.uint8)
        action = self.agent.get_action(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, NUM_ACTIONS)

    def test_none_state_resets_controller(self):
        self.agent.repeat_count = 9
        self.agent.escape_remaining = 3
        action = self.agent.get_action(None)
        self.assertEqual(action, 0)
        self.assertEqual(self.agent.repeat_count, 0)
        self.assertEqual(self.agent.escape_remaining, 0)
        self.assertEqual(self.agent.episode_step, 0)

    def test_invalid_observation_shape_is_rejected(self):
        invalid = np.zeros((84, 84, 4), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Unexpected observation shape"):
            self.agent.get_action(invalid)


if __name__ == "__main__":
    unittest.main()
