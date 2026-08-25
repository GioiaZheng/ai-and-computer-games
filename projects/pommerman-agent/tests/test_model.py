import torch

from src.encoding import NUM_ACTIONS, NUM_CHANNELS
from src.model import ActorCritic


def test_actor_critic_shapes():
    model = ActorCritic()
    observations = torch.zeros((3, NUM_CHANNELS, 11, 11))
    logits, values = model(observations)
    assert logits.shape == (3, NUM_ACTIONS)
    assert values.shape == (3,)


def test_distribution_never_samples_masked_actions():
    model = ActorCritic()
    observations = torch.zeros((32, NUM_CHANNELS, 11, 11))
    masks = torch.zeros((32, NUM_ACTIONS), dtype=torch.bool)
    masks[:, 2] = True
    distribution, _ = model.distribution(observations, masks)
    assert torch.all(distribution.sample() == 2)
