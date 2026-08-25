import numpy as np
import torch

from src.encoding import NUM_CHANNELS
from src.model import ActorCritic
from src.train_bc import train_batch


def test_behavioral_cloning_loss_is_finite_for_every_expert_action():
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = [
        (np.zeros((NUM_CHANNELS, 11, 11), dtype=np.float32), action)
        for action in range(6)
    ]

    loss, accuracy = train_batch(model, optimizer, samples, torch.device("cpu"))

    assert np.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
