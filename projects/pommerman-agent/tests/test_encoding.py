import numpy as np

from src.encoding import NUM_CHANNELS, encode_observation, valid_action_mask


def sample_observation():
    board = np.ones((11, 11), dtype=np.int16)
    board[1:10, 1:10] = 0
    board[5, 5] = 10
    board[5, 7] = 11
    board[4, 5] = 2
    bomb_life = np.zeros((11, 11), dtype=np.float32)
    bomb_strength = np.zeros((11, 11), dtype=np.float32)
    return {
        "board": board,
        "bomb_life": bomb_life,
        "bomb_blast_strength": bomb_strength,
        "position": (5, 5),
        "ammo": 1,
        "blast_strength": 2,
        "can_kick": False,
        "alive": [10, 11],
        "enemies": [11, 12, 13],
    }


def test_encode_observation_shape_and_identity_channels():
    encoded = encode_observation(sample_observation())
    assert encoded.shape == (NUM_CHANNELS, 11, 11)
    assert encoded.dtype == np.float32
    assert encoded[10, 5, 5] == 1.0
    assert encoded[11, 5, 7] == 1.0
    assert encoded[11, 5, 5] == 0.0


def test_valid_action_mask_rejects_wall_and_allows_bomb():
    mask = valid_action_mask(sample_observation())
    assert mask.tolist() == [True, False, True, True, True, True]


def test_valid_action_mask_rejects_second_bomb_on_same_cell():
    observation = sample_observation()
    observation["bomb_life"][5, 5] = 9
    mask = valid_action_mask(observation)
    assert not mask[5]
