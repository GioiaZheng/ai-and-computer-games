"""Observation encoding and action constraints for Pommerman FFA."""

from typing import Dict, Iterable, Tuple

import numpy as np


BOARD_SIZE = 11
NUM_ACTIONS = 6
NUM_CHANNELS = 16

PASSAGE = 0
RIGID = 1
WOOD = 2
BOMB = 3
FLAMES = 4
EXTRA_BOMB = 6
INCREASE_RANGE = 7
KICK = 8
FIRST_AGENT = 10
LAST_AGENT = 13

CHANNEL_NAMES = (
    "passage",
    "rigid_wall",
    "wood_wall",
    "flames",
    "extra_bomb_powerup",
    "range_powerup",
    "kick_powerup",
    "bomb",
    "bomb_life",
    "bomb_blast_strength",
    "self",
    "enemies",
    "ammo",
    "blast_strength",
    "can_kick",
    "alive_fraction",
)


def _constant_plane(value: float) -> np.ndarray:
    return np.full((BOARD_SIZE, BOARD_SIZE), value, dtype=np.float32)


def encode_observation(observation: Dict) -> np.ndarray:
    """Convert one Pommerman observation into a role-invariant tensor.

    Board entities become spatial channels. Scalar capabilities are repeated as
    constant planes so a compact CNN can consume the complete observation.
    """

    board = np.asarray(observation["board"], dtype=np.int16)
    if board.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("expected an 11x11 Pommerman board")

    bomb_life = np.asarray(observation["bomb_life"], dtype=np.float32)
    bomb_strength = np.asarray(
        observation["bomb_blast_strength"], dtype=np.float32
    )
    position = tuple(int(value) for value in observation["position"])

    self_plane = np.zeros_like(board, dtype=np.float32)
    self_plane[position] = 1.0

    enemy_plane = ((board >= FIRST_AGENT) & (board <= LAST_AGENT)).astype(
        np.float32
    )
    enemy_plane[position] = 0.0

    alive = observation.get("alive", ())
    alive_fraction = min(len(alive), 4) / 4.0

    channels = (
        (board == PASSAGE).astype(np.float32),
        (board == RIGID).astype(np.float32),
        (board == WOOD).astype(np.float32),
        (board == FLAMES).astype(np.float32),
        (board == EXTRA_BOMB).astype(np.float32),
        (board == INCREASE_RANGE).astype(np.float32),
        (board == KICK).astype(np.float32),
        ((board == BOMB) | (bomb_life > 0)).astype(np.float32),
        np.clip(bomb_life / 10.0, 0.0, 1.0),
        np.clip(bomb_strength / 10.0, 0.0, 1.0),
        self_plane,
        enemy_plane,
        _constant_plane(min(float(observation.get("ammo", 0)) / 5.0, 1.0)),
        _constant_plane(
            min(float(observation.get("blast_strength", 0)) / 10.0, 1.0)
        ),
        _constant_plane(float(bool(observation.get("can_kick", False)))),
        _constant_plane(alive_fraction),
    )
    encoded = np.stack(channels, axis=0).astype(np.float32, copy=False)
    if encoded.shape != (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE):
        raise RuntimeError("unexpected encoded observation shape")
    return encoded


def valid_action_mask(observation: Dict) -> np.ndarray:
    """Return the six actions that are physically available in this state."""

    board = np.asarray(observation["board"])
    bomb_life = np.asarray(observation["bomb_life"])
    row, column = (int(value) for value in observation["position"])
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[0] = True

    passable = {PASSAGE, FLAMES, EXTRA_BOMB, INCREASE_RANGE, KICK}
    directions = {
        1: (row - 1, column),
        2: (row + 1, column),
        3: (row, column - 1),
        4: (row, column + 1),
    }
    for action, (next_row, next_column) in directions.items():
        if not (0 <= next_row < BOARD_SIZE and 0 <= next_column < BOARD_SIZE):
            continue
        if int(board[next_row, next_column]) in passable:
            mask[action] = True

    has_ammo = int(observation.get("ammo", 0)) > 0
    standing_on_bomb = float(bomb_life[row, column]) > 0
    mask[5] = has_ammo and not standing_on_bomb
    return mask


def alive_enemy_count(observation: Dict, self_agent_value: int = None) -> int:
    """Count living opponents using the observation's alive agent IDs."""

    alive = {int(value) for value in observation.get("alive", ())}
    if self_agent_value is None:
        row, column = (int(value) for value in observation["position"])
        board_value = int(np.asarray(observation["board"])[row, column])
        self_agent_value = board_value if FIRST_AGENT <= board_value <= LAST_AGENT else -1
    return sum(value != self_agent_value for value in alive)


def observation_position(observation: Dict) -> Tuple[int, int]:
    return tuple(int(value) for value in observation["position"])
