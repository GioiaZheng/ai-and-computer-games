import numpy as np

from src.environment import (
    ACTION_SETS,
    FRAME_SIZE,
    FRAME_STACK,
    FRAME_CHANNELS,
    REFERENCE_FRAME_HEIGHT,
    REFERENCE_FRAME_WIDTH,
    TANK_COLORS,
    OFFICIAL_ENVIRONMENT,
    OFFICIAL_ENV_KWARGS,
    FrameHistory,
    navigation_distance,
    preprocess_frame,
    tank_center,
    tank_heading,
)
from src.scripted_agent import (
    ScriptedOpponentPolicy,
    SpawnHunterAgent,
    WaypointCoverageAgent,
)
from src.opponents import MixedOpponentPolicy, PPOOpponentPolicy
from src.curriculum import ScriptedPrefixWrapper


def test_preprocess_frame_shape_and_dtype():
    observation = np.zeros((256, 160, 3), dtype=np.uint8)
    result = preprocess_frame(observation)
    assert result.shape == (FRAME_SIZE, FRAME_SIZE, FRAME_CHANNELS)
    assert result.dtype == np.uint8


def test_frame_history_stacks_four_frames():
    history = FrameHistory()
    observation = np.zeros((256, 160, 3), dtype=np.uint8)
    result = history.reset(observation)
    assert result.shape == (FRAME_SIZE, FRAME_SIZE, FRAME_STACK * FRAME_CHANNELS)
    assert result.dtype == np.uint8


def test_tank_center_preserves_raw_combat_tank_coordinates():
    observation = np.zeros(
        (REFERENCE_FRAME_HEIGHT, REFERENCE_FRAME_WIDTH, 3), dtype=np.uint8
    )
    observation[180:193, 10:20] = TANK_COLORS["first_0"]
    center = tank_center(observation, "first_0")
    assert center is not None
    assert np.allclose(center, (14.5, 186.0))


def test_waypoint_teacher_moves_away_from_spawn_wall_first():
    observation = np.zeros(
        (REFERENCE_FRAME_HEIGHT, REFERENCE_FRAME_WIDTH, 3), dtype=np.uint8
    )
    observation[130:144, 11:17] = TANK_COLORS["first_0"]
    observation[136:138, 17:20] = TANK_COLORS["first_0"]
    teacher = WaypointCoverageAgent("first_0", route="top")
    assert teacher.get_action(observation) == 4


def test_waypoint_teacher_probes_forward_after_short_turn():
    observation = np.zeros(
        (REFERENCE_FRAME_HEIGHT, REFERENCE_FRAME_WIDTH, 3), dtype=np.uint8
    )
    observation[130:144, 11:17] = TANK_COLORS["first_0"]
    observation[136:138, 17:20] = TANK_COLORS["first_0"]
    teacher = WaypointCoverageAgent("first_0", route="top")
    turn_actions = [teacher.get_action(observation) for _ in range(8)]
    assert set(turn_actions) == {4}
    assert teacher.get_action(observation) == 2


def test_waypoint_teacher_changes_heading_after_failed_probe():
    observation = np.zeros(
        (REFERENCE_FRAME_HEIGHT, REFERENCE_FRAME_WIDTH, 3), dtype=np.uint8
    )
    observation[130:144, 11:17] = TANK_COLORS["first_0"]
    observation[136:138, 17:20] = TANK_COLORS["first_0"]
    teacher = WaypointCoverageAgent("first_0", route="top")
    actions = [teacher.get_action(observation) for _ in range(8 + 16)]
    assert actions.count(2) == 16
    assert teacher.get_action(observation) == 4


def test_waypoint_teacher_targets_reachable_maze_passages():
    teacher = WaypointCoverageAgent("first_0", route="top")
    waypoints = teacher._waypoints()
    assert all(
        0 <= x < REFERENCE_FRAME_WIDTH and 35 <= y < REFERENCE_FRAME_HEIGHT - 10
        for x, y in waypoints
    )
    assert waypoints[0] == (8.0, 136.0)
    assert waypoints[1][1] == 91.0
    assert any(y == 186.0 for _, y in waypoints)


def test_waypoint_teacher_accepts_tank_sized_gate_crossing_region():
    teacher = WaypointCoverageAgent("first_0", route="bottom")
    teacher.waypoint_index = 2
    target = teacher._waypoints()[teacher.waypoint_index]
    assert target == (80.0, 186.0)
    assert teacher._target_reached((69.0, 194.5), target)
    assert teacher._target_reached((77.0, 173.0), target)


def test_waypoint_teacher_keeps_spawn_clearance_strict():
    teacher = WaypointCoverageAgent("first_0", route="top")
    target = teacher._waypoints()[0]
    assert not teacher._target_reached((14.0, 136.0), target)


def test_diagonal_fire_curriculum_uses_legal_actions():
    assert ACTION_SETS["fire_diagonal"] == (14, 15, 16, 17)


def test_sweep_curriculum_uses_turning_fire_actions():
    assert ACTION_SETS["sweep"] == (11, 12)


def test_official_environment_configuration():
    assert OFFICIAL_ENVIRONMENT == "atari/combat_tank-v2"
    assert OFFICIAL_ENV_KWARGS == {
        "has_maze": True,
        "is_invisible": False,
        "billiard_hit": False,
    }


def test_navigation_distance_uses_maze_gates():
    direct = navigation_distance((40.0, 120.0), (120.0, 120.0))
    around_walls = navigation_distance((14.0, 136.0), (146.0, 136.0))
    assert direct == 80.0
    assert around_walls > direct


def test_spawn_hunter_mirrors_turns_between_roles():
    first = SpawnHunterAgent("first_0")
    second = SpawnHunterAgent("second_0")
    assert first.get_action() == 4
    assert second.get_action() == 3


def test_spawn_hunter_reacts_to_hit_feedback():
    agent = SpawnHunterAgent("first_0")
    agent.observe_reward(1.0)
    assert agent.get_action() == 10
    agent.observe_reward(-1.0)
    assert agent.get_action() == 14
    agent.observe_reward(-1.0)
    assert agent.get_action() == 15


def test_scripted_opponent_tracks_each_role_separately():
    opponent = ScriptedOpponentPolicy()
    assert opponent(None, "first_0") == 4
    assert opponent(None, "second_0") == 3


def test_ppo_opponent_uses_scripted_prefix_without_loading_model():
    opponent = PPOOpponentPolicy("missing.zip", scripted_prefix_steps=3)
    assert opponent(None, "first_0") == 4
    assert opponent(None, "first_0") == 4
    opponent.observe_reward(1.0, "first_0")
    assert opponent(None, "first_0") == 10
    assert opponent(None, "second_0") == 3


def test_mixed_opponent_can_select_random_without_loading_model():
    opponent = MixedOpponentPolicy(
        "missing.zip", probabilities=(1.0, 0.0, 0.0), seed=7
    )
    opponent.reset("first_0")
    assert 0 <= opponent(None, "first_0") < 18


def test_scripted_prefix_rejects_negative_steps():
    class Placeholder:
        pass

    try:
        ScriptedPrefixWrapper(Placeholder(), -1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative prefix_steps should fail")
