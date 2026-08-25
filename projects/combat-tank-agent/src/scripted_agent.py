"""Deterministic maze-navigation policies used to bootstrap learning."""

from __future__ import annotations

from dataclasses import dataclass

from .environment import AGENTS, LEFT_GATES, tank_center


@dataclass
class WaypointCoverageAgent:
    """Closed-loop teacher that visits both maze gates and all three regions."""

    role: str
    route: str = "top"
    tolerance: float = 3.0
    turn_probe_frames: int = 8
    forward_probe_frames: int = 16
    waypoint_index: int = 0
    waypoints_reached: int = 0
    turn_remaining: int = 0
    forward_remaining: int = 0
    probe_start_distance: float | None = None

    def __post_init__(self):
        if self.role not in AGENTS:
            raise ValueError(f"Unknown role: {self.role}")
        if self.route not in {"top", "bottom"}:
            raise ValueError(f"Unknown route: {self.route}")
        self.turn_remaining = self.turn_probe_frames

    def reset(self):
        self.waypoint_index = 0
        self.waypoints_reached = 0
        self.turn_remaining = self.turn_probe_frames
        self.forward_remaining = 0
        self.probe_start_distance = None

    def _waypoints(self):
        # Spawned tanks touch the long side walls. Move a few pixels toward
        # the outer border before travelling vertically, otherwise their body
        # remains caught on the wall even when the requested direction is open.
        own_x = 8.0 if self.role == AGENTS[0] else 152.0
        other_x = 152.0 if self.role == AGENTS[0] else 8.0
        center_x = 80.0
        # These are centres of the real horizontal passages in the 160x256
        # game frame. Using y=30 or y=246 makes the tank press against the
        # arena border instead of entering a gate.
        top_y = LEFT_GATES[0][1]
        middle_y = 136.0
        bottom_y = LEFT_GATES[1][1]
        if self.route == "top":
            return (
                (own_x, middle_y),
                (own_x, top_y),
                (center_x, top_y),
                (center_x, middle_y),
                (center_x, bottom_y),
                (other_x, bottom_y),
                (other_x, middle_y),
                (other_x, top_y),
                (center_x, top_y),
                (own_x, top_y),
                (own_x, middle_y),
                (own_x, bottom_y),
                (center_x, bottom_y),
            )
        return (
            (own_x, middle_y),
            (own_x, bottom_y),
            (center_x, bottom_y),
            (center_x, middle_y),
            (center_x, top_y),
            (other_x, top_y),
            (other_x, middle_y),
            (other_x, bottom_y),
            (center_x, bottom_y),
            (own_x, bottom_y),
            (own_x, middle_y),
            (own_x, top_y),
            (center_x, top_y),
        )

    def _target_reached(
        self,
        center: tuple[float, float],
        target: tuple[float, float],
    ) -> bool:
        """Treat waypoints as tank-sized regions instead of single pixels."""
        if self.waypoint_index == 0:
            # The first target deliberately pulls the tank clear of its spawn
            # wall, so it keeps the stricter point-level tolerance.
            x_tolerance = self.tolerance
            y_tolerance = self.tolerance
        elif target[0] == 80.0:
            # A tank crossing a gate can pass on either side of the exact
            # centreline. Requiring +/-3 pixels made valid crossings orbit the
            # target forever because the sprite itself is roughly this wide.
            x_tolerance = 14.0
            y_tolerance = 14.0
        else:
            x_tolerance = 7.0
            y_tolerance = 10.0
        return (
            abs(center[0] - target[0]) <= x_tolerance
            and abs(center[1] - target[1]) <= y_tolerance
        )

    def get_action(self, observation=None) -> int:
        center = tank_center(observation, self.role) if observation is not None else None
        if center is None:
            return 4 if self.role == AGENTS[0] else 3
        waypoints = self._waypoints()
        target = waypoints[self.waypoint_index % len(waypoints)]
        if self._target_reached(center, target):
            self.waypoints_reached += 1
            self.waypoint_index = (self.waypoint_index + 1) % len(waypoints)
            target = waypoints[self.waypoint_index]
            self.turn_remaining = 0
            self.forward_remaining = 0
            self.probe_start_distance = None

        dx = target[0] - center[0]
        dy = target[1] - center[1]
        distance = (dx * dx + dy * dy) ** 0.5

        if self.turn_remaining > 0:
            self.turn_remaining -= 1
            return 4
        if self.forward_remaining > 0:
            self.forward_remaining -= 1
            return 2

        if self.probe_start_distance is not None:
            progress = self.probe_start_distance - distance
            if progress < 0.75:
                self.probe_start_distance = None
                self.turn_remaining = max(self.turn_probe_frames - 1, 0)
                return 4

        self.probe_start_distance = distance
        self.forward_remaining = max(self.forward_probe_frames - 1, 0)
        return 2

    def observe_reward(self, _reward: float):
        """Keep the navigation route independent of combat score."""


@dataclass
class MazeRushAgent:
    """Use mirrored tank controls to enter the arena through the top opening."""

    role: str
    vertical_frames: int = 250
    turn_frames: int = 48
    step_count: int = 0

    def __post_init__(self):
        if self.role not in AGENTS:
            raise ValueError(f"Unknown role: {self.role}")

    def reset(self):
        self.step_count = 0

    def get_action(self, _observation=None) -> int:
        first_turn = 4 if self.role == AGENTS[0] else 3
        second_turn = 3 if self.role == AGENTS[0] else 4
        boundaries = (
            self.turn_frames,
            self.turn_frames + self.vertical_frames,
            2 * self.turn_frames + self.vertical_frames,
        )
        if self.step_count < boundaries[0]:
            action = first_turn
        elif self.step_count < boundaries[1]:
            action = 2
        elif self.step_count < boundaries[2]:
            action = second_turn
        else:
            action = 10
        self.step_count += 1
        return action


@dataclass
class SpawnHunterAgent:
    """Cross the top gate, then approach and fire toward the opposing spawn."""

    role: str
    vertical_frames: int = 420
    cross_frames: int = 1040
    descent_frames: int = 420
    turn_frames: int = 48
    sweep_actions: tuple[int, ...] | None = None
    sweep_block: int = 96
    advance_frames: int = 96
    attack_lock_frames: int = 120
    evade_frames: int = 48
    step_count: int = 0
    attack_remaining: int = 0
    evade_remaining: int = 0
    evade_count: int = 0
    last_action: int = 1

    def __post_init__(self):
        if self.role not in AGENTS:
            raise ValueError(f"Unknown role: {self.role}")
        if self.sweep_actions is None:
            self.sweep_actions = (
                (11, 11, 12) if self.role == AGENTS[0] else (11, 12, 12)
            )

    def reset(self):
        self.step_count = 0
        self.attack_remaining = 0
        self.evade_remaining = 0
        self.evade_count = 0
        self.last_action = 1

    def observe_reward(self, reward: float):
        """Turn hit feedback into short attack and evasion reactions."""
        if reward < 0:
            self.evade_count += 1
            self.evade_remaining = self.evade_frames
            self.attack_remaining = 0
        elif reward > 0:
            self.attack_remaining = self.attack_lock_frames

    def get_action(self, _observation=None) -> int:
        if self.evade_remaining > 0:
            # Alternate diagonal fire directions after each incoming hit.
            action = 14 if self.evade_count % 2 else 15
            self.evade_remaining -= 1
            self.last_action = action
            self.step_count += 1
            return action
        if self.attack_remaining > 0:
            # Advance while firing instead of freezing in place after a hit.
            self.attack_remaining -= 1
            self.last_action = 10
            self.step_count += 1
            return 10

        inward_turn = 4 if self.role == AGENTS[0] else 3
        outward_turn = 3 if self.role == AGENTS[0] else 4
        phases = (
            (inward_turn, self.turn_frames),
            (2, self.vertical_frames),
            (outward_turn, self.turn_frames),
            (10, self.cross_frames),
            (outward_turn, self.turn_frames),
            (10, self.descent_frames),
        )
        elapsed = 0
        action = 1
        for phase_action, duration in phases:
            elapsed += duration
            if self.step_count < elapsed:
                action = phase_action
                break
        else:
            # Patrol instead of becoming a stationary turret after navigation.
            patrol_step = self.step_count - elapsed
            sweep_duration = self.sweep_block * len(self.sweep_actions)
            cycle_step = patrol_step % (self.advance_frames + sweep_duration)
            if cycle_step < self.advance_frames:
                action = 10
            else:
                sweep_step = cycle_step - self.advance_frames
                action = self.sweep_actions[
                    (sweep_step // self.sweep_block) % len(self.sweep_actions)
                ]
        self.step_count += 1
        self.last_action = action
        return action


class ScriptedOpponentPolicy:
    """Stateful adapter that controls whichever role opposes the learner."""

    def __init__(self):
        self.agents = {role: SpawnHunterAgent(role=role) for role in AGENTS}

    def reset(self, role: str):
        self.agents[role].reset()

    def __call__(self, observation, role: str) -> int:
        return self.agents[role].get_action(observation)

    def observe_reward(self, reward: float, role: str):
        self.agents[role].observe_reward(reward)
