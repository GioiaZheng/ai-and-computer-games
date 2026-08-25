"""Validate closed-loop top and bottom maze coverage teachers."""

from __future__ import annotations

import argparse
from collections import Counter

from .environment import (
    ACTION_SETS,
    AGENTS,
    SingleAgentCombatEnv,
    tank_center,
    tank_heading,
)
from .scripted_agent import WaypointCoverageAgent


def run(
    role: str,
    route: str,
    seed: int,
    trace_steps: int = 0,
    max_steps: int | None = None,
):
    env = SingleAgentCombatEnv(
        fixed_role=role,
        learner_actions=ACTION_SETS["all"],
        reward_shaping=True,
        shaping_scale=0.0,
        seed=seed,
    )
    observation, _ = env.reset(seed=seed)
    teacher = WaypointCoverageAgent(role=role, route=route)
    start_center = tank_center(env.raw_observations[role], role)
    final_center = start_center
    action_counts = Counter()
    cells = set()
    idle = 0
    steps = 0
    score = 0.0
    terminated = truncated = False
    while not (terminated or truncated) and (
        max_steps is None or steps < max_steps
    ):
        if steps < trace_steps:
            raw_observation = env.raw_observations[role]
            center = tank_center(raw_observation, role)
            heading = tank_heading(raw_observation, role)
            target = teacher._waypoints()[teacher.waypoint_index]
            print(
                f"trace role={role} route={route} step={steps} "
                f"center={center} heading={heading} target={target}"
            )
        action = teacher.get_action(env.raw_observations[role])
        action_counts[action] += 1
        observation, reward, terminated, truncated, info = env.step_game_action(action)
        final_center = tank_center(env.raw_observations[role], role)
        if "position_cell" in info:
            cells.add(tuple(info["position_cell"]))
        idle += int(info.get("idle_steps", 0) > 0)
        score += float(info.get("official_reward", reward))
        steps += 1
    env.close()
    return (
        len(cells),
        idle / max(steps, 1),
        score,
        teacher.waypoints_reached,
        start_center,
        final_center,
        action_counts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=200831)
    parser.add_argument("--minimum-waypoints", type=int, default=0)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    failed = []
    for role in AGENTS:
        for route in ("top", "bottom"):
            cells, idle, score, waypoints, start, final, actions = run(
                role, route, args.seed, args.trace_steps, args.max_steps
            )
            print(
                f"{role} route={route}: cells={cells} idle={idle:.1%} "
                f"official_score={score:+.0f} waypoints_reached={waypoints} "
                f"center={start}->{final} actions={dict(sorted(actions.items()))}"
            )
            if waypoints < args.minimum_waypoints:
                failed.append(f"{role}/{route}={waypoints}")
    if failed:
        raise SystemExit(
            "Waypoint benchmark failed: "
            + ", ".join(failed)
            + f"; required at least {args.minimum_waypoints}"
        )


if __name__ == "__main__":
    main()
