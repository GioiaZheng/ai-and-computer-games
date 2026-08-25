"""Measure Combat Tank movement produced by short action sequences."""

from __future__ import annotations

import argparse

import numpy as np

from .environment import AGENTS, TANK_COLORS, create_environment, tank_center


def run_sequence(role: str, seed: int, sequence: tuple[tuple[int, int], ...]):
    env = create_environment()
    observations, _ = env.reset(seed=seed)
    opponent = AGENTS[1] if role == AGENTS[0] else AGENTS[0]
    start = tank_center(observations[role], role)
    last = start
    for action, frames in sequence:
        for _ in range(frames):
            observations, _, terminations, truncations, _ = env.step(
                {role: action, opponent: 0}
            )
            if role in observations:
                last = tank_center(observations[role], role)
            if terminations.get(role, False) or truncations.get(role, False):
                break
    env.close()
    return start, last


def trace_sequence(role: str, seed: int, sequence: tuple[tuple[int, int], ...]):
    """Return the tank center after every phase of a control sequence."""
    env = create_environment()
    observations, _ = env.reset(seed=seed)
    opponent = AGENTS[1] if role == AGENTS[0] else AGENTS[0]
    trace = [("start", tank_center(observations[role], role))]
    for action, frames in sequence:
        for _ in range(frames):
            observations, _, terminations, truncations, _ = env.step(
                {role: action, opponent: 0}
            )
            if terminations.get(role, False) or truncations.get(role, False):
                break
        trace.append((f"action={action} frames={frames}", tank_center(observations[role], role)))
    env.close()
    return trace


def pose_summary(observation, role: str) -> str:
    mask = np.all(np.asarray(observation) == TANK_COLORS[role], axis=-1)
    mask[:35] = False
    ys, xs = np.where(mask)
    center_x, center_y = np.median(xs), np.median(ys)
    return (
        f"center=({center_x:.1f},{center_y:.1f}) mean=({xs.mean():.1f},{ys.mean():.1f}) "
        f"bounds=({xs.min()},{ys.min()})..({xs.max()},{ys.max()}) pixels={len(xs)}"
    )


def inspect_turn_poses(role: str, seed: int):
    turn_action = 4 if role == AGENTS[0] else 3
    for turn_frames in (0, 16, 32, 48, 64, 80, 96):
        env = create_environment()
        observations, _ = env.reset(seed=seed)
        opponent = AGENTS[1] if role == AGENTS[0] else AGENTS[0]
        for _ in range(turn_frames):
            observations, _, _, _, _ = env.step({role: turn_action, opponent: 0})
        print(f"pose turn_frames={turn_frames:3d} {pose_summary(observations[role], role)}")
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=AGENTS, default=AGENTS[0])
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--forward-frames", type=int, default=60)
    parser.add_argument("--poses-only", action="store_true")
    args = parser.parse_args()

    if args.poses_only:
        inspect_turn_poses(args.role, args.seed)
        return

    print("turn,turn_frames,start_x,start_y,end_x,end_y,dx,dy")
    for turn_action in (3, 4):
        for turn_frames in (32, 40, 48, 56, 64, 80):
            start, end = run_sequence(
                args.role,
                args.seed,
                ((turn_action, turn_frames), (2, args.forward_frames)),
            )
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            print(
                f"{turn_action},{turn_frames},{start[0]:.1f},{start[1]:.1f},"
                f"{end[0]:.1f},{end[1]:.1f},{dx:+.1f},{dy:+.1f}"
            )

    top_route = (
        ((4, 48), (2, 420), (3, 48), (10, 1040), (3, 48), (10, 420))
        if args.role == AGENTS[0]
        else ((3, 48), (2, 420), (4, 48), (10, 1040), (4, 48), (10, 420))
    )
    print("spawn-hunter-route")
    for phase, center in trace_sequence(args.role, args.seed, top_route):
        print(f"{phase},{center[0]:.1f},{center[1]:.1f}")


if __name__ == "__main__":
    main()
