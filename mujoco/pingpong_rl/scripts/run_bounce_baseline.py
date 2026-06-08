from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl.defaults import DEFAULT_BALL_HEIGHT, DEFAULT_BOUNCE_EPISODES, DEFAULT_BOUNCE_MAX_STEPS
from pingpong_rl.envs import PingPongSim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-learning bounce sanity check.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_BOUNCE_EPISODES, help="Number of reset-and-drop trials to run.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_BOUNCE_MAX_STEPS, help="Maximum physics steps per episode.")
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT, help="Spawn height above racket_center.")
    parser.add_argument(
        "--xy-offset",
        type=float,
        nargs=2,
        metavar=("DX", "DY"),
        default=(0.0, 0.0),
        help="Optional XY offset from racket_center for spawn probing.",
    )
    parser.add_argument(
        "--ball-velocity",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=(0.0, 0.0, 0.0),
        help="Initial ball velocity applied at spawn.",
    )
    return parser.parse_args()


def run_episode(sim: PingPongSim, max_steps: int) -> dict[str, object]:
    first_target_contact: str | None = None
    peak_ball_height = float(sim.ball_position[2])

    for step in range(1, max_steps + 1):
        sim.step(n_substeps=1)
        peak_ball_height = max(peak_ball_height, float(sim.ball_position[2]))

        if first_target_contact is None:
            if sim.has_contact("ball_geom", "racket_head"):
                first_target_contact = "racket_head"
            elif sim.has_contact("ball_geom", "floor"):
                first_target_contact = "floor"

        failure_reason = sim.failure_reason()
        if failure_reason is not None:
            return {
                "steps": step,
                "sim_time": float(sim.data.time),
                "first_target_contact": first_target_contact,
                "failure_reason": failure_reason,
                "peak_ball_height": peak_ball_height,
                "final_ball_position": sim.ball_position.copy(),
            }

    return {
        "steps": max_steps,
        "sim_time": float(sim.data.time),
        "first_target_contact": first_target_contact,
        "failure_reason": "max_steps",
        "peak_ball_height": peak_ball_height,
        "final_ball_position": sim.ball_position.copy(),
    }


def main() -> None:
    args = parse_args()
    sim = PingPongSim()

    racket_first_count = 0
    for episode in range(1, args.episodes + 1):
        sim.reset()
        sim.reset_ball_above_racket(
            height=args.ball_height,
            xy_offset=args.xy_offset,
            velocity=args.ball_velocity,
        )

        summary = run_episode(sim, max_steps=args.max_steps)
        if summary["first_target_contact"] == "racket_head":
            racket_first_count += 1

        final_ball_position = np.array2string(summary["final_ball_position"], precision=3, separator=", ")
        first_target_contact = summary["first_target_contact"] or "none"
        print(
            f"episode={episode} first_target_contact={first_target_contact} "
            f"failure_reason={summary['failure_reason']} steps={summary['steps']} "
            f"time={summary['sim_time']:.3f} peak_z={summary['peak_ball_height']:.3f} "
            f"final_ball={final_ball_position}"
        )

    print(f"summary racket_first={racket_first_count}/{args.episodes}")


if __name__ == "__main__":
    main()