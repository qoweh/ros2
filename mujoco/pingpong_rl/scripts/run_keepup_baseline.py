from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl.controllers import KeepUpHeuristicController
from pingpong_rl.defaults import (
    DEFAULT_ACTION_LIMIT,
    DEFAULT_BALL_HEIGHT,
    DEFAULT_KEEPUP_EPISODES,
    DEFAULT_KEEPUP_MAX_STEPS,
    DEFAULT_KEEPUP_PREVIEW_TIME,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
)
from pingpong_rl.envs import PingPongSim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a scripted keep-up baseline that tries to keep the ball from dropping.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_KEEPUP_EPISODES, help="Number of keep-up trials to run.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_KEEPUP_MAX_STEPS, help="Maximum control steps per episode.")
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT, help="Spawn height above racket_center.")
    parser.add_argument(
        "--ball-velocity",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=(0.0, 0.0, 0.0),
        help="Initial ball velocity applied at spawn.",
    )
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
        help="Counts a contact as an upward bounce when the traced contact z velocity exceeds this threshold.",
    )
    parser.add_argument("--preview-time", type=float, default=DEFAULT_KEEPUP_PREVIEW_TIME, help="Simple XY lead time in seconds.")
    parser.add_argument("--strike-plane-offset", type=float, default=0.02, help="Target Z offset relative to home racket pose for descending-ball interception.")
    parser.add_argument("--max-xy-offset", type=float, default=0.18, help="Clamp for XY tracking around the home racket pose.")
    parser.add_argument("--return-height-offset", type=float, default=-0.03, help="How far above home pose to return while the ball is rising.")
    parser.add_argument("--min-z-offset", type=float, default=-0.03, help="Minimum Z offset relative to home pose.")
    parser.add_argument("--max-z-offset", type=float, default=0.10, help="Maximum Z offset relative to home pose.")
    parser.add_argument("--max-intercept-time", type=float, default=0.35, help="Upper clamp for ballistic intercept-time prediction.")
    parser.add_argument("--descent-resume-velocity", type=float, default=-0.05, help="Resume tracking only after post-contact ball vertical velocity falls below this value.")
    parser.add_argument("--contact-retreat-steps", type=int, default=2, help="After a contact event, hold a recovery pose for this many control steps.")
    parser.add_argument("--position-gain", type=float, default=0.2, help="Task-space controller gain for the scripted keep-up baseline.")
    parser.add_argument("--ik-damping", type=float, default=1.0e-3, help="Damping used by the scripted keep-up Jacobian solve.")
    parser.add_argument("--action-limit", type=float, default=DEFAULT_ACTION_LIMIT, help="Max Cartesian position step passed to the Jacobian controller.")
    return parser.parse_args()


def run_episode(sim: PingPongSim, controller: KeepUpHeuristicController, args: argparse.Namespace) -> dict[str, object]:
    sim.reset(ball_height=args.ball_height, ball_velocity=args.ball_velocity)
    controller.reset()

    peak_ball_height = float(sim.ball_position[2])
    contact_count = 0
    upward_bounce_count = 0
    first_upward_bounce_step: int | None = None
    contact_active_previous_step = False

    for step in range(1, args.max_steps + 1):
        joint_targets = controller.compute_joint_targets()
        contact_trace = sim.step_with_contact_trace(joint_targets=joint_targets, n_substeps=sim.n_substeps)
        peak_ball_height = max(peak_ball_height, float(sim.ball_position[2]))

        contact_active = bool(contact_trace["contact_observed"])
        contact_event = contact_active and not contact_active_previous_step
        controller.notify_contact_event(contact_event)

        if contact_event:
            contact_count += 1
            contact_ball_velocity_z = contact_trace["contact_ball_velocity_z"]
            if contact_ball_velocity_z is not None and float(contact_ball_velocity_z) > args.success_velocity_threshold:
                upward_bounce_count += 1
                if first_upward_bounce_step is None:
                    first_upward_bounce_step = step

        contact_active_previous_step = contact_active

        failure_reason = sim.failure_reason()
        if failure_reason is not None:
            return {
                "steps": step,
                "sim_time": float(sim.data.time),
                "failure_reason": failure_reason,
                "peak_ball_height": peak_ball_height,
                "contact_count": contact_count,
                "upward_bounce_count": upward_bounce_count,
                "first_upward_bounce_step": first_upward_bounce_step,
                "final_ball_position": sim.ball_position.copy(),
            }

    return {
        "steps": args.max_steps,
        "sim_time": float(sim.data.time),
        "failure_reason": "time_limit",
        "peak_ball_height": peak_ball_height,
        "contact_count": contact_count,
        "upward_bounce_count": upward_bounce_count,
        "first_upward_bounce_step": first_upward_bounce_step,
        "final_ball_position": sim.ball_position.copy(),
    }


def main() -> None:
    args = parse_args()
    sim = PingPongSim()
    controller = KeepUpHeuristicController(
        sim,
        preview_time=args.preview_time,
        strike_plane_offset=args.strike_plane_offset,
        return_height_offset=args.return_height_offset,
        max_xy_offset=args.max_xy_offset,
        min_z_offset=args.min_z_offset,
        max_z_offset=args.max_z_offset,
        max_intercept_time=args.max_intercept_time,
        descent_resume_velocity=args.descent_resume_velocity,
        contact_hold_steps=args.contact_retreat_steps,
        position_gain=args.position_gain,
        damping=args.ik_damping,
        max_position_step=args.action_limit,
    )

    episode_summaries: list[dict[str, object]] = []
    for episode in range(1, args.episodes + 1):
        summary = run_episode(sim, controller, args)
        episode_summaries.append(summary)
        final_ball_position = np.array2string(summary["final_ball_position"], precision=3, separator=", ")
        first_upward_bounce_step = summary["first_upward_bounce_step"]
        first_upward_bounce_text = "none" if first_upward_bounce_step is None else str(first_upward_bounce_step)
        print(
            f"episode={episode} failure_reason={summary['failure_reason']} steps={summary['steps']} "
            f"time={summary['sim_time']:.3f} contacts={summary['contact_count']} "
            f"upward_bounces={summary['upward_bounce_count']} first_upward_bounce_step={first_upward_bounce_text} "
            f"peak_z={summary['peak_ball_height']:.3f} final_ball={final_ball_position}"
        )

    step_counts = np.asarray([float(summary["steps"]) for summary in episode_summaries], dtype=float)
    contact_counts = np.asarray([float(summary["contact_count"]) for summary in episode_summaries], dtype=float)
    upward_bounce_counts = np.asarray([float(summary["upward_bounce_count"]) for summary in episode_summaries], dtype=float)
    time_limit_episodes = sum(summary["failure_reason"] == "time_limit" for summary in episode_summaries)
    print(
        "summary "
        f"mean_steps={step_counts.mean():.1f} max_steps={int(step_counts.max())} "
        f"mean_contacts={contact_counts.mean():.2f} mean_upward_bounces={upward_bounce_counts.mean():.2f} "
        f"time_limit_episodes={time_limit_episodes}/{args.episodes}"
    )


if __name__ == "__main__":
    main()