from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.controllers import HeuristicKeepUpPolicy
from pingpong_rl2.defaults import DEFAULT_BALL_HEIGHT, DEFAULT_MAX_EPISODE_STEPS
from pingpong_rl2.envs import PingPongKeepUpGymEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the scripted keep-up diagnostic baseline.")
    parser.add_argument("--analysis-name", type=str, default="heuristic_keepup_diagnostic")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--reset-xy-range", type=float, default=0.0)
    parser.add_argument("--reset-velocity-xy-range", type=float, default=0.0)
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(-0.01, 0.01),
    )
    parser.add_argument("--return-blend", type=float, default=0.72)
    parser.add_argument("--recovery-blend", type=float, default=0.52)
    parser.add_argument("--strike-z-boost", type=float, default=0.018)
    parser.add_argument("--strike-time-horizon", type=float, default=0.14)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--print-episodes", action="store_true")
    return parser.parse_args()


def write_csv(file_path: Path, rows: list[dict[str, object]]) -> None:
    field_names: list[str] = []
    for row in rows:
        for key in row:
            if key not in field_names:
                field_names.append(key)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_env_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "action_mode": "position_strike",
        "ball_height": args.ball_height,
        "target_ball_height": args.ball_height,
        "max_episode_steps": args.max_episode_steps,
        "reset_xy_range": args.reset_xy_range,
        "reset_velocity_xy_range": args.reset_velocity_xy_range,
        "reset_velocity_z_range": tuple(args.reset_velocity_z_range),
        "strike_tilt_ramp_pitch": -0.03,
        "strike_tilt_ramp_xy_tolerance": 0.04,
        "post_contact_return_assist_weight": 0.5,
        "post_contact_return_max_intercept_time": 0.6,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (ROOT / "artifacts" / "benchmarks" / args.analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PingPongKeepUpGymEnv(**build_env_kwargs(args))
    policy = HeuristicKeepUpPolicy(
        return_blend=args.return_blend,
        recovery_blend=args.recovery_blend,
        strike_z_boost=args.strike_z_boost,
        strike_time_horizon=args.strike_time_horizon,
    )

    episode_rows: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    returns: list[float] = []
    useful_bounces: list[int] = []
    reachable_contacts = 0
    reachable_useful_contacts = 0
    contact_events = 0
    useful_contact_events = 0
    easy_scores: list[float] = []
    useful_easy_scores: list[float] = []

    try:
        for episode_index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode_index)
            del observation
            policy.reset()
            episode_return = 0.0
            info: dict[str, object] = {}
            while True:
                action = policy.predict(env.base_env).astype(np.float32, copy=False)
                _, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                if bool(info.get("contact_event_during_step", False)):
                    contact_events += 1
                    if bool(info.get("next_intercept_reachable", False)):
                        reachable_contacts += 1
                    easy_score = info.get("easy_next_ball_score")
                    if easy_score is not None:
                        easy_scores.append(float(easy_score))
                    if info.get("success_reason") == "useful_keepup_bounce":
                        useful_contact_events += 1
                        if bool(info.get("next_intercept_reachable", False)):
                            reachable_useful_contacts += 1
                        if easy_score is not None:
                            useful_easy_scores.append(float(easy_score))
                if terminated or truncated:
                    break

            useful_bounce_count = int(info.get("successful_bounce_count", 0))
            failure_reason = info.get("failure_reason")
            if failure_reason is None:
                failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
            failure_counts[str(failure_reason)] += 1
            returns.append(episode_return)
            useful_bounces.append(useful_bounce_count)
            episode_row = {
                "episode": episode_index + 1,
                "return": episode_return,
                "useful_bounces": useful_bounce_count,
                "contacts": int(info.get("contact_count", 0)),
                "failure_reason": failure_reason,
                "last_phase": info.get("phase_name"),
                "last_next_intercept_reachable": info.get("next_intercept_reachable"),
                "last_easy_next_ball_score": info.get("easy_next_ball_score"),
            }
            episode_rows.append(episode_row)
            if args.print_episodes:
                print(
                    f"episode={episode_row['episode']} return={episode_row['return']:.3f} "
                    f"useful_bounces={episode_row['useful_bounces']} contacts={episode_row['contacts']} "
                    f"failure_reason={episode_row['failure_reason']}"
                )
    finally:
        env.close()

    bounce_array = np.asarray(useful_bounces, dtype=float)
    summary = {
        "analysis_name": args.analysis_name,
        "episodes": args.episodes,
        "seed": args.seed,
        "env_kwargs": build_env_kwargs(args),
        "heuristic_config": {
            "return_blend": args.return_blend,
            "recovery_blend": args.recovery_blend,
            "strike_z_boost": args.strike_z_boost,
            "strike_time_horizon": args.strike_time_horizon,
        },
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "one_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 1.0)) if bounce_array.size else 0.0,
        "two_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 2.0)) if bounce_array.size else 0.0,
        "contact_event_count": contact_events,
        "useful_contact_event_count": useful_contact_events,
        "next_intercept_reachable_rate": (reachable_contacts / contact_events) if contact_events > 0 else 0.0,
        "useful_contact_next_intercept_reachable_rate": (
            reachable_useful_contacts / useful_contact_events if useful_contact_events > 0 else 0.0
        ),
        "mean_easy_next_ball_score": float(np.mean(easy_scores)) if easy_scores else 0.0,
        "useful_contact_mean_easy_next_ball_score": float(np.mean(useful_easy_scores)) if useful_easy_scores else 0.0,
        "failure_counts": dict(failure_counts),
    }

    summary_path = output_dir / f"{args.analysis_name}_summary.json"
    episodes_path = output_dir / f"{args.analysis_name}_episodes.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(episodes_path, episode_rows)
    print(f"summary_path={summary_path}")
    print(f"episodes_path={episodes_path}")
    print(
        "heuristic_summary "
        f"mean_useful_bounces={summary['mean_useful_bounces']:.3f} "
        f"two_or_more_rate={summary['two_or_more_useful_bounce_rate']:.3f} "
        f"reachable_rate={summary['next_intercept_reachable_rate']:.3f}"
    )


if __name__ == "__main__":
    main()