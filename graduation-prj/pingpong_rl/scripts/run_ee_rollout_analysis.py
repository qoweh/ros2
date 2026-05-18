from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl.defaults import (
    DEFAULT_ANALYSIS_EPISODES,
    DEFAULT_ANALYSIS_PREFIX,
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
)
from pingpong_rl.envs import PingPongEEDeltaEnv
from pingpong_rl.utils import ROLLOUT_ANALYSIS_ROOT, resolve_output_path


EPISODE_FIELDS: tuple[str, ...] = (
    "episode_index",
    "terminated",
    "truncated",
    "success_reason",
    "episode_success_reason",
    "failure_reason",
    "time_limit_reached",
    "episode_steps",
    "contact_count",
    "successful_bounce_count",
    "first_contact_step",
    "reward_total_sum",
    "reward_height_sum",
    "reward_distance_sum",
    "reward_contact_sum",
    "reward_success_sum",
    "reward_failure_sum",
)

STEP_FIELDS: tuple[str, ...] = (
    "episode_index",
    "step",
    "reward_total",
    "reward_height",
    "reward_distance",
    "reward_contact",
    "reward_success",
    "reward_failure",
    "terminated",
    "truncated",
    "time_limit_reached",
    "success_reason",
    "episode_success_reason",
    "failure_reason",
    "successful_bounce_count",
    "racket_contact",
    "contact_observed_during_step",
    "contact_event_during_step",
    "contact_substep",
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    "ball_speed_norm",
    "contact_ball_velocity_x",
    "contact_ball_velocity_y",
    "contact_ball_velocity_z",
    "contact_ball_speed_norm",
)

CONTACT_FIELDS: tuple[str, ...] = (
    "episode_index",
    "contact_step",
    "contact_substep",
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    "ball_speed_norm",
    "success_reason",
    "failure_reason",
    "terminated",
    "truncated",
    "time_limit_reached",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EE env rollouts and export reward/contact analysis logs.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_ANALYSIS_EPISODES, help="Number of episodes to roll out.")
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Time-limit steps passed to PingPongEEDeltaEnv.",
    )
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT, help="Spawn height above racket_center.")
    parser.add_argument(
        "--ball-velocity",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=(0.0, 0.0, 0.0),
        help="Initial ball velocity passed to env.reset().",
    )
    parser.add_argument(
        "--action",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="Constant action applied at every env step for this analysis run.",
    )
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
        help="Success threshold forwarded to PingPongEEDeltaEnv. This script does not modify it automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/etc/rollout_analysis"),
        help="Directory for CSV/JSON exports.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_ANALYSIS_PREFIX,
        help="Prefix used for exported filenames.",
    )
    return parser.parse_args()


def _normalize_reason(value: object) -> str:
    return "" if value is None else str(value)


def _quantile_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "p50": None, "p75": None, "p90": None, "max": None}

    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _episode_metric_stats(rows: Sequence[dict[str, object]], field_name: str) -> dict[str, float | None]:
    return _quantile_stats([float(row[field_name]) for row in rows])


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_episode(
    env: PingPongEEDeltaEnv,
    episode_index: int,
    action: Sequence[float],
    ball_height: float,
    ball_velocity: Sequence[float],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    env.reset(ball_height=ball_height, ball_velocity=ball_velocity)

    reward_sums = {
        "reward_total_sum": 0.0,
        "reward_height_sum": 0.0,
        "reward_distance_sum": 0.0,
        "reward_contact_sum": 0.0,
        "reward_success_sum": 0.0,
        "reward_failure_sum": 0.0,
    }
    step_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    first_contact_step: int | None = None

    while True:
        _, _, terminated, truncated, info = env.step(action)
        reward_sums["reward_total_sum"] += float(info["reward_total"])
        reward_sums["reward_height_sum"] += float(info["reward_height"])
        reward_sums["reward_distance_sum"] += float(info["reward_distance"])
        reward_sums["reward_contact_sum"] += float(info["reward_contact"])
        reward_sums["reward_success_sum"] += float(info["reward_success"])
        reward_sums["reward_failure_sum"] += float(info["reward_failure"])

        step_rows.append(
            {
                "episode_index": episode_index,
                "step": int(info["episode_steps"]),
                "reward_total": float(info["reward_total"]),
                "reward_height": float(info["reward_height"]),
                "reward_distance": float(info["reward_distance"]),
                "reward_contact": float(info["reward_contact"]),
                "reward_success": float(info["reward_success"]),
                "reward_failure": float(info["reward_failure"]),
                "terminated": bool(info["terminated"]),
                "truncated": bool(info["truncated"]),
                "time_limit_reached": bool(info["time_limit_reached"]),
                "success_reason": _normalize_reason(info["success_reason"]),
                "episode_success_reason": _normalize_reason(info.get("episode_success_reason")),
                "failure_reason": _normalize_reason(info["failure_reason"]),
                "successful_bounce_count": int(info.get("successful_bounce_count", 0)),
                "racket_contact": bool(info["racket_contact"]),
                "contact_observed_during_step": bool(info["contact_observed_during_step"]),
                "contact_event_during_step": bool(info.get("contact_event_during_step", False)),
                "contact_substep": info["contact_substep"],
                "ball_velocity_x": float(info["ball_velocity_x"]),
                "ball_velocity_y": float(info["ball_velocity_y"]),
                "ball_velocity_z": float(info["ball_velocity_z"]),
                "ball_speed_norm": float(info["ball_speed_norm"]),
                "contact_ball_velocity_x": info["contact_ball_velocity_x"],
                "contact_ball_velocity_y": info["contact_ball_velocity_y"],
                "contact_ball_velocity_z": info["contact_ball_velocity_z"],
                "contact_ball_speed_norm": info["contact_ball_speed_norm"],
            }
        )

        if info.get("contact_event_during_step", False):
            if first_contact_step is None:
                first_contact_step = int(info["episode_steps"])
            contact_rows.append(
                {
                    "episode_index": episode_index,
                    "contact_step": int(info["episode_steps"]),
                    "contact_substep": info["contact_substep"],
                    "ball_velocity_x": info["contact_ball_velocity_x"],
                    "ball_velocity_y": info["contact_ball_velocity_y"],
                    "ball_velocity_z": info["contact_ball_velocity_z"],
                    "ball_speed_norm": info["contact_ball_speed_norm"],
                    "success_reason": _normalize_reason(info["success_reason"]),
                    "failure_reason": _normalize_reason(info["failure_reason"]),
                    "terminated": bool(info["terminated"]),
                    "truncated": bool(info["truncated"]),
                    "time_limit_reached": bool(info["time_limit_reached"]),
                }
            )

        if terminated or truncated:
            return (
                {
                    "episode_index": episode_index,
                    "terminated": terminated,
                    "truncated": truncated,
                    "success_reason": _normalize_reason(info["success_reason"]),
                    "episode_success_reason": _normalize_reason(info.get("episode_success_reason")),
                    "failure_reason": _normalize_reason(info["failure_reason"]),
                    "time_limit_reached": bool(info["time_limit_reached"]),
                    "episode_steps": int(info["episode_steps"]),
                    "contact_count": len(contact_rows),
                    "successful_bounce_count": int(info.get("successful_bounce_count", 0)),
                    "first_contact_step": first_contact_step if first_contact_step is not None else "",
                    **reward_sums,
                },
                step_rows,
                contact_rows,
            )


def build_summary(
    episode_rows: Sequence[dict[str, object]],
    contact_rows: Sequence[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    success_counter = Counter(str(row["episode_success_reason"]) for row in episode_rows if row["episode_success_reason"])
    failure_counter = Counter(str(row["failure_reason"]) for row in episode_rows if row["failure_reason"])
    contact_velocity_x = [float(row["ball_velocity_x"]) for row in contact_rows if row["ball_velocity_x"] is not None]
    contact_velocity_y = [float(row["ball_velocity_y"]) for row in contact_rows if row["ball_velocity_y"] is not None]
    contact_velocity_z = [float(row["ball_velocity_z"]) for row in contact_rows if row["ball_velocity_z"] is not None]
    contact_speed_norm = [float(row["ball_speed_norm"]) for row in contact_rows if row["ball_speed_norm"] is not None]
    zero_contact_reward_episodes = sum(float(row["reward_contact_sum"]) == 0.0 for row in episode_rows)
    zero_success_reward_episodes = sum(float(row["reward_success_sum"]) == 0.0 for row in episode_rows)
    height_dominant_episodes = sum(
        float(row["reward_height_sum"]) > float(row["reward_contact_sum"]) + float(row["reward_success_sum"])
        for row in episode_rows
    )
    survival_without_success_episodes = sum(
        (not bool(row["truncated"])) and row["episode_success_reason"] == "" and float(row["reward_height_sum"]) > 0.0
        for row in episode_rows
    )

    return {
        "config": {
            "episodes": int(args.episodes),
            "max_episode_steps": int(args.max_episode_steps),
            "ball_height": float(args.ball_height),
            "ball_velocity": [float(value) for value in args.ball_velocity],
            "action": [float(value) for value in args.action],
            "success_velocity_threshold": float(args.success_velocity_threshold),
        },
        "episode_counts": {
            "episodes": len(episode_rows),
            "terminated": sum(bool(row["terminated"]) for row in episode_rows),
            "truncated": sum(bool(row["truncated"]) for row in episode_rows),
            "successes": int(sum(success_counter.values())),
            "failures": int(sum(failure_counter.values())),
            "contacts": len(contact_rows),
            "episodes_with_bounces": sum(int(row["successful_bounce_count"]) > 0 for row in episode_rows),
        },
        "success_reason_counts": dict(success_counter),
        "failure_reason_counts": dict(failure_counter),
        "bounce_count_stats": _episode_metric_stats(episode_rows, "successful_bounce_count"),
        "reward_sum_stats": {
            "reward_total_sum": _episode_metric_stats(episode_rows, "reward_total_sum"),
            "reward_height_sum": _episode_metric_stats(episode_rows, "reward_height_sum"),
            "reward_distance_sum": _episode_metric_stats(episode_rows, "reward_distance_sum"),
            "reward_contact_sum": _episode_metric_stats(episode_rows, "reward_contact_sum"),
            "reward_success_sum": _episode_metric_stats(episode_rows, "reward_success_sum"),
            "reward_failure_sum": _episode_metric_stats(episode_rows, "reward_failure_sum"),
        },
        "reward_dominance": {
            "zero_contact_reward_episodes": zero_contact_reward_episodes,
            "zero_success_reward_episodes": zero_success_reward_episodes,
            "height_dominant_episodes": height_dominant_episodes,
            "survival_without_success_episodes": survival_without_success_episodes,
        },
        "contact_velocity_stats": {
            "ball_velocity_x": _quantile_stats(contact_velocity_x),
            "ball_velocity_y": _quantile_stats(contact_velocity_y),
            "ball_velocity_z": _quantile_stats(contact_velocity_z),
            "ball_speed_norm": _quantile_stats(contact_speed_norm),
        },
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError(f"episodes must be positive, got {args.episodes}.")

    action = np.asarray(args.action, dtype=float)
    if action.shape != (3,):
        raise ValueError(f"Action must have shape (3,), got {action.shape}.")

    env = PingPongEEDeltaEnv(
        max_episode_steps=args.max_episode_steps,
        ball_height=args.ball_height,
        success_velocity_threshold=args.success_velocity_threshold,
    )
    output_dir = (
        ROLLOUT_ANALYSIS_ROOT
        if args.output_dir == Path("docs/etc/rollout_analysis")
        else resolve_output_path(args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    for episode_index in range(1, args.episodes + 1):
        episode_row, episode_step_rows, episode_contact_rows = run_episode(
            env=env,
            episode_index=episode_index,
            action=action,
            ball_height=args.ball_height,
            ball_velocity=args.ball_velocity,
        )
        episode_rows.append(episode_row)
        step_rows.extend(episode_step_rows)
        contact_rows.extend(episode_contact_rows)

    summary = build_summary(episode_rows, contact_rows, args)
    episode_path = output_dir / f"{args.output_prefix}_episodes.csv"
    step_path = output_dir / f"{args.output_prefix}_steps.csv"
    contact_path = output_dir / f"{args.output_prefix}_contacts.csv"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"

    _write_csv(episode_path, episode_rows, EPISODE_FIELDS)
    _write_csv(step_path, step_rows, STEP_FIELDS)
    _write_csv(contact_path, contact_rows, CONTACT_FIELDS)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    vz_stats = summary["contact_velocity_stats"]["ball_velocity_z"]
    print(
        f"episodes={summary['episode_counts']['episodes']} "
        f"successes={summary['episode_counts']['successes']} "
        f"failures={summary['episode_counts']['failures']} "
        f"truncated={summary['episode_counts']['truncated']} "
        f"contacts={summary['episode_counts']['contacts']}"
    )
    print(
        "contact_velocity_z_stats "
        f"p50={vz_stats['p50']} p75={vz_stats['p75']} p90={vz_stats['p90']} max={vz_stats['max']}"
    )
    print(f"wrote {episode_path}")
    print(f"wrote {step_path}")
    print(f"wrote {contact_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
