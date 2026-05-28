from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.utils import infer_run_name_from_model_path, resolve_env_kwargs_for_model, resolve_requested_run_name, resolve_saved_model_path

_APEX_TARGET_CHOICES = (
    "controller_anchor",
    "racket_home",
    "racket_position",
    "target_position",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rebound direction/contact quality for a saved pingpong_rl2 PPO policy.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-version", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--ball-height", type=float, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--reset-xy-range", type=float, default=None)
    parser.add_argument("--reset-velocity-xy-range", type=float, default=None)
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--success-velocity-threshold", type=float, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--analysis-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--apex-target",
        type=str,
        default="controller_anchor",
        choices=_APEX_TARGET_CHOICES,
        help="Which XY target to use for the primary projected_apex_xy_error metric.",
    )
    parser.add_argument(
        "--compare-apex-targets",
        action="store_true",
        help="Also summarize projected_apex_xy_error against every supported XY target candidate.",
    )
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


def summarize_contacts(
    contact_rows: list[dict[str, object]],
    *,
    selected_apex_target: str,
    compare_apex_targets: bool,
) -> dict[str, object]:
    selected_error_key = f"projected_apex_xy_error_{selected_apex_target}"
    if not contact_rows:
        summary = {
            "total_contacts": 0,
            "useful_contact_rate": 0.0,
            "mean_ball_lateral_speed": 0.0,
            "mean_ball_lateral_to_vertical_ratio": 0.0,
            "mean_projected_apex_xy_error": 0.0,
            "useful_contact_mean_projected_apex_xy_error": 0.0,
            "selected_apex_target": selected_apex_target,
        }
        if compare_apex_targets:
            summary["apex_target_metrics"] = {
                target_name: {
                    "mean_projected_apex_xy_error": 0.0,
                    "useful_contact_mean_projected_apex_xy_error": 0.0,
                }
                for target_name in _APEX_TARGET_CHOICES
            }
        return summary

    def float_series(key: str, rows: list[dict[str, object]]) -> np.ndarray:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return np.asarray(values, dtype=float)

    useful_rows = [row for row in contact_rows if bool(row.get("is_useful_contact", False))]
    ball_lateral_speed = float_series("ball_lateral_speed", contact_rows)
    ball_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", contact_rows)
    useful_lateral_speed = float_series("ball_lateral_speed", useful_rows)
    useful_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", useful_rows)
    projected_apex_xy_error = float_series(selected_error_key, contact_rows)
    useful_projected_apex_xy_error = float_series(selected_error_key, useful_rows)
    summary = {
        "total_contacts": len(contact_rows),
        "useful_contact_rate": len(useful_rows) / len(contact_rows),
        "selected_apex_target": selected_apex_target,
        "mean_ball_lateral_speed": float(ball_lateral_speed.mean()) if ball_lateral_speed.size else 0.0,
        "mean_ball_lateral_to_vertical_ratio": (
            float(ball_lateral_ratio.mean()) if ball_lateral_ratio.size else 0.0
        ),
        "useful_contact_mean_ball_lateral_speed": (
            float(useful_lateral_speed.mean()) if useful_lateral_speed.size else 0.0
        ),
        "useful_contact_mean_ball_lateral_to_vertical_ratio": (
            float(useful_lateral_ratio.mean()) if useful_lateral_ratio.size else 0.0
        ),
        "mean_projected_apex_xy_error": (
            float(projected_apex_xy_error.mean()) if projected_apex_xy_error.size else 0.0
        ),
        "useful_contact_mean_projected_apex_xy_error": (
            float(useful_projected_apex_xy_error.mean()) if useful_projected_apex_xy_error.size else 0.0
        ),
    }
    if compare_apex_targets:
        apex_target_metrics: dict[str, dict[str, float]] = {}
        for target_name in _APEX_TARGET_CHOICES:
            target_error_key = f"projected_apex_xy_error_{target_name}"
            target_error = float_series(target_error_key, contact_rows)
            useful_target_error = float_series(target_error_key, useful_rows)
            apex_target_metrics[target_name] = {
                "mean_projected_apex_xy_error": float(target_error.mean()) if target_error.size else 0.0,
                "useful_contact_mean_projected_apex_xy_error": (
                    float(useful_target_error.mean()) if useful_target_error.size else 0.0
                ),
            }
        summary["apex_target_metrics"] = apex_target_metrics
    return summary


def summarize_episode_apex_targets(
    episode_rows: list[dict[str, object]],
    contact_rows: list[dict[str, object]],
    *,
    compare_apex_targets: bool,
) -> dict[str, object]:
    episode_useful_bounces: dict[int, int] = {
        int(row["episode"]): int(row.get("useful_bounces", 0))
        for row in episode_rows
    }
    first_contact_by_episode: dict[int, dict[str, object]] = {}
    for row in contact_rows:
        episode = int(row["episode"])
        if episode not in first_contact_by_episode:
            first_contact_by_episode[episode] = row

    summary: dict[str, object] = {
        "episodes_with_one_or_more_useful_bounces": sum(value >= 1 for value in episode_useful_bounces.values()),
        "episodes_with_two_or_more_useful_bounces": sum(value >= 2 for value in episode_useful_bounces.values()),
    }
    if not compare_apex_targets:
        return summary

    target_metrics: dict[str, dict[str, object]] = {}
    for target_name in _APEX_TARGET_CHOICES:
        error_key = f"projected_apex_xy_error_{target_name}"
        first_contact_two_or_more: list[float] = []
        first_contact_fewer_than_two: list[float] = []
        first_contact_one_or_more: list[float] = []
        first_contact_zero: list[float] = []
        for episode, useful_bounce_count in episode_useful_bounces.items():
            first_contact = first_contact_by_episode.get(episode)
            if first_contact is None or first_contact.get(error_key) is None:
                continue
            error_value = float(first_contact[error_key])
            if useful_bounce_count >= 2:
                first_contact_two_or_more.append(error_value)
            else:
                first_contact_fewer_than_two.append(error_value)
            if useful_bounce_count >= 1:
                first_contact_one_or_more.append(error_value)
            else:
                first_contact_zero.append(error_value)

        mean_two_or_more = (
            float(np.mean(first_contact_two_or_more)) if first_contact_two_or_more else None
        )
        mean_fewer_than_two = (
            float(np.mean(first_contact_fewer_than_two)) if first_contact_fewer_than_two else None
        )
        mean_one_or_more = (
            float(np.mean(first_contact_one_or_more)) if first_contact_one_or_more else None
        )
        mean_zero = float(np.mean(first_contact_zero)) if first_contact_zero else None
        target_metrics[target_name] = {
            "episodes_with_two_or_more_useful_bounces_count": len(first_contact_two_or_more),
            "first_contact_mean_error_two_or_more_useful_bounces": mean_two_or_more,
            "episodes_with_fewer_than_two_useful_bounces_count": len(first_contact_fewer_than_two),
            "first_contact_mean_error_fewer_than_two_useful_bounces": mean_fewer_than_two,
            "two_or_more_useful_bounces_gap": (
                None
                if mean_two_or_more is None or mean_fewer_than_two is None
                else mean_two_or_more - mean_fewer_than_two
            ),
            "episodes_with_one_or_more_useful_bounces_count": len(first_contact_one_or_more),
            "first_contact_mean_error_one_or_more_useful_bounces": mean_one_or_more,
            "episodes_with_zero_useful_bounces_count": len(first_contact_zero),
            "first_contact_mean_error_zero_useful_bounces": mean_zero,
            "one_or_more_useful_bounces_gap": (
                None
                if mean_one_or_more is None or mean_zero is None
                else mean_one_or_more - mean_zero
            ),
        }

    summary["apex_target_episode_metrics"] = target_metrics
    return summary


def apex_target_xy_candidates(
    *,
    info: dict[str, object],
    racket_home_xy: np.ndarray,
    racket_position_xy: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {
        "racket_home": np.asarray(racket_home_xy, dtype=float)[:2],
        "racket_position": np.asarray(racket_position_xy, dtype=float)[:2],
    }
    controller_anchor_position = info.get("controller_anchor_position")
    if controller_anchor_position is not None:
        candidates["controller_anchor"] = np.asarray(controller_anchor_position, dtype=float)[:2]
    target_position = info.get("target_position")
    if target_position is not None:
        candidates["target_position"] = np.asarray(target_position, dtype=float)[:2]
    return candidates


def main() -> None:
    args = parse_args()
    resolved_run_name = None if args.run_name is None else resolve_requested_run_name(args.run_name, args.run_version)
    model_path = resolve_saved_model_path(args.model_path, resolved_run_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved PPO model not found: {model_path}")

    env_kwargs = resolve_env_kwargs_for_model(
        model_path,
        ball_height=args.ball_height,
        max_episode_steps=args.max_episode_steps,
        reset_xy_range=args.reset_xy_range,
        reset_velocity_xy_range=args.reset_velocity_xy_range,
        reset_velocity_z_range=args.reset_velocity_z_range,
        success_velocity_threshold=args.success_velocity_threshold,
    )
    env = PingPongKeepUpGymEnv(**env_kwargs)
    model = PPO.load(str(model_path))
    run_name = infer_run_name_from_model_path(model_path)
    gravity_magnitude = max(abs(float(env.base_env.sim.model.opt.gravity[2])), 1.0e-6)
    output_dir = (model_path.parent / "analysis") if args.output_dir is None else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_name = args.analysis_name or f"{run_name}_rebound_{args.episodes}ep"

    episode_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    returns: list[float] = []
    useful_bounces: list[int] = []
    failure_counts: Counter[str] = Counter()

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode - 1)
            racket_home_xy = np.asarray(env.base_env.sim.racket_position[:2], dtype=float)
            episode_return = 0.0
            step_count = 0
            contact_count = 0
            first_contact_step: int | None = None
            info: dict[str, object] = {}
            while True:
                action, _ = model.predict(observation, deterministic=not args.stochastic)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                step_count += 1

                if bool(info.get("contact_event_during_step", False)):
                    contact_count += 1
                    if first_contact_step is None:
                        first_contact_step = step_count
                    racket_position_xy = np.asarray(env.base_env.sim.racket_position[:2], dtype=float)
                    apex_targets = apex_target_xy_candidates(
                        info=info,
                        racket_home_xy=racket_home_xy,
                        racket_position_xy=racket_position_xy,
                    )
                    contact_ball_position_x = info.get("contact_ball_position_x")
                    contact_ball_position_y = info.get("contact_ball_position_y")
                    ball_velocity_x = info.get("contact_ball_velocity_x")
                    ball_velocity_y = info.get("contact_ball_velocity_y")
                    ball_velocity_z = info.get("contact_ball_velocity_z")
                    ball_lateral_speed = None
                    ball_lateral_to_vertical_ratio = None
                    projected_apex_time = None
                    projected_apex_x = None
                    projected_apex_y = None
                    projected_apex_xy_error = None
                    selected_apex_target_xy = apex_targets.get(args.apex_target)
                    if ball_velocity_x is not None and ball_velocity_y is not None:
                        ball_lateral_speed = math.hypot(float(ball_velocity_x), float(ball_velocity_y))
                    if ball_lateral_speed is not None and ball_velocity_z is not None:
                        ball_lateral_to_vertical_ratio = ball_lateral_speed / max(abs(float(ball_velocity_z)), 1.0e-6)
                    if (
                        selected_apex_target_xy is not None
                        and contact_ball_position_x is not None
                        and contact_ball_position_y is not None
                        and ball_velocity_x is not None
                        and ball_velocity_y is not None
                        and ball_velocity_z is not None
                    ):
                        projected_apex_time = max(float(ball_velocity_z), 0.0) / gravity_magnitude
                        projected_apex_x = float(contact_ball_position_x) + float(ball_velocity_x) * projected_apex_time
                        projected_apex_y = float(contact_ball_position_y) + float(ball_velocity_y) * projected_apex_time
                        projected_apex_xy_error = float(
                            np.linalg.norm(
                                np.array([projected_apex_x, projected_apex_y], dtype=float) - selected_apex_target_xy
                            )
                        )
                    contact_row = {
                        "episode": episode,
                        "step": step_count,
                        "contact_index": contact_count,
                        "success_reason": info.get("success_reason"),
                        "is_useful_contact": info.get("success_reason") == "useful_keepup_bounce",
                        "contact_ball_position_x": contact_ball_position_x,
                        "contact_ball_position_y": contact_ball_position_y,
                        "ball_velocity_x": ball_velocity_x,
                        "ball_velocity_y": ball_velocity_y,
                        "ball_velocity_z": ball_velocity_z,
                        "ball_speed_norm": info.get("contact_ball_speed_norm"),
                        "ball_lateral_speed": ball_lateral_speed,
                        "ball_lateral_to_vertical_ratio": ball_lateral_to_vertical_ratio,
                        "controller_anchor_x": (
                            None if "controller_anchor" not in apex_targets else float(apex_targets["controller_anchor"][0])
                        ),
                        "controller_anchor_y": (
                            None if "controller_anchor" not in apex_targets else float(apex_targets["controller_anchor"][1])
                        ),
                        "racket_home_x": float(apex_targets["racket_home"][0]),
                        "racket_home_y": float(apex_targets["racket_home"][1]),
                        "racket_position_x": float(apex_targets["racket_position"][0]),
                        "racket_position_y": float(apex_targets["racket_position"][1]),
                        "target_position_x": (
                            None if "target_position" not in apex_targets else float(apex_targets["target_position"][0])
                        ),
                        "target_position_y": (
                            None if "target_position" not in apex_targets else float(apex_targets["target_position"][1])
                        ),
                        "apex_target_name": args.apex_target,
                        "apex_target_x": (
                            None if selected_apex_target_xy is None else float(selected_apex_target_xy[0])
                        ),
                        "apex_target_y": (
                            None if selected_apex_target_xy is None else float(selected_apex_target_xy[1])
                        ),
                        "projected_apex_time": projected_apex_time,
                        "projected_apex_x": projected_apex_x,
                        "projected_apex_y": projected_apex_y,
                        "projected_apex_xy_error": projected_apex_xy_error,
                        "racket_velocity_x": info.get("contact_racket_velocity_x"),
                        "racket_velocity_y": info.get("contact_racket_velocity_y"),
                        "racket_velocity_z": info.get("contact_racket_velocity_z"),
                        "racket_speed_norm": info.get("contact_racket_speed_norm"),
                        "xy_alignment_error": info.get("contact_xy_alignment_error"),
                        "contact_ball_height_above_racket": info.get("contact_ball_height_above_racket"),
                        "projected_contact_apex_height_above_racket": info.get(
                            "projected_contact_apex_height_above_racket"
                        ),
                        "target_tilt_0": (
                            None if info.get("target_tilt") is None else float(np.asarray(info["target_tilt"])[0])
                        ),
                        "target_tilt_1": (
                            None if info.get("target_tilt") is None else float(np.asarray(info["target_tilt"])[1])
                        ),
                    }
                    for target_name, target_xy in apex_targets.items():
                        contact_row[f"projected_apex_xy_error_{target_name}"] = None
                        if projected_apex_x is None or projected_apex_y is None:
                            continue
                        contact_row[f"projected_apex_xy_error_{target_name}"] = float(
                            np.linalg.norm(np.array([projected_apex_x, projected_apex_y], dtype=float) - target_xy)
                        )
                    contact_rows.append(
                        contact_row
                    )

                if terminated or truncated:
                    break

            useful_bounce_count = int(info.get("successful_bounce_count", 0))
            failure_reason = info.get("failure_reason")
            if failure_reason is None:
                failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
            failure_counts[str(failure_reason)] += 1
            returns.append(episode_return)
            useful_bounces.append(useful_bounce_count)
            episode_rows.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "steps": step_count,
                    "contact_count": contact_count,
                    "first_contact_step": first_contact_step,
                    "useful_bounces": useful_bounce_count,
                    "failure_reason": failure_reason,
                }
            )
            print(
                f"episode={episode} steps={step_count} contacts={contact_count} return={episode_return:.3f} "
                f"useful_bounces={useful_bounce_count} failure_reason={failure_reason}"
            )
    finally:
        env.close()

    returns_array = np.asarray(returns, dtype=float)
    bounce_array = np.asarray(useful_bounces, dtype=float)
    summary = {
        "model_path": str(model_path.resolve()),
        "run_name": run_name,
        "episodes": args.episodes,
        "env_config": env.training_config() if False else env_kwargs,
        "mean_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "episodes_with_one_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 1.0)) if bounce_array.size else 0,
        "one_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 1.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_two_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 2.0)) if bounce_array.size else 0,
        "two_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 2.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "contact_summary": summarize_contacts(
            contact_rows,
            selected_apex_target=args.apex_target,
            compare_apex_targets=args.compare_apex_targets,
        ),
        "episode_apex_summary": summarize_episode_apex_targets(
            episode_rows,
            contact_rows,
            compare_apex_targets=args.compare_apex_targets,
        ),
        "output_files": {
            "episodes_csv": str((output_dir / f"{analysis_name}_episodes.csv").resolve()),
            "contacts_csv": str((output_dir / f"{analysis_name}_contacts.csv").resolve()),
            "summary_json": str((output_dir / f"{analysis_name}_summary.json").resolve()),
        },
    }

    episodes_csv_path = output_dir / f"{analysis_name}_episodes.csv"
    contacts_csv_path = output_dir / f"{analysis_name}_contacts.csv"
    summary_json_path = output_dir / f"{analysis_name}_summary.json"
    write_csv(episodes_csv_path, episode_rows)
    write_csv(contacts_csv_path, contact_rows)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"analysis_dir={output_dir}")
    print(f"episodes_csv={episodes_csv_path}")
    print(f"contacts_csv={contacts_csv_path}")
    print(f"summary_json={summary_json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()