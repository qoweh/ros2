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


def summarize_contacts(contact_rows: list[dict[str, object]]) -> dict[str, object]:
    if not contact_rows:
        return {
            "total_contacts": 0,
            "useful_contact_rate": 0.0,
            "mean_ball_lateral_speed": 0.0,
            "mean_ball_lateral_to_vertical_ratio": 0.0,
            "mean_projected_apex_xy_error": 0.0,
            "useful_contact_mean_projected_apex_xy_error": 0.0,
        }

    def float_series(key: str, rows: list[dict[str, object]]) -> np.ndarray:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return np.asarray(values, dtype=float)

    useful_rows = [row for row in contact_rows if bool(row.get("is_useful_contact", False))]
    ball_lateral_speed = float_series("ball_lateral_speed", contact_rows)
    ball_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", contact_rows)
    useful_lateral_speed = float_series("ball_lateral_speed", useful_rows)
    useful_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", useful_rows)
    projected_apex_xy_error = float_series("projected_apex_xy_error", contact_rows)
    useful_projected_apex_xy_error = float_series("projected_apex_xy_error", useful_rows)
    return {
        "total_contacts": len(contact_rows),
        "useful_contact_rate": len(useful_rows) / len(contact_rows),
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
                    controller_anchor_position = info.get("controller_anchor_position")
                    controller_anchor_xy = None
                    if controller_anchor_position is not None:
                        controller_anchor_xy = np.asarray(controller_anchor_position, dtype=float)[:2]
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
                    if ball_velocity_x is not None and ball_velocity_y is not None:
                        ball_lateral_speed = math.hypot(float(ball_velocity_x), float(ball_velocity_y))
                    if ball_lateral_speed is not None and ball_velocity_z is not None:
                        ball_lateral_to_vertical_ratio = ball_lateral_speed / max(abs(float(ball_velocity_z)), 1.0e-6)
                    if (
                        controller_anchor_xy is not None
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
                            np.linalg.norm(np.array([projected_apex_x, projected_apex_y], dtype=float) - controller_anchor_xy)
                        )
                    contact_rows.append(
                        {
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
                                None if controller_anchor_xy is None else float(controller_anchor_xy[0])
                            ),
                            "controller_anchor_y": (
                                None if controller_anchor_xy is None else float(controller_anchor_xy[1])
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
        "failure_counts": dict(failure_counts),
        "contact_summary": summarize_contacts(contact_rows),
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