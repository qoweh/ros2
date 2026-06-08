from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_heuristic_keepup_diagnostic import write_csv

from pingpong_rl2.controllers import HeuristicKeepUpPolicy
from pingpong_rl2.defaults import DEFAULT_BALL_HEIGHT, DEFAULT_MAX_EPISODE_STEPS
from pingpong_rl2.envs import PingPongKeepUpGymEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep heuristic contact-control candidates before PPO.")
    parser.add_argument("--analysis-name", type=str, default="contact_feasibility_map_v1")
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
    parser.add_argument(
        "--pitch-values",
        type=float,
        nargs="+",
        default=(-0.06, -0.03, 0.0, 0.03),
    )
    parser.add_argument(
        "--roll-values",
        type=float,
        nargs="+",
        default=(-0.03, 0.0, 0.03),
    )
    parser.add_argument(
        "--strike-z-boost-values",
        type=float,
        nargs="+",
        default=(0.012, 0.018, 0.024),
    )
    parser.add_argument(
        "--followup-lift-boost-values",
        type=float,
        nargs="+",
        default=(0.0, 0.02),
    )
    parser.add_argument(
        "--contact-offset-ratio-values",
        type=float,
        nargs="+",
        default=(0.0, 0.35),
    )
    parser.add_argument("--contact-offset-max", type=float, default=0.03)
    parser.add_argument("--coarse-episodes", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--finalist-episodes", type=int, default=30)
    parser.add_argument("--return-blend", type=float, default=0.72)
    parser.add_argument("--recovery-blend", type=float, default=0.52)
    parser.add_argument("--strike-time-horizon", type=float, default=0.14)
    parser.add_argument("--post-contact-return-assist-weight", type=float, default=0.5)
    parser.add_argument("--post-contact-return-max-intercept-time", type=float, default=0.6)
    parser.add_argument(
        "--target-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=(0.06, 0.06),
    )
    parser.add_argument("--print-configs", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def safe_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def safe_max_abs(values: list[float]) -> float | None:
    return float(np.max(np.abs(values))) if values else None


def build_env_kwargs(
    args: argparse.Namespace,
    *,
    pitch: float,
    roll: float,
    followup_lift_boost: float,
    contact_offset_ratio: float,
) -> dict[str, object]:
    contact_offset_max = args.contact_offset_max if contact_offset_ratio > 0.0 else 0.0
    return {
        "action_mode": "position_strike",
        "ball_height": args.ball_height,
        "target_ball_height": args.ball_height,
        "max_episode_steps": args.max_episode_steps,
        "reset_xy_range": args.reset_xy_range,
        "reset_velocity_xy_range": args.reset_velocity_xy_range,
        "reset_velocity_z_range": tuple(args.reset_velocity_z_range),
        "target_tilt_limit": tuple(args.target_tilt_limit),
        "followup_strike_target_tilt": (pitch, roll),
        "followup_strike_contact_offset_ratio": contact_offset_ratio,
        "followup_strike_contact_offset_max": contact_offset_max,
        "followup_strike_lift_boost": followup_lift_boost,
        "post_contact_return_assist_weight": args.post_contact_return_assist_weight,
        "post_contact_return_max_intercept_time": args.post_contact_return_max_intercept_time,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
        "strike_tilt_ramp_pitch": pitch,
        "strike_tilt_ramp_xy_tolerance": 0.04,
    }


def config_label(
    *,
    pitch: float,
    roll: float,
    strike_z_boost: float,
    followup_lift_boost: float,
    contact_offset_ratio: float,
) -> str:
    return (
        f"pitch={pitch:+.3f}|roll={roll:+.3f}|strike_z_boost={strike_z_boost:.3f}|"
        f"followup_lift_boost={followup_lift_boost:.3f}|contact_offset_ratio={contact_offset_ratio:.3f}"
    )


def summary_sort_key(row: dict[str, object]) -> tuple[float, float, float, float, float, float]:
    mean_error = row.get("all_contact_mean_outgoing_velocity_error_norm")
    error_score = -float(mean_error) if mean_error is not None else float("-inf")
    return (
        float(row.get("max_useful_bounces", 0.0)),
        float(row.get("three_or_more_useful_bounce_rate", 0.0)),
        float(row.get("two_or_more_useful_bounce_rate", 0.0)),
        float(row.get("mean_useful_bounces", 0.0)),
        float(row.get("useful_contact_rate", 0.0)),
        error_score,
    )


def evaluate_configuration(
    *,
    stage: str,
    analysis_name: str,
    config_index: int,
    pitch: float,
    roll: float,
    strike_z_boost: float,
    followup_lift_boost: float,
    contact_offset_ratio: float,
    env_kwargs: dict[str, object],
    episodes: int,
    seed: int,
    return_blend: float,
    recovery_blend: float,
    strike_time_horizon: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    # 각 후보 설정은 PPO가 아니라 HeuristicKeepUpPolicy로 먼저 feasibility를 검사한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py:49
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:17
    env = PingPongKeepUpGymEnv(**env_kwargs)
    policy = HeuristicKeepUpPolicy(
        return_blend=return_blend,
        recovery_blend=recovery_blend,
        strike_z_boost=strike_z_boost,
        strike_time_horizon=strike_time_horizon,
    )

    useful_bounces: list[int] = []
    contact_rows: list[dict[str, object]] = []
    outgoing_velocity_errors: list[float] = []
    useful_outgoing_velocity_errors: list[float] = []
    contact_racket_velocity_z_values: list[float] = []
    pre_contact_ball_velocity_z_values: list[float] = []
    actual_outgoing_velocity_z_values: list[float] = []
    contact_normal_alignment_values: list[float] = []
    useful_contact_count = 0
    contact_count = 0
    outgoing_velocity_source_counts: Counter[str] = Counter()

    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            del observation
            policy.reset()
            info: dict[str, object] = {}
            episode_contact_rows: list[dict[str, object]] = []
            while True:
                action = policy.predict(env.base_env).astype(np.float32, copy=False)
                _, _, terminated, truncated, info = env.step(action)
                if bool(info.get("contact_event_during_step", False)):
                    contact_count += 1
                    is_useful_contact = info.get("success_reason") == "useful_keepup_bounce"
                    if is_useful_contact:
                        useful_contact_count += 1

                    outgoing_velocity_source = info.get("actual_outgoing_velocity_source")
                    source_key = "none" if outgoing_velocity_source is None else str(outgoing_velocity_source)
                    outgoing_velocity_source_counts[source_key] += 1

                    outgoing_velocity_error = info.get("outgoing_velocity_error_norm")
                    if outgoing_velocity_error is not None:
                        outgoing_velocity_error_value = float(outgoing_velocity_error)
                        outgoing_velocity_errors.append(outgoing_velocity_error_value)
                        if is_useful_contact:
                            useful_outgoing_velocity_errors.append(outgoing_velocity_error_value)

                    contact_racket_velocity_z = info.get("contact_racket_velocity_z")
                    if contact_racket_velocity_z is not None:
                        contact_racket_velocity_z_values.append(float(contact_racket_velocity_z))

                    pre_contact_ball_velocity_z = info.get("pre_contact_ball_velocity_z")
                    if pre_contact_ball_velocity_z is not None:
                        pre_contact_ball_velocity_z_values.append(float(pre_contact_ball_velocity_z))

                    actual_outgoing_velocity_z = info.get("actual_outgoing_velocity_z")
                    if actual_outgoing_velocity_z is not None:
                        actual_outgoing_velocity_z_values.append(float(actual_outgoing_velocity_z))

                    contact_normal_alignment = info.get("contact_normal_alignment_with_racket_face")
                    if contact_normal_alignment is not None:
                        contact_normal_alignment_values.append(float(contact_normal_alignment))

                    episode_contact_rows.append(
                        {
                            "stage": stage,
                            "analysis_name": analysis_name,
                            "config_index": config_index,
                            "config_label": config_label(
                                pitch=pitch,
                                roll=roll,
                                strike_z_boost=strike_z_boost,
                                followup_lift_boost=followup_lift_boost,
                                contact_offset_ratio=contact_offset_ratio,
                            ),
                            "episode": episode_index + 1,
                            "step": int(info.get("step_count", 0)),
                            "contact_index": contact_count,
                            "pitch": pitch,
                            "roll": roll,
                            "requested_strike_z_boost": strike_z_boost,
                            "requested_followup_lift_boost": followup_lift_boost,
                            "requested_contact_offset_ratio": contact_offset_ratio,
                            "success_reason": info.get("success_reason"),
                            "is_useful_contact": is_useful_contact,
                            "actual_outgoing_velocity_source": outgoing_velocity_source,
                            "pre_contact_ball_velocity_x": info.get("pre_contact_ball_velocity_x"),
                            "pre_contact_ball_velocity_y": info.get("pre_contact_ball_velocity_y"),
                            "pre_contact_ball_velocity_z": pre_contact_ball_velocity_z,
                            "contact_racket_velocity_x": info.get("contact_racket_velocity_x"),
                            "contact_racket_velocity_y": info.get("contact_racket_velocity_y"),
                            "contact_racket_velocity_z": contact_racket_velocity_z,
                            "contact_mujoco_normal_x": info.get("contact_mujoco_normal_x"),
                            "contact_mujoco_normal_y": info.get("contact_mujoco_normal_y"),
                            "contact_mujoco_normal_z": info.get("contact_mujoco_normal_z"),
                            "contact_mujoco_normal_racket_to_ball_x": info.get(
                                "contact_mujoco_normal_racket_to_ball_x"
                            ),
                            "contact_mujoco_normal_racket_to_ball_y": info.get(
                                "contact_mujoco_normal_racket_to_ball_y"
                            ),
                            "contact_mujoco_normal_racket_to_ball_z": info.get(
                                "contact_mujoco_normal_racket_to_ball_z"
                            ),
                            "contact_racket_face_normal_x": info.get("contact_racket_face_normal_x"),
                            "contact_racket_face_normal_y": info.get("contact_racket_face_normal_y"),
                            "contact_racket_face_normal_z": info.get("contact_racket_face_normal_z"),
                            "contact_normal_alignment_with_racket_face": contact_normal_alignment,
                            "desired_outgoing_velocity_x": info.get("desired_outgoing_velocity_x"),
                            "desired_outgoing_velocity_y": info.get("desired_outgoing_velocity_y"),
                            "desired_outgoing_velocity_z": info.get("desired_outgoing_velocity_z"),
                            "actual_outgoing_velocity_x": info.get("actual_outgoing_velocity_x"),
                            "actual_outgoing_velocity_y": info.get("actual_outgoing_velocity_y"),
                            "actual_outgoing_velocity_z": actual_outgoing_velocity_z,
                            "outgoing_velocity_error_norm": outgoing_velocity_error,
                            "predicted_apex_x_from_actual_velocity": info.get(
                                "predicted_apex_x_from_actual_velocity"
                            ),
                            "predicted_apex_y_from_actual_velocity": info.get(
                                "predicted_apex_y_from_actual_velocity"
                            ),
                            "predicted_apex_xy_error": info.get("predicted_apex_xy_error"),
                            "next_intercept_reachable": info.get("next_intercept_reachable"),
                            "easy_next_ball_score": info.get("easy_next_ball_score"),
                        }
                    )
                if terminated or truncated:
                    break

            useful_bounce_count = int(info.get("successful_bounce_count", 0))
            useful_bounces.append(useful_bounce_count)
            for row in episode_contact_rows:
                row["episode_useful_bounces"] = useful_bounce_count
                row["episode_failure_reason"] = info.get("failure_reason")
            contact_rows.extend(episode_contact_rows)
    finally:
        env.close()

    bounce_array = np.asarray(useful_bounces, dtype=float)
    summary_row = {
        "stage": stage,
        "analysis_name": analysis_name,
        "config_index": config_index,
        "config_label": config_label(
            pitch=pitch,
            roll=roll,
            strike_z_boost=strike_z_boost,
            followup_lift_boost=followup_lift_boost,
            contact_offset_ratio=contact_offset_ratio,
        ),
        "pitch": pitch,
        "roll": roll,
        "strike_z_boost": strike_z_boost,
        "followup_lift_boost": followup_lift_boost,
        "contact_offset_ratio": contact_offset_ratio,
        "episodes": episodes,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "one_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 1.0)) if bounce_array.size else 0.0,
        "two_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 2.0)) if bounce_array.size else 0.0,
        "three_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 3.0)) if bounce_array.size else 0.0,
        "contact_event_count": contact_count,
        "useful_contact_event_count": useful_contact_count,
        "useful_contact_rate": (useful_contact_count / contact_count) if contact_count > 0 else 0.0,
        "all_contact_mean_outgoing_velocity_error_norm": safe_mean(outgoing_velocity_errors),
        "useful_contact_mean_outgoing_velocity_error_norm": safe_mean(useful_outgoing_velocity_errors),
        "mean_contact_racket_velocity_z": safe_mean(contact_racket_velocity_z_values),
        "mean_pre_contact_ball_velocity_z": safe_mean(pre_contact_ball_velocity_z_values),
        "mean_actual_outgoing_velocity_z": safe_mean(actual_outgoing_velocity_z_values),
        "contact_normal_alignment_mean": safe_mean(contact_normal_alignment_values),
        "contact_normal_alignment_max_abs": safe_max_abs(contact_normal_alignment_values),
        "outgoing_velocity_source_counts": dict(outgoing_velocity_source_counts),
    }
    return summary_row, contact_rows


def main() -> None:
    # coarse grid를 넓게 훑고, 점수가 좋은 후보만 finalist 단계에서 더 많은 episode로 재검증한다.
    args = parse_args()
    output_dir = args.output_dir or (ROOT / "artifacts" / "benchmarks" / args.analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    coarse_rows: list[dict[str, object]] = []
    coarse_contact_rows: list[dict[str, object]] = []
    config_specs = list(
        product(
            args.pitch_values,
            args.roll_values,
            args.strike_z_boost_values,
            args.followup_lift_boost_values,
            args.contact_offset_ratio_values,
        )
    )

    for config_index, spec in enumerate(config_specs, start=1):
        pitch, roll, strike_z_boost, followup_lift_boost, contact_offset_ratio = spec
        env_kwargs = build_env_kwargs(
            args,
            pitch=float(pitch),
            roll=float(roll),
            followup_lift_boost=float(followup_lift_boost),
            contact_offset_ratio=float(contact_offset_ratio),
        )
        summary_row, contact_rows = evaluate_configuration(
            stage="coarse",
            analysis_name=args.analysis_name,
            config_index=config_index,
            pitch=float(pitch),
            roll=float(roll),
            strike_z_boost=float(strike_z_boost),
            followup_lift_boost=float(followup_lift_boost),
            contact_offset_ratio=float(contact_offset_ratio),
            env_kwargs=env_kwargs,
            episodes=args.coarse_episodes,
            seed=args.seed + 1_000 * config_index,
            return_blend=args.return_blend,
            recovery_blend=args.recovery_blend,
            strike_time_horizon=args.strike_time_horizon,
        )
        coarse_rows.append(summary_row)
        coarse_contact_rows.extend(contact_rows)
        if args.print_configs:
            print(
                "coarse_config "
                f"index={config_index} label={summary_row['config_label']} "
                f"max_useful_bounces={summary_row['max_useful_bounces']} "
                f"mean_useful_bounces={summary_row['mean_useful_bounces']:.3f} "
                f"three_plus_rate={summary_row['three_or_more_useful_bounce_rate']:.3f} "
                f"mean_outgoing_error={summary_row['all_contact_mean_outgoing_velocity_error_norm']}"
            )

    sorted_coarse_rows = sorted(coarse_rows, key=summary_sort_key, reverse=True)
    finalist_count = max(0, min(int(args.top_k), len(sorted_coarse_rows)))
    finalist_rows: list[dict[str, object]] = []
    finalist_contact_rows: list[dict[str, object]] = []

    # finalist는 coarse에서 살아남은 설정을 장기 샘플로 다시 확인하는 단계다.
    for finalist_rank, coarse_row in enumerate(sorted_coarse_rows[:finalist_count], start=1):
        env_kwargs = build_env_kwargs(
            args,
            pitch=float(coarse_row["pitch"]),
            roll=float(coarse_row["roll"]),
            followup_lift_boost=float(coarse_row["followup_lift_boost"]),
            contact_offset_ratio=float(coarse_row["contact_offset_ratio"]),
        )
        summary_row, contact_rows = evaluate_configuration(
            stage="finalist",
            analysis_name=args.analysis_name,
            config_index=int(coarse_row["config_index"]),
            pitch=float(coarse_row["pitch"]),
            roll=float(coarse_row["roll"]),
            strike_z_boost=float(coarse_row["strike_z_boost"]),
            followup_lift_boost=float(coarse_row["followup_lift_boost"]),
            contact_offset_ratio=float(coarse_row["contact_offset_ratio"]),
            env_kwargs=env_kwargs,
            episodes=args.finalist_episodes,
            seed=args.seed + 100_000 + 10_000 * finalist_rank,
            return_blend=args.return_blend,
            recovery_blend=args.recovery_blend,
            strike_time_horizon=args.strike_time_horizon,
        )
        finalist_rows.append(summary_row)
        finalist_contact_rows.extend(contact_rows)
        if args.print_configs:
            print(
                "finalist_config "
                f"rank={finalist_rank} label={summary_row['config_label']} "
                f"max_useful_bounces={summary_row['max_useful_bounces']} "
                f"mean_useful_bounces={summary_row['mean_useful_bounces']:.3f} "
                f"three_plus_rate={summary_row['three_or_more_useful_bounce_rate']:.3f} "
                f"mean_outgoing_error={summary_row['all_contact_mean_outgoing_velocity_error_norm']}"
            )

    all_rows = coarse_rows + finalist_rows
    all_contact_rows = coarse_contact_rows + finalist_contact_rows
    best_coarse_row = sorted_coarse_rows[0] if sorted_coarse_rows else None
    sorted_finalists = sorted(finalist_rows, key=summary_sort_key, reverse=True)
    best_finalist_row = sorted_finalists[0] if sorted_finalists else None

    pass_row = best_finalist_row if best_finalist_row is not None else best_coarse_row
    # 결과는 summary JSON과 두 CSV(config별 요약, contact별 원자료)로 저장된다.
    # LINK: pingpong_rl2/scripts/run_heuristic_keepup_diagnostic.py:242
    feasibility_summary = {
        "analysis_name": args.analysis_name,
        "grid": {
            "pitch_values": list(args.pitch_values),
            "roll_values": list(args.roll_values),
            "strike_z_boost_values": list(args.strike_z_boost_values),
            "followup_lift_boost_values": list(args.followup_lift_boost_values),
            "contact_offset_ratio_values": list(args.contact_offset_ratio_values),
            "contact_offset_max": args.contact_offset_max,
        },
        "coarse_episodes": args.coarse_episodes,
        "finalist_episodes": args.finalist_episodes,
        "coarse_config_count": len(coarse_rows),
        "top_k": finalist_count,
        "best_coarse": best_coarse_row,
        "best_finalist": best_finalist_row,
        "pass_row": pass_row,
        "shows_three_plus": bool(pass_row is not None and int(pass_row["max_useful_bounces"]) >= 3),
        "meets_three_plus_rate_target": bool(
            pass_row is not None and float(pass_row["three_or_more_useful_bounce_rate"]) >= 0.10
        ),
    }

    summary_path = output_dir / f"{args.analysis_name}_summary.json"
    config_rows_path = output_dir / f"{args.analysis_name}_config_rows.csv"
    contacts_path = output_dir / f"{args.analysis_name}_contacts.csv"
    summary_path.write_text(json.dumps(feasibility_summary, indent=2), encoding="utf-8")
    write_csv(config_rows_path, all_rows)
    write_csv(contacts_path, all_contact_rows)
    print(f"summary_path={summary_path}")
    print(f"config_rows_path={config_rows_path}")
    print(f"contacts_path={contacts_path}")
    if pass_row is not None:
        print(
            "feasibility_summary "
            f"best_label={pass_row['config_label']} "
            f"max_useful_bounces={pass_row['max_useful_bounces']} "
            f"three_plus_rate={pass_row['three_or_more_useful_bounce_rate']:.3f} "
            f"mean_useful_bounces={pass_row['mean_useful_bounces']:.3f} "
            f"mean_outgoing_error={pass_row['all_contact_mean_outgoing_velocity_error_norm']}"
        )


if __name__ == "__main__":
    main()
