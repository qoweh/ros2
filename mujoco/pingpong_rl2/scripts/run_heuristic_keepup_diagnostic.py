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
from pingpong_rl2.utils import resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the scripted keep-up diagnostic baseline.")
    parser.add_argument("--analysis-name", type=str, default="heuristic_keepup_diagnostic")
    parser.add_argument("--variant-name", type=str, default="default")
    parser.add_argument(
        "--action-mode",
        type=str,
        default="position_strike",
        choices=(
            "position_strike",
            "position_strike_tilt",
            "position_strike_tilt_lift",
            "position_contact_frame",
            "position_contact_frame_velocity_residual",
            "position_contact_frame_velocity_tilt_residual",
            "position_contact_frame_velocity_tilt_lateral_residual",
            "position_contact_frame_velocity_tilt_lateral_apex_residual",
            "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
        ),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Optional MuJoCo scene XML for geometry A/B diagnostics.",
    )
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT)
    parser.add_argument(
        "--target-ball-height",
        type=float,
        default=None,
        help="Desired post-contact apex height above the racket. Defaults to --ball-height.",
    )
    parser.add_argument(
        "--keepup-target-xy-offset",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=(0.0, 0.0),
    )
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
    parser.add_argument("--strike-xy-correction-gain", type=float, default=0.0)
    parser.add_argument("--strike-xy-correction-max", type=float, default=0.02)
    parser.add_argument(
        "--fixed-position-residual",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--strike-position-residual",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--strike-phase-only-position-residual",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--recovery-position-residual",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument(
        "--fixed-tilt-residual",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=(0.0, 0.0),
    )
    parser.add_argument(
        "--strike-tilt-residual",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument(
        "--recovery-tilt-residual",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--fixed-followup-lift-residual", type=float, default=0.0)
    parser.add_argument("--strike-followup-lift-residual", type=float, default=None)
    parser.add_argument("--recovery-followup-lift-residual", type=float, default=None)
    parser.add_argument("--strike-tilt-ramp-pitch", type=float, default=-0.03)
    parser.add_argument("--strike-tilt-ramp-xy-tolerance", type=float, default=0.04)
    parser.add_argument(
        "--strike-tilt-assist-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
        help="Optional center-seeking pitch/roll assist. When set, the fixed strike ramp is omitted.",
    )
    parser.add_argument("--strike-tilt-assist-deadband", type=float, default=0.015)
    parser.add_argument(
        "--followup-strike-target-tilt",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--followup-strike-contact-offset-ratio", type=float, default=0.0)
    parser.add_argument("--followup-strike-contact-offset-max", type=float, default=0.0)
    parser.add_argument("--followup-strike-lift-boost", type=float, default=0.0)
    parser.add_argument("--followup-lift-action-limit", type=float, default=0.02)
    parser.add_argument("--contact-frame-apex-lift-gain", type=float, default=0.0)
    parser.add_argument("--contact-frame-apex-lift-max", type=float, default=0.0)
    parser.add_argument("--contact-frame-apex-lift-reference-velocity-z", type=float, default=-1.0)
    parser.add_argument("--contact-frame-apex-lift-restitution", type=float, default=0.8)
    parser.add_argument("--contact-frame-velocity-lead-gain", type=float, default=0.0)
    parser.add_argument("--contact-frame-velocity-lead-max", type=float, default=0.0)
    parser.add_argument("--contact-frame-velocity-target-gain", type=float, default=0.0)
    parser.add_argument("--contact-frame-velocity-target-max", type=float, default=0.0)
    parser.add_argument("--contact-frame-intercept-velocity-gain", type=float, default=0.0)
    parser.add_argument("--contact-frame-intercept-velocity-max", type=float, default=0.0)
    parser.add_argument("--contact-frame-intercept-velocity-time-floor", type=float, default=0.08)
    parser.add_argument("--contact-frame-planner-enabled", action="store_true")
    parser.add_argument(
        "--disable-contact-frame-planner-hold-during-descent",
        action="store_false",
        dest="contact_frame_planner_hold_during_descent",
        default=True,
    )
    parser.add_argument("--contact-frame-planner-min-intercept-time", type=float, default=0.03)
    parser.add_argument("--contact-frame-planner-max-intercept-time", type=float, default=0.60)
    parser.add_argument("--contact-frame-planner-target-apex-z-offset", type=float, default=0.0)
    parser.add_argument("--contact-frame-followthrough-gain", type=float, default=0.0)
    parser.add_argument("--contact-frame-followthrough-time", type=float, default=0.06)
    parser.add_argument("--contact-frame-followthrough-max", type=float, default=0.0)
    parser.add_argument("--contact-frame-trajectory-tilt-gain", type=float, default=0.0)
    parser.add_argument(
        "--contact-frame-trajectory-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-trajectory-tilt-deadband", type=float, default=0.0)
    parser.add_argument("--controller-velocity-gain", type=float, default=1.0)
    parser.add_argument("--controller-velocity-feedback-gain", type=float, default=0.0)
    parser.add_argument("--controller-max-velocity-step", type=float, default=0.02)
    parser.add_argument(
        "--contact-frame-centering-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-centering-tilt-radius", type=float, default=None)
    parser.add_argument("--contact-frame-centering-tilt-deadband", type=float, default=0.015)
    parser.add_argument("--post-contact-return-assist-weight", type=float, default=0.5)
    parser.add_argument("--post-contact-return-max-intercept-time", type=float, default=0.6)
    parser.add_argument("--post-contact-return-z-offset", type=float, default=0.0)
    parser.add_argument("--require-reachable-next-intercept-for-success", action="store_true")
    parser.add_argument("--min-easy-next-ball-score-for-success", type=float, default=None)
    parser.add_argument("--terminate-on-nonuseful-contact", action="store_true")
    parser.add_argument("--next-intercept-xy-error-penalty-weight", type=float, default=0.0)
    parser.add_argument("--post-contact-lateral-velocity-penalty-weight", type=float, default=0.0)
    parser.add_argument("--contact-xy-error-penalty-weight", type=float, default=0.0)
    parser.add_argument("--nonuseful-contact-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--target-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=(0.06, 0.06),
    )
    parser.add_argument(
        "--initial-target-tilt",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--contact-oracle-mode",
        type=str,
        choices=("none", "desired_outgoing_velocity"),
        default="none",
    )
    parser.add_argument("--contact-oracle-blend", type=float, default=1.0)
    parser.add_argument(
        "--desired-outgoing-xy-mode",
        type=str,
        choices=("next_intercept", "apex"),
        default="next_intercept",
    )
    parser.add_argument("--print-episodes", action="store_true")
    return parser.parse_args()


def write_csv(file_path: Path, rows: list[dict[str, object]]) -> None:
    # row마다 채워진 key가 조금 달라도 모든 column을 보존해 후속 분석 CSV로 남긴다.
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
    # diagnostic CLI 옵션을 학습 env 생성자 kwargs로 바꿔 heuristic과 PPO env를 같은 계약으로 비교한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/training/env_config.py:96
    env_kwargs: dict[str, object] = {
        "action_mode": args.action_mode,
        "ball_height": args.ball_height,
        "target_ball_height": args.ball_height if args.target_ball_height is None else args.target_ball_height,
        "keepup_target_xy_offset": tuple(args.keepup_target_xy_offset),
        "max_episode_steps": args.max_episode_steps,
        "reset_xy_range": args.reset_xy_range,
        "reset_velocity_xy_range": args.reset_velocity_xy_range,
        "reset_velocity_z_range": tuple(args.reset_velocity_z_range),
        "target_tilt_limit": tuple(args.target_tilt_limit),
        "followup_strike_contact_offset_ratio": args.followup_strike_contact_offset_ratio,
        "followup_strike_contact_offset_max": args.followup_strike_contact_offset_max,
        "followup_strike_lift_boost": args.followup_strike_lift_boost,
        "followup_lift_action_limit": args.followup_lift_action_limit,
        "contact_frame_apex_lift_gain": args.contact_frame_apex_lift_gain,
        "contact_frame_apex_lift_max": args.contact_frame_apex_lift_max,
        "contact_frame_apex_lift_reference_velocity_z": args.contact_frame_apex_lift_reference_velocity_z,
        "contact_frame_apex_lift_restitution": args.contact_frame_apex_lift_restitution,
        "contact_frame_velocity_lead_gain": args.contact_frame_velocity_lead_gain,
        "contact_frame_velocity_lead_max": args.contact_frame_velocity_lead_max,
        "contact_frame_velocity_target_gain": args.contact_frame_velocity_target_gain,
        "contact_frame_velocity_target_max": args.contact_frame_velocity_target_max,
        "contact_frame_intercept_velocity_gain": args.contact_frame_intercept_velocity_gain,
        "contact_frame_intercept_velocity_max": args.contact_frame_intercept_velocity_max,
        "contact_frame_intercept_velocity_time_floor": args.contact_frame_intercept_velocity_time_floor,
        "contact_frame_planner_enabled": args.contact_frame_planner_enabled,
        "contact_frame_planner_hold_during_descent": args.contact_frame_planner_hold_during_descent,
        "contact_frame_planner_min_intercept_time": args.contact_frame_planner_min_intercept_time,
        "contact_frame_planner_max_intercept_time": args.contact_frame_planner_max_intercept_time,
        "contact_frame_planner_target_apex_z_offset": args.contact_frame_planner_target_apex_z_offset,
        "contact_frame_followthrough_gain": args.contact_frame_followthrough_gain,
        "contact_frame_followthrough_time": args.contact_frame_followthrough_time,
        "contact_frame_followthrough_max": args.contact_frame_followthrough_max,
        "contact_frame_trajectory_tilt_gain": args.contact_frame_trajectory_tilt_gain,
        "contact_frame_trajectory_tilt_deadband": args.contact_frame_trajectory_tilt_deadband,
        "controller_velocity_gain": args.controller_velocity_gain,
        "controller_velocity_feedback_gain": args.controller_velocity_feedback_gain,
        "controller_max_velocity_step": args.controller_max_velocity_step,
        "contact_frame_centering_tilt_deadband": args.contact_frame_centering_tilt_deadband,
        "post_contact_return_assist_weight": args.post_contact_return_assist_weight,
        "post_contact_return_max_intercept_time": args.post_contact_return_max_intercept_time,
        "post_contact_return_z_offset": args.post_contact_return_z_offset,
        "require_reachable_next_intercept_for_success": args.require_reachable_next_intercept_for_success,
        "min_easy_next_ball_score_for_success": args.min_easy_next_ball_score_for_success,
        "terminate_on_nonuseful_contact": args.terminate_on_nonuseful_contact,
        "next_intercept_xy_error_penalty_weight": args.next_intercept_xy_error_penalty_weight,
        "post_contact_lateral_velocity_penalty_weight": args.post_contact_lateral_velocity_penalty_weight,
        "contact_xy_error_penalty_weight": args.contact_xy_error_penalty_weight,
        "nonuseful_contact_penalty_weight": args.nonuseful_contact_penalty_weight,
        "contact_oracle_mode": args.contact_oracle_mode,
        "contact_oracle_blend": args.contact_oracle_blend,
        "desired_outgoing_xy_mode": args.desired_outgoing_xy_mode,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
    }
    if args.scene_path is not None:
        env_kwargs["scene_path"] = str(resolve_input_path(args.scene_path))
    if args.followup_strike_target_tilt is not None:
        env_kwargs["followup_strike_target_tilt"] = tuple(args.followup_strike_target_tilt)
    if args.initial_target_tilt is not None:
        env_kwargs["initial_target_tilt"] = tuple(args.initial_target_tilt)
    if args.strike_tilt_assist_limit is not None:
        env_kwargs["strike_tilt_assist_limit"] = tuple(args.strike_tilt_assist_limit)
        env_kwargs["strike_tilt_assist_deadband"] = args.strike_tilt_assist_deadband
    else:
        env_kwargs["strike_tilt_ramp_pitch"] = args.strike_tilt_ramp_pitch
        env_kwargs["strike_tilt_ramp_xy_tolerance"] = args.strike_tilt_ramp_xy_tolerance
    if args.contact_frame_centering_tilt_limit is not None:
        env_kwargs["contact_frame_centering_tilt_limit"] = tuple(args.contact_frame_centering_tilt_limit)
    if args.contact_frame_trajectory_tilt_limit is not None:
        env_kwargs["contact_frame_trajectory_tilt_limit"] = tuple(args.contact_frame_trajectory_tilt_limit)
    if args.contact_frame_centering_tilt_radius is not None:
        env_kwargs["contact_frame_centering_tilt_radius"] = args.contact_frame_centering_tilt_radius
    return env_kwargs


def main() -> None:
    # hand-coded heuristic baseline을 Gym env에서 실행해 PPO 없이 가능한 성능 상한/실패 원인을 본다.
    # LINK: pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py:49
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:17
    args = parse_args()
    output_dir = args.output_dir or (ROOT / "artifacts" / "benchmarks" / args.analysis_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PingPongKeepUpGymEnv(**build_env_kwargs(args))
    policy = HeuristicKeepUpPolicy(
        return_blend=args.return_blend,
        recovery_blend=args.recovery_blend,
        strike_z_boost=args.strike_z_boost,
        strike_time_horizon=args.strike_time_horizon,
        strike_xy_correction_gain=float(args.strike_xy_correction_gain),
        strike_xy_correction_max=float(args.strike_xy_correction_max),
        fixed_position_residual=tuple(float(value) for value in args.fixed_position_residual),
        strike_position_residual=(
            None if args.strike_position_residual is None else tuple(float(value) for value in args.strike_position_residual)
        ),
        strike_phase_only_position_residual=(
            None
            if args.strike_phase_only_position_residual is None
            else tuple(float(value) for value in args.strike_phase_only_position_residual)
        ),
        recovery_position_residual=(
            None
            if args.recovery_position_residual is None
            else tuple(float(value) for value in args.recovery_position_residual)
        ),
        fixed_tilt_residual_pitch=float(args.fixed_tilt_residual[0]),
        fixed_tilt_residual_roll=float(args.fixed_tilt_residual[1]),
        strike_tilt_residual_pitch=(None if args.strike_tilt_residual is None else float(args.strike_tilt_residual[0])),
        strike_tilt_residual_roll=(None if args.strike_tilt_residual is None else float(args.strike_tilt_residual[1])),
        recovery_tilt_residual_pitch=(
            None if args.recovery_tilt_residual is None else float(args.recovery_tilt_residual[0])
        ),
        recovery_tilt_residual_roll=(
            None if args.recovery_tilt_residual is None else float(args.recovery_tilt_residual[1])
        ),
        fixed_followup_lift_residual=float(args.fixed_followup_lift_residual),
        strike_followup_lift_residual=(
            None if args.strike_followup_lift_residual is None else float(args.strike_followup_lift_residual)
        ),
        recovery_followup_lift_residual=(
            None if args.recovery_followup_lift_residual is None else float(args.recovery_followup_lift_residual)
        ),
    )

    episode_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    returns: list[float] = []
    useful_bounces: list[int] = []
    reachable_contacts = 0
    reachable_useful_contacts = 0
    contact_events = 0
    useful_contact_events = 0
    easy_scores: list[float] = []
    useful_easy_scores: list[float] = []
    all_outgoing_velocity_errors: list[float] = []
    useful_outgoing_velocity_errors: list[float] = []
    all_predicted_apex_errors: list[float] = []
    useful_predicted_apex_errors: list[float] = []
    all_predicted_next_intercept_errors: list[float] = []
    useful_predicted_next_intercept_errors: list[float] = []
    two_or_more_episode_outgoing_velocity_errors: list[float] = []
    zero_episode_outgoing_velocity_errors: list[float] = []
    two_or_more_episode_predicted_apex_errors: list[float] = []
    zero_episode_predicted_apex_errors: list[float] = []
    two_or_more_episode_predicted_next_intercept_errors: list[float] = []
    zero_episode_predicted_next_intercept_errors: list[float] = []
    outgoing_velocity_source_counts: Counter[str] = Counter()
    resolved_minus_contact_velocity_z: list[float] = []
    resolved_minus_contact_error_norm: list[float] = []
    contact_normal_alignment_scores: list[float] = []
    contact_started_during_trace_count = 0
    oracle_contact_events = 0

    try:
        # episode loop: heuristic action을 적용하고 contact event마다 outgoing/apex/intercept 지표를 축적한다.
        # LINK: pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:53
        for episode_index in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode_index)
            del observation
            policy.reset()
            episode_return = 0.0
            info: dict[str, object] = {}
            episode_outgoing_velocity_errors: list[float] = []
            episode_predicted_apex_errors: list[float] = []
            episode_predicted_next_intercept_errors: list[float] = []
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
                    outgoing_velocity_error = info.get("outgoing_velocity_error_norm")
                    contact_substep_outgoing_velocity_error = info.get("contact_substep_outgoing_velocity_error_norm")
                    actual_outgoing_velocity_source = info.get("actual_outgoing_velocity_source")
                    source_key = "none" if actual_outgoing_velocity_source is None else str(actual_outgoing_velocity_source)
                    outgoing_velocity_source_counts[source_key] += 1
                    if outgoing_velocity_error is not None:
                        outgoing_velocity_error_value = float(outgoing_velocity_error)
                        all_outgoing_velocity_errors.append(outgoing_velocity_error_value)
                        episode_outgoing_velocity_errors.append(outgoing_velocity_error_value)
                        if contact_substep_outgoing_velocity_error is not None:
                            resolved_minus_contact_error_norm.append(
                                outgoing_velocity_error_value - float(contact_substep_outgoing_velocity_error)
                            )
                    predicted_apex_error = info.get("predicted_apex_xy_error")
                    if predicted_apex_error is not None:
                        predicted_apex_error_value = float(predicted_apex_error)
                        all_predicted_apex_errors.append(predicted_apex_error_value)
                        episode_predicted_apex_errors.append(predicted_apex_error_value)
                    predicted_next_intercept_error = info.get("predicted_next_intercept_xy_error")
                    if predicted_next_intercept_error is not None:
                        predicted_next_intercept_error_value = float(predicted_next_intercept_error)
                        all_predicted_next_intercept_errors.append(predicted_next_intercept_error_value)
                        episode_predicted_next_intercept_errors.append(predicted_next_intercept_error_value)
                    actual_outgoing_velocity_z = info.get("actual_outgoing_velocity_z")
                    contact_ball_velocity_z = info.get("contact_ball_velocity_z")
                    if actual_outgoing_velocity_z is not None and contact_ball_velocity_z is not None:
                        resolved_minus_contact_velocity_z.append(
                            float(actual_outgoing_velocity_z) - float(contact_ball_velocity_z)
                        )
                    racket_face_normal = np.array(
                        [
                            info.get("contact_racket_face_normal_x"),
                            info.get("contact_racket_face_normal_y"),
                            info.get("contact_racket_face_normal_z"),
                        ],
                        dtype=object,
                    )
                    contact_normal_racket_to_ball = np.array(
                        [
                            info.get("contact_mujoco_normal_racket_to_ball_x"),
                            info.get("contact_mujoco_normal_racket_to_ball_y"),
                            info.get("contact_mujoco_normal_racket_to_ball_z"),
                        ],
                        dtype=object,
                    )
                    contact_normal_alignment = None
                    if not any(value is None for value in racket_face_normal) and not any(
                        value is None for value in contact_normal_racket_to_ball
                    ):
                        racket_face_normal_vector = np.asarray(racket_face_normal, dtype=float)
                        contact_normal_vector = np.asarray(contact_normal_racket_to_ball, dtype=float)
                        contact_normal_alignment = float(np.dot(racket_face_normal_vector, contact_normal_vector))
                        contact_normal_alignment_scores.append(contact_normal_alignment)
                    if bool(info.get("contact_started_during_trace", False)):
                        contact_started_during_trace_count += 1
                    if bool(info.get("oracle_contact_applied", False)):
                        oracle_contact_events += 1
                    contact_rows.append(
                        {
                            "episode": episode_index + 1,
                            "step": int(info.get("step_count", 0)),
                            "contact_index": contact_events,
                            "success_reason": info.get("success_reason"),
                            "is_useful_contact": info.get("success_reason") == "useful_keepup_bounce",
                            "actual_outgoing_velocity_source": actual_outgoing_velocity_source,
                            "oracle_contact_applied": info.get("oracle_contact_applied"),
                            "oracle_contact_mode": info.get("oracle_contact_mode"),
                            "oracle_contact_blend": info.get("oracle_contact_blend"),
                            "oracle_contact_base_source": info.get("oracle_contact_base_source"),
                            "contact_started_during_trace": info.get("contact_started_during_trace"),
                            "contact_active_at_step_start": info.get("contact_active_at_step_start"),
                            "contact_substep": info.get("contact_substep"),
                            "contact_end_substep": info.get("contact_end_substep"),
                            "pre_contact_ball_velocity_x": info.get("pre_contact_ball_velocity_x"),
                            "pre_contact_ball_velocity_y": info.get("pre_contact_ball_velocity_y"),
                            "pre_contact_ball_velocity_z": info.get("pre_contact_ball_velocity_z"),
                            "contact_ball_velocity_x": info.get("contact_ball_velocity_x"),
                            "contact_ball_velocity_y": info.get("contact_ball_velocity_y"),
                            "contact_ball_velocity_z": info.get("contact_ball_velocity_z"),
                            "post_contact_1_ball_velocity_x": info.get("post_contact_1_ball_velocity_x"),
                            "post_contact_1_ball_velocity_y": info.get("post_contact_1_ball_velocity_y"),
                            "post_contact_1_ball_velocity_z": info.get("post_contact_1_ball_velocity_z"),
                            "contact_end_ball_velocity_x": info.get("contact_end_ball_velocity_x"),
                            "contact_end_ball_velocity_y": info.get("contact_end_ball_velocity_y"),
                            "contact_end_ball_velocity_z": info.get("contact_end_ball_velocity_z"),
                            "oracle_preoverride_outgoing_ball_velocity_x": info.get(
                                "oracle_preoverride_outgoing_ball_velocity_x"
                            ),
                            "oracle_preoverride_outgoing_ball_velocity_y": info.get(
                                "oracle_preoverride_outgoing_ball_velocity_y"
                            ),
                            "oracle_preoverride_outgoing_ball_velocity_z": info.get(
                                "oracle_preoverride_outgoing_ball_velocity_z"
                            ),
                            "oracle_post_contact_ball_velocity_x": info.get("oracle_post_contact_ball_velocity_x"),
                            "oracle_post_contact_ball_velocity_y": info.get("oracle_post_contact_ball_velocity_y"),
                            "oracle_post_contact_ball_velocity_z": info.get("oracle_post_contact_ball_velocity_z"),
                            "oracle_desired_outgoing_velocity_x": info.get("oracle_desired_outgoing_velocity_x"),
                            "oracle_desired_outgoing_velocity_y": info.get("oracle_desired_outgoing_velocity_y"),
                            "oracle_desired_outgoing_velocity_z": info.get("oracle_desired_outgoing_velocity_z"),
                            "desired_outgoing_velocity_x": info.get("desired_outgoing_velocity_x"),
                            "desired_outgoing_velocity_y": info.get("desired_outgoing_velocity_y"),
                            "desired_outgoing_velocity_z": info.get("desired_outgoing_velocity_z"),
                            "actual_outgoing_velocity_x": info.get("actual_outgoing_velocity_x"),
                            "actual_outgoing_velocity_y": info.get("actual_outgoing_velocity_y"),
                            "actual_outgoing_velocity_z": info.get("actual_outgoing_velocity_z"),
                            "outgoing_velocity_error_norm": info.get("outgoing_velocity_error_norm"),
                            "contact_substep_outgoing_velocity_error_norm": contact_substep_outgoing_velocity_error,
                            "resolved_minus_contact_velocity_z": (
                                None
                                if actual_outgoing_velocity_z is None or contact_ball_velocity_z is None
                                else float(actual_outgoing_velocity_z) - float(contact_ball_velocity_z)
                            ),
                            "resolved_minus_contact_error_norm": (
                                None
                                if outgoing_velocity_error is None or contact_substep_outgoing_velocity_error is None
                                else float(outgoing_velocity_error) - float(contact_substep_outgoing_velocity_error)
                            ),
                            "predicted_apex_xy_error": info.get("predicted_apex_xy_error"),
                            "contact_substep_predicted_apex_xy_error": info.get("contact_substep_predicted_apex_xy_error"),
                            "predicted_next_intercept_xy_error": info.get("predicted_next_intercept_xy_error"),
                            "contact_substep_predicted_next_intercept_xy_error": info.get(
                                "contact_substep_predicted_next_intercept_xy_error"
                            ),
                            "desired_outgoing_xy_mode": info.get("desired_outgoing_xy_mode"),
                            "desired_outgoing_target_z": info.get("desired_outgoing_target_z"),
                            "desired_outgoing_apex_x": info.get("desired_outgoing_apex_x"),
                            "desired_outgoing_apex_y": info.get("desired_outgoing_apex_y"),
                            "predicted_next_intercept_x_from_actual_velocity": info.get(
                                "predicted_next_intercept_x_from_actual_velocity"
                            ),
                            "predicted_next_intercept_y_from_actual_velocity": info.get(
                                "predicted_next_intercept_y_from_actual_velocity"
                            ),
                            "predicted_next_intercept_time_from_actual_velocity": info.get(
                                "predicted_next_intercept_time_from_actual_velocity"
                            ),
                            "contact_racket_velocity_x": info.get("contact_racket_velocity_x"),
                            "contact_racket_velocity_y": info.get("contact_racket_velocity_y"),
                            "contact_racket_velocity_z": info.get("contact_racket_velocity_z"),
                            "target_velocity_x": (
                                None if info.get("target_velocity") is None else float(info["target_velocity"][0])
                            ),
                            "target_velocity_y": (
                                None if info.get("target_velocity") is None else float(info["target_velocity"][1])
                            ),
                            "target_velocity_z": (
                                None if info.get("target_velocity") is None else float(info["target_velocity"][2])
                            ),
                            "contact_frame_intercept_velocity_target_x": (
                                None
                                if info.get("contact_frame_intercept_velocity_target") is None
                                else float(info["contact_frame_intercept_velocity_target"][0])
                            ),
                            "contact_frame_intercept_velocity_target_y": (
                                None
                                if info.get("contact_frame_intercept_velocity_target") is None
                                else float(info["contact_frame_intercept_velocity_target"][1])
                            ),
                            "contact_frame_intercept_velocity_target_z": (
                                None
                                if info.get("contact_frame_intercept_velocity_target") is None
                                else float(info["contact_frame_intercept_velocity_target"][2])
                            ),
                            "contact_frame_planner_active": info.get("contact_frame_planner_active"),
                            "contact_frame_planner_intercept_time": info.get("contact_frame_planner_intercept_time"),
                            "contact_frame_planner_contact_x": (
                                None
                                if info.get("contact_frame_planner_contact_position") is None
                                else float(info["contact_frame_planner_contact_position"][0])
                            ),
                            "contact_frame_planner_contact_y": (
                                None
                                if info.get("contact_frame_planner_contact_position") is None
                                else float(info["contact_frame_planner_contact_position"][1])
                            ),
                            "contact_frame_planner_contact_z": (
                                None
                                if info.get("contact_frame_planner_contact_position") is None
                                else float(info["contact_frame_planner_contact_position"][2])
                            ),
                            "contact_frame_planner_desired_velocity_x": (
                                None
                                if info.get("contact_frame_planner_desired_velocity") is None
                                else float(info["contact_frame_planner_desired_velocity"][0])
                            ),
                            "contact_frame_planner_desired_velocity_y": (
                                None
                                if info.get("contact_frame_planner_desired_velocity") is None
                                else float(info["contact_frame_planner_desired_velocity"][1])
                            ),
                            "contact_frame_planner_desired_velocity_z": (
                                None
                                if info.get("contact_frame_planner_desired_velocity") is None
                                else float(info["contact_frame_planner_desired_velocity"][2])
                            ),
                            "contact_racket_face_normal_x": info.get("contact_racket_face_normal_x"),
                            "contact_racket_face_normal_y": info.get("contact_racket_face_normal_y"),
                            "contact_racket_face_normal_z": info.get("contact_racket_face_normal_z"),
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
                            "contact_normal_alignment_with_racket_face": contact_normal_alignment,
                            "contact_relative_velocity_x": info.get("contact_relative_velocity_x"),
                            "contact_relative_velocity_y": info.get("contact_relative_velocity_y"),
                            "contact_relative_velocity_z": info.get("contact_relative_velocity_z"),
                            "pre_contact_relative_velocity_x": info.get("pre_contact_relative_velocity_x"),
                            "pre_contact_relative_velocity_y": info.get("pre_contact_relative_velocity_y"),
                            "pre_contact_relative_velocity_z": info.get("pre_contact_relative_velocity_z"),
                            "next_intercept_reachable": info.get("next_intercept_reachable"),
                            "easy_next_ball_score": info.get("easy_next_ball_score"),
                        }
                    )
                    if info.get("success_reason") == "useful_keepup_bounce":
                        useful_contact_events += 1
                        if bool(info.get("next_intercept_reachable", False)):
                            reachable_useful_contacts += 1
                        if easy_score is not None:
                            useful_easy_scores.append(float(easy_score))
                        if outgoing_velocity_error is not None:
                            useful_outgoing_velocity_errors.append(float(outgoing_velocity_error))
                        if predicted_apex_error is not None:
                            useful_predicted_apex_errors.append(float(predicted_apex_error))
                        if predicted_next_intercept_error is not None:
                            useful_predicted_next_intercept_errors.append(float(predicted_next_intercept_error))
                if terminated or truncated:
                    break

            useful_bounce_count = int(info.get("successful_bounce_count", 0))
            failure_reason = info.get("failure_reason")
            if failure_reason is None:
                failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
            failure_counts[str(failure_reason)] += 1
            returns.append(episode_return)
            useful_bounces.append(useful_bounce_count)
            if useful_bounce_count >= 2:
                two_or_more_episode_outgoing_velocity_errors.extend(episode_outgoing_velocity_errors)
                two_or_more_episode_predicted_apex_errors.extend(episode_predicted_apex_errors)
                two_or_more_episode_predicted_next_intercept_errors.extend(
                    episode_predicted_next_intercept_errors
                )
            if useful_bounce_count == 0:
                zero_episode_outgoing_velocity_errors.extend(episode_outgoing_velocity_errors)
                zero_episode_predicted_apex_errors.extend(episode_predicted_apex_errors)
                zero_episode_predicted_next_intercept_errors.extend(episode_predicted_next_intercept_errors)
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
    contact_count_array = np.asarray([float(row["contacts"]) for row in episode_rows], dtype=float)
    time_limit_episodes = sum(1 for row in episode_rows if row["failure_reason"] == "time_limit")
    # summary는 baseline 수치, episodes.csv는 episode 결과, contacts.csv는 contact별 물리량이다.
    summary = {
        "analysis_name": args.analysis_name,
        "variant_name": args.variant_name,
        "episodes": args.episodes,
        "seed": args.seed,
        "env_kwargs": build_env_kwargs(args),
        "heuristic_config": {
            "return_blend": args.return_blend,
            "recovery_blend": args.recovery_blend,
            "strike_z_boost": args.strike_z_boost,
            "strike_time_horizon": args.strike_time_horizon,
            "strike_xy_correction_gain": float(args.strike_xy_correction_gain),
            "strike_xy_correction_max": float(args.strike_xy_correction_max),
            "fixed_position_residual": [float(value) for value in args.fixed_position_residual],
            "strike_position_residual": (
                None
                if args.strike_position_residual is None
                else [float(value) for value in args.strike_position_residual]
            ),
            "strike_phase_only_position_residual": (
                None
                if args.strike_phase_only_position_residual is None
                else [float(value) for value in args.strike_phase_only_position_residual]
            ),
            "recovery_position_residual": (
                None
                if args.recovery_position_residual is None
                else [float(value) for value in args.recovery_position_residual]
            ),
            "fixed_tilt_residual": [
                float(args.fixed_tilt_residual[0]),
                float(args.fixed_tilt_residual[1]),
            ],
            "strike_tilt_residual": (
                None
                if args.strike_tilt_residual is None
                else [float(args.strike_tilt_residual[0]), float(args.strike_tilt_residual[1])]
            ),
            "recovery_tilt_residual": (
                None
                if args.recovery_tilt_residual is None
                else [float(args.recovery_tilt_residual[0]), float(args.recovery_tilt_residual[1])]
            ),
            "fixed_followup_lift_residual": float(args.fixed_followup_lift_residual),
            "strike_followup_lift_residual": (
                None if args.strike_followup_lift_residual is None else float(args.strike_followup_lift_residual)
            ),
            "recovery_followup_lift_residual": (
                None if args.recovery_followup_lift_residual is None else float(args.recovery_followup_lift_residual)
            ),
        },
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "mean_contacts": float(contact_count_array.mean()) if contact_count_array.size else 0.0,
        "max_contacts": int(contact_count_array.max()) if contact_count_array.size else 0,
        "time_limit_episode_rate": time_limit_episodes / args.episodes if args.episodes > 0 else 0.0,
        "one_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 1.0)) if bounce_array.size else 0.0,
        "two_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 2.0)) if bounce_array.size else 0.0,
        "three_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 3.0)) if bounce_array.size else 0.0,
        "contact_event_count": contact_events,
        "useful_contact_event_count": useful_contact_events,
        "oracle_contact_event_count": oracle_contact_events,
        "oracle_contact_rate": (oracle_contact_events / contact_events) if contact_events > 0 else 0.0,
        "outgoing_velocity_source_counts": dict(outgoing_velocity_source_counts),
        "contact_started_during_trace_rate": (
            contact_started_during_trace_count / contact_events if contact_events > 0 else 0.0
        ),
        "contact_end_velocity_source_rate": (
            outgoing_velocity_source_counts.get("contact_end_ball_velocity", 0) / contact_events
            if contact_events > 0
            else 0.0
        ),
        "resolved_post_contact_source_rate": (
            1.0 - outgoing_velocity_source_counts.get("contact_ball_velocity", 0) / contact_events
            if contact_events > 0
            else 0.0
        ),
        "next_intercept_reachable_rate": (reachable_contacts / contact_events) if contact_events > 0 else 0.0,
        "useful_contact_next_intercept_reachable_rate": (
            reachable_useful_contacts / useful_contact_events if useful_contact_events > 0 else 0.0
        ),
        "mean_easy_next_ball_score": float(np.mean(easy_scores)) if easy_scores else 0.0,
        "useful_contact_mean_easy_next_ball_score": float(np.mean(useful_easy_scores)) if useful_easy_scores else 0.0,
        "all_contact_mean_outgoing_velocity_error_norm": (
            float(np.mean(all_outgoing_velocity_errors)) if all_outgoing_velocity_errors else None
        ),
        "useful_contact_mean_outgoing_velocity_error_norm": (
            float(np.mean(useful_outgoing_velocity_errors)) if useful_outgoing_velocity_errors else None
        ),
        "two_or_more_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm": (
            float(np.mean(two_or_more_episode_outgoing_velocity_errors))
            if two_or_more_episode_outgoing_velocity_errors
            else None
        ),
        "zero_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm": (
            float(np.mean(zero_episode_outgoing_velocity_errors)) if zero_episode_outgoing_velocity_errors else None
        ),
        "all_contact_mean_predicted_apex_xy_error": (
            float(np.mean(all_predicted_apex_errors)) if all_predicted_apex_errors else None
        ),
        "useful_contact_mean_predicted_apex_xy_error": (
            float(np.mean(useful_predicted_apex_errors)) if useful_predicted_apex_errors else None
        ),
        "two_or_more_useful_bounce_episode_contact_mean_predicted_apex_xy_error": (
            float(np.mean(two_or_more_episode_predicted_apex_errors))
            if two_or_more_episode_predicted_apex_errors
            else None
        ),
        "zero_useful_bounce_episode_contact_mean_predicted_apex_xy_error": (
            float(np.mean(zero_episode_predicted_apex_errors)) if zero_episode_predicted_apex_errors else None
        ),
        "all_contact_mean_predicted_next_intercept_xy_error": (
            float(np.mean(all_predicted_next_intercept_errors))
            if all_predicted_next_intercept_errors
            else None
        ),
        "useful_contact_mean_predicted_next_intercept_xy_error": (
            float(np.mean(useful_predicted_next_intercept_errors))
            if useful_predicted_next_intercept_errors
            else None
        ),
        "two_or_more_useful_bounce_episode_contact_mean_predicted_next_intercept_xy_error": (
            float(np.mean(two_or_more_episode_predicted_next_intercept_errors))
            if two_or_more_episode_predicted_next_intercept_errors
            else None
        ),
        "zero_useful_bounce_episode_contact_mean_predicted_next_intercept_xy_error": (
            float(np.mean(zero_episode_predicted_next_intercept_errors))
            if zero_episode_predicted_next_intercept_errors
            else None
        ),
        "mean_resolved_minus_contact_velocity_z": (
            float(np.mean(resolved_minus_contact_velocity_z)) if resolved_minus_contact_velocity_z else None
        ),
        "mean_abs_resolved_minus_contact_velocity_z": (
            float(np.mean(np.abs(resolved_minus_contact_velocity_z))) if resolved_minus_contact_velocity_z else None
        ),
        "max_abs_resolved_minus_contact_velocity_z": (
            float(np.max(np.abs(resolved_minus_contact_velocity_z))) if resolved_minus_contact_velocity_z else None
        ),
        "mean_resolved_minus_contact_error_norm": (
            float(np.mean(resolved_minus_contact_error_norm)) if resolved_minus_contact_error_norm else None
        ),
        "mean_abs_resolved_minus_contact_error_norm": (
            float(np.mean(np.abs(resolved_minus_contact_error_norm))) if resolved_minus_contact_error_norm else None
        ),
        "contact_normal_alignment_mean": (
            float(np.mean(contact_normal_alignment_scores)) if contact_normal_alignment_scores else None
        ),
        "contact_normal_alignment_min": (
            float(np.min(contact_normal_alignment_scores)) if contact_normal_alignment_scores else None
        ),
        "contact_normal_alignment_max": (
            float(np.max(contact_normal_alignment_scores)) if contact_normal_alignment_scores else None
        ),
        "failure_counts": dict(failure_counts),
    }

    summary_path = output_dir / f"{args.analysis_name}_summary.json"
    episodes_path = output_dir / f"{args.analysis_name}_episodes.csv"
    contacts_path = output_dir / f"{args.analysis_name}_contacts.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(episodes_path, episode_rows)
    write_csv(contacts_path, contact_rows)
    print(f"summary_path={summary_path}")
    print(f"episodes_path={episodes_path}")
    print(f"contacts_path={contacts_path}")
    print(
        "heuristic_summary "
        f"variant={args.variant_name} "
        f"mean_useful_bounces={summary['mean_useful_bounces']:.3f} "
        f"two_or_more_rate={summary['two_or_more_useful_bounce_rate']:.3f} "
        f"reachable_rate={summary['next_intercept_reachable_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
