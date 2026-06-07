from __future__ import annotations

import argparse
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

from pingpong_rl2.analysis.csv_io import write_csv
from pingpong_rl2.analysis.rebound_env import apply_rebound_env_overrides
from pingpong_rl2.analysis.rebound_metrics import (
    _APEX_TARGET_CHOICES,
    _UNLIMITED_ANALYSIS_STEP_LIMIT,
    apex_target_xy_candidates,
    compute_contact_quality_metrics,
    compute_next_intercept_metrics,
)
from pingpong_rl2.analysis.rebound_summary import (
    summarize_contacts,
    summarize_episode_apex_targets,
    summarize_episode_next_intercepts,
    summarize_episode_outgoing_velocities,
    summarize_terminal_contacts,
)
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.utils import (
    infer_run_name_from_model_path,
    resolve_env_kwargs_for_model,
    resolve_requested_run_name,
    resolve_saved_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rebound direction/contact quality for a saved pingpong_rl2 PPO policy.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-version", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Optional MuJoCo scene XML override. Saved model env_config scene_path is used when this is omitted.",
    )
    parser.add_argument("--ball-height", type=float, default=None)
    parser.add_argument(
        "--target-ball-height",
        type=float,
        default=None,
        help="Override only the desired post-contact apex height. --ball-height still controls reset height.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--reset-ball-height-range", type=float, default=None)
    parser.add_argument(
        "--reset-ball-height-bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--reset-xy-range", type=float, default=None)
    parser.add_argument("--reset-xy-sampling", type=str, choices=("square", "disk"), default=None)
    parser.add_argument("--reset-velocity-xy-range", type=float, default=None)
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--reset-ball-angular-velocity-range", type=float, default=None)
    parser.add_argument("--success-velocity-threshold", type=float, default=None)
    parser.add_argument(
        "--keepup-target-xy-offset",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Override the repeat keep-up target XY offset from the controller anchor.",
    )
    parser.add_argument(
        "--post-contact-return-z-offset",
        type=float,
        default=None,
        help="Override the env post-contact vertical racket return offset.",
    )
    parser.add_argument(
        "--disable-post-contact-return-predict-during-rise",
        action="store_false",
        dest="post_contact_return_predict_during_rise",
        default=True,
        help="Return to the anchor while the ball rises instead of chasing a predicted future intercept.",
    )
    parser.add_argument("--contact-frame-velocity-target-gain", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-target-max", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-scale-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-outgoing-xy-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-racket-vz-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-racket-xy-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-tilt-scale-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-target-apex-z-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-strike-plane-z-action-limit", type=float, default=None)
    parser.add_argument("--tracking-strike-plane-offset", type=float, default=None)
    parser.add_argument("--contact-frame-intercept-velocity-gain", type=float, default=None)
    parser.add_argument("--contact-frame-intercept-velocity-max", type=float, default=None)
    parser.add_argument("--contact-frame-intercept-velocity-time-floor", type=float, default=None)
    parser.add_argument("--contact-frame-planner-enabled", action="store_true")
    parser.add_argument(
        "--disable-contact-frame-planner-hold-during-descent",
        action="store_false",
        dest="contact_frame_planner_hold_during_descent",
        default=True,
    )
    parser.add_argument("--contact-frame-planner-min-intercept-time", type=float, default=None)
    parser.add_argument("--contact-frame-planner-max-intercept-time", type=float, default=None)
    parser.add_argument("--contact-frame-planner-target-apex-z-offset", type=float, default=None)
    parser.add_argument("--contact-frame-planner-contact-offset-ratio", type=float, default=None)
    parser.add_argument("--contact-frame-planner-contact-offset-max", type=float, default=None)
    parser.add_argument("--contact-frame-lateral-brake-gain", type=float, default=None)
    parser.add_argument("--contact-frame-lateral-brake-max", type=float, default=None)
    parser.add_argument("--contact-frame-lateral-brake-radius", type=float, default=None)
    parser.add_argument("--controller-velocity-gain", type=float, default=None)
    parser.add_argument("--controller-velocity-feedback-gain", type=float, default=None)
    parser.add_argument("--controller-max-velocity-step", type=float, default=None)
    parser.add_argument("--controller-nullspace-posture-gain", type=float, default=None)
    parser.add_argument("--controller-nullspace-posture-max-step", type=float, default=None)
    parser.add_argument("--controller-body-clearance-gain", type=float, default=None)
    parser.add_argument("--controller-body-clearance-margin", type=float, default=None)
    parser.add_argument("--controller-body-clearance-vertical-margin", type=float, default=None)
    parser.add_argument("--controller-body-clearance-max-step", type=float, default=None)
    parser.add_argument("--controller-body-clearance-body-names", type=str, nargs="+", default=None)
    parser.add_argument("--contact-frame-trajectory-tilt-gain", type=float, default=None)
    parser.add_argument(
        "--contact-frame-trajectory-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-trajectory-tilt-deadband", type=float, default=None)
    parser.add_argument(
        "--contact-frame-centering-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-centering-tilt-radius", type=float, default=None)
    parser.add_argument("--contact-frame-centering-tilt-deadband", type=float, default=None)
    parser.add_argument("--next-intercept-success-radius", type=float, default=None)
    parser.add_argument("--easy-next-ball-xy-radius", type=float, default=None)
    parser.add_argument("--next-intercept-xy-error-penalty-weight", type=float, default=None)
    parser.add_argument("--post-contact-lateral-velocity-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-xy-error-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-racket-outward-velocity-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-racket-outward-velocity-tolerance", type=float, default=None)
    parser.add_argument("--nonuseful-contact-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-apex-potential-reward-weight", type=float, default=None)
    parser.add_argument("--contact-apex-potential-gamma", type=float, default=None)
    parser.add_argument("--contact-apex-potential-cap", type=float, default=None)
    parser.add_argument("--contact-lateral-stability-min-apex-ratio", type=float, default=None)
    parser.add_argument("--stable-contact-min-apex-ratio", type=float, default=None)
    parser.add_argument(
        "--require-reachable-next-intercept-for-success",
        action="store_true",
        help="Override the env so useful contacts must leave the next descending intercept reachable.",
    )
    parser.add_argument(
        "--min-easy-next-ball-score-for-success",
        type=float,
        default=None,
        help="Override the env with a lower bound on easy_next_ball_score for useful contacts.",
    )
    parser.add_argument(
        "--terminate-on-nonuseful-contact",
        action="store_true",
        help="Override the env so a racket contact that is not useful terminates the episode.",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--episode-step-limit",
        type=int,
        default=None,
        help="Analysis-only safety cap for unlimited envs. Defaults to 3600 steps when max_episode_steps is unlimited.",
    )
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


def main() -> None:
    args = parse_args()
    resolved_run_name = None if args.run_name is None else resolve_requested_run_name(args.run_name, args.run_version)
    model_path = resolve_saved_model_path(args.model_path, resolved_run_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved PPO model not found: {model_path}")

    env_kwargs = resolve_env_kwargs_for_model(
        model_path,
        scene_path=args.scene_path,
        ball_height=args.ball_height,
        target_ball_height=args.target_ball_height,
        max_episode_steps=args.max_episode_steps,
        reset_ball_height_range=args.reset_ball_height_range,
        reset_ball_height_bounds=args.reset_ball_height_bounds,
        reset_xy_range=args.reset_xy_range,
        reset_xy_sampling=args.reset_xy_sampling,
        reset_velocity_xy_range=args.reset_velocity_xy_range,
        reset_velocity_z_range=args.reset_velocity_z_range,
        reset_ball_angular_velocity_range=args.reset_ball_angular_velocity_range,
        success_velocity_threshold=args.success_velocity_threshold,
    )
    env_kwargs = apply_rebound_env_overrides(args, env_kwargs)
    env = PingPongKeepUpGymEnv(**env_kwargs)
    if args.episode_step_limit is None:
        episode_step_limit = _UNLIMITED_ANALYSIS_STEP_LIMIT if env.base_env.max_episode_steps is None else None
    else:
        episode_step_limit = None if args.episode_step_limit <= 0 else int(args.episode_step_limit)
    model = PPO.load(str(model_path))
    run_name = infer_run_name_from_model_path(model_path)
    gravity_z = float(env.base_env.sim.model.opt.gravity[2])
    gravity_magnitude = max(abs(gravity_z), 1.0e-6)
    strike_plane_offset = float(env.base_env._tracking_strike_plane_offset())
    strike_zone_xy_radius = float(env.base_env.next_intercept_success_radius)
    output_dir = (model_path.parent / "analysis") if args.output_dir is None else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_name = args.analysis_name or f"{run_name}_rebound_{args.episodes}ep"

    episode_rows: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    returns: list[float] = []
    useful_bounces: list[int] = []
    stable_cycles: list[int] = []
    failure_counts: Counter[str] = Counter()
    robot_body_contact_counts: Counter[str] = Counter()

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
                if not terminated and not truncated and episode_step_limit is not None and step_count >= episode_step_limit:
                    truncated = True
                    info = dict(info)
                    info["truncated"] = True
                    info["analysis_step_limit"] = episode_step_limit

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
                    contact_ball_position_z = info.get("contact_ball_position_z")
                    ball_velocity_x = info.get("contact_ball_velocity_x")
                    ball_velocity_y = info.get("contact_ball_velocity_y")
                    ball_velocity_z = info.get("contact_ball_velocity_z")
                    controller_anchor_position = info.get("controller_anchor_position")
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
                    contact_ball_position = None
                    if (
                        contact_ball_position_x is not None
                        and contact_ball_position_y is not None
                        and contact_ball_position_z is not None
                    ):
                        contact_ball_position = np.array(
                            [
                                float(contact_ball_position_x),
                                float(contact_ball_position_y),
                                float(contact_ball_position_z),
                            ],
                            dtype=float,
                        )
                    ball_velocity = None
                    if ball_velocity_x is not None and ball_velocity_y is not None and ball_velocity_z is not None:
                        ball_velocity = np.array(
                            [float(ball_velocity_x), float(ball_velocity_y), float(ball_velocity_z)],
                            dtype=float,
                        )
                    controller_anchor = None
                    if controller_anchor_position is not None:
                        controller_anchor = np.asarray(controller_anchor_position, dtype=float)
                    racket_velocity = None
                    if (
                        info.get("contact_racket_velocity_x") is not None
                        and info.get("contact_racket_velocity_y") is not None
                        and info.get("contact_racket_velocity_z") is not None
                    ):
                        racket_velocity = np.array(
                            [
                                float(info["contact_racket_velocity_x"]),
                                float(info["contact_racket_velocity_y"]),
                                float(info["contact_racket_velocity_z"]),
                            ],
                            dtype=float,
                        )
                    racket_face_normal = None
                    if (
                        info.get("contact_racket_face_normal_x") is not None
                        and info.get("contact_racket_face_normal_y") is not None
                        and info.get("contact_racket_face_normal_z") is not None
                    ):
                        racket_face_normal = np.array(
                            [
                                float(info["contact_racket_face_normal_x"]),
                                float(info["contact_racket_face_normal_y"]),
                                float(info["contact_racket_face_normal_z"]),
                            ],
                            dtype=float,
                        )
                    next_intercept_metrics = compute_next_intercept_metrics(
                        contact_ball_position=contact_ball_position,
                        ball_velocity=ball_velocity,
                        controller_anchor_position=controller_anchor,
                        gravity_z=gravity_z,
                        strike_plane_offset=strike_plane_offset,
                        strike_zone_xy_radius=strike_zone_xy_radius,
                    )
                    contact_quality_metrics = compute_contact_quality_metrics(
                        ball_velocity=ball_velocity,
                        racket_velocity=racket_velocity,
                        racket_face_normal=racket_face_normal,
                    )
                    reward_terms = info.get("reward_terms")
                    if not isinstance(reward_terms, dict):
                        reward_terms = {}
                    applied_action = None
                    if info.get("applied_action") is not None:
                        applied_action = np.asarray(info["applied_action"], dtype=float).reshape(-1)
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
                        "stable_cycle_observed": info.get("stable_cycle_observed"),
                        "stable_cycle_count": info.get("stable_cycle_count"),
                        "consecutive_stable_cycle_count": info.get("consecutive_stable_cycle_count"),
                        "stable_contact_term": reward_terms.get("stable_contact_term"),
                        "stable_cycle_term": reward_terms.get("stable_cycle_term"),
                        "contact_apex_progress_term": reward_terms.get("contact_apex_progress_term"),
                        "contact_apex_recovery_progress_term": reward_terms.get(
                            "contact_apex_recovery_progress_term"
                        ),
                        "contact_apex_potential_term": reward_terms.get("contact_apex_potential_term"),
                        "contact_apex_progress_easy_next_ball_gate": info.get(
                            "contact_apex_progress_easy_next_ball_gate"
                        ),
                        "contact_lateral_stability_term": reward_terms.get("contact_lateral_stability_term"),
                        "contact_racket_outward_velocity_penalty": reward_terms.get(
                            "contact_racket_outward_velocity_penalty"
                        ),
                        "applied_action_norm": info.get("applied_action_norm"),
                        "applied_action_normalized_norm": info.get("applied_action_normalized_norm"),
                        "applied_position_action_norm": info.get("applied_position_action_norm"),
                        "applied_tilt_action_norm": info.get("applied_tilt_action_norm"),
                        "applied_action_0_radial": (
                            None if applied_action is None or applied_action.size <= 0 else float(applied_action[0])
                        ),
                        "applied_action_1_tangent": (
                            None if applied_action is None or applied_action.size <= 1 else float(applied_action[1])
                        ),
                        "applied_action_2_z": (
                            None if applied_action is None or applied_action.size <= 2 else float(applied_action[2])
                        ),
                        "applied_action_3_tilt_pitch": (
                            None if applied_action is None or applied_action.size <= 3 else float(applied_action[3])
                        ),
                        "applied_action_4_tilt_roll": (
                            None if applied_action is None or applied_action.size <= 4 else float(applied_action[4])
                        ),
                        "applied_action_5_vz_scale": (
                            None if applied_action is None or applied_action.size <= 5 else float(applied_action[5])
                        ),
                        "applied_action_6_outgoing_x_residual": (
                            None if applied_action is None or applied_action.size <= 6 else float(applied_action[6])
                        ),
                        "applied_action_7_outgoing_y_residual": (
                            None if applied_action is None or applied_action.size <= 7 else float(applied_action[7])
                        ),
                        "applied_action_8_racket_vz_residual": (
                            None if applied_action is None or applied_action.size <= 8 else float(applied_action[8])
                        ),
                        "applied_action_9_trajectory_tilt_scale": (
                            None if applied_action is None or applied_action.size <= 9 else float(applied_action[9])
                        ),
                        "applied_action_10_centering_tilt_scale": (
                            None if applied_action is None or applied_action.size <= 10 else float(applied_action[10])
                        ),
                        "applied_action_11_racket_vx_residual": (
                            None if applied_action is None or applied_action.size <= 11 else float(applied_action[11])
                        ),
                        "applied_action_12_racket_vy_residual": (
                            None if applied_action is None or applied_action.size <= 12 else float(applied_action[12])
                        ),
                        "applied_action_13_target_apex_z_residual": (
                            None if applied_action is None or applied_action.size <= 13 else float(applied_action[13])
                        ),
                        "applied_action_14_strike_plane_z_residual": (
                            None if applied_action is None or applied_action.size <= 14 else float(applied_action[14])
                        ),
                        "applied_action_15_tracking_vx_residual": (
                            None if applied_action is None or applied_action.size <= 15 else float(applied_action[15])
                        ),
                        "applied_action_16_tracking_vy_residual": (
                            None if applied_action is None or applied_action.size <= 16 else float(applied_action[16])
                        ),
                        "contact_frame_vz_scale": info.get("contact_frame_vz_scale"),
                        "contact_frame_outgoing_x_residual_action": info.get(
                            "contact_frame_outgoing_x_residual_action"
                        ),
                        "contact_frame_outgoing_y_residual_action": info.get(
                            "contact_frame_outgoing_y_residual_action"
                        ),
                        "contact_frame_racket_vz_residual_action": info.get(
                            "contact_frame_racket_vz_residual_action"
                        ),
                        "contact_frame_racket_x_residual_action": info.get(
                            "contact_frame_racket_x_residual_action"
                        ),
                        "contact_frame_racket_y_residual_action": info.get(
                            "contact_frame_racket_y_residual_action"
                        ),
                        "contact_frame_tracking_x_residual_action": info.get(
                            "contact_frame_tracking_x_residual_action"
                        ),
                        "contact_frame_tracking_y_residual_action": info.get(
                            "contact_frame_tracking_y_residual_action"
                        ),
                        "contact_frame_trajectory_tilt_scale": info.get("contact_frame_trajectory_tilt_scale"),
                        "contact_frame_centering_tilt_scale": info.get("contact_frame_centering_tilt_scale"),
                        "contact_frame_target_apex_z_residual_action": info.get(
                            "contact_frame_target_apex_z_residual_action"
                        ),
                        "contact_frame_strike_plane_z_residual_action": info.get(
                            "contact_frame_strike_plane_z_residual_action"
                        ),
                        "tracking_strike_plane_offset": info.get("tracking_strike_plane_offset"),
                        "contact_ball_position_x": contact_ball_position_x,
                        "contact_ball_position_y": contact_ball_position_y,
                        "contact_ball_position_z": contact_ball_position_z,
                        "desired_outgoing_velocity_x": info.get("desired_outgoing_velocity_x"),
                        "desired_outgoing_velocity_y": info.get("desired_outgoing_velocity_y"),
                        "desired_outgoing_velocity_z": info.get("desired_outgoing_velocity_z"),
                        "actual_outgoing_velocity_x": info.get("actual_outgoing_velocity_x"),
                        "actual_outgoing_velocity_y": info.get("actual_outgoing_velocity_y"),
                        "actual_outgoing_velocity_z": info.get("actual_outgoing_velocity_z"),
                        "outgoing_velocity_error_norm": info.get("outgoing_velocity_error_norm"),
                        "outgoing_velocity_xy_error": info.get("outgoing_velocity_xy_error"),
                        "outgoing_velocity_z_error": info.get("outgoing_velocity_z_error"),
                        "desired_time_to_apex": info.get("desired_time_to_apex"),
                        "desired_target_x": info.get("desired_outgoing_target_x"),
                        "desired_target_y": info.get("desired_outgoing_target_y"),
                        "predicted_apex_x_from_actual_velocity": info.get(
                            "predicted_apex_x_from_actual_velocity"
                        ),
                        "predicted_apex_y_from_actual_velocity": info.get(
                            "predicted_apex_y_from_actual_velocity"
                        ),
                        "predicted_apex_xy_error": info.get("predicted_apex_xy_error"),
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
                        "controller_anchor_z": (
                            None if controller_anchor is None else float(controller_anchor[2])
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
                        "racket_lateral_speed": info.get("contact_racket_lateral_speed"),
                        "contact_racket_outward_speed": info.get("contact_racket_outward_speed"),
                        "racket_speed_norm": info.get("contact_racket_speed_norm"),
                        "target_velocity_x": (
                            None if info.get("target_velocity") is None else float(info["target_velocity"][0])
                        ),
                        "target_velocity_y": (
                            None if info.get("target_velocity") is None else float(info["target_velocity"][1])
                        ),
                        "target_velocity_z": (
                            None if info.get("target_velocity") is None else float(info["target_velocity"][2])
                        ),
                        "contact_frame_controller_desired_velocity_x": (
                            None
                            if info.get("contact_frame_controller_desired_velocity") is None
                            else float(info["contact_frame_controller_desired_velocity"][0])
                        ),
                        "contact_frame_controller_desired_velocity_y": (
                            None
                            if info.get("contact_frame_controller_desired_velocity") is None
                            else float(info["contact_frame_controller_desired_velocity"][1])
                        ),
                        "contact_frame_controller_desired_velocity_z": (
                            None
                            if info.get("contact_frame_controller_desired_velocity") is None
                            else float(info["contact_frame_controller_desired_velocity"][2])
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
                        "contact_frame_lateral_brake_velocity_x": (
                            None
                            if info.get("contact_frame_lateral_brake_velocity") is None
                            else float(info["contact_frame_lateral_brake_velocity"][0])
                        ),
                        "contact_frame_lateral_brake_velocity_y": (
                            None
                            if info.get("contact_frame_lateral_brake_velocity") is None
                            else float(info["contact_frame_lateral_brake_velocity"][1])
                        ),
                        "contact_frame_planner_active": info.get("contact_frame_planner_active"),
                        "contact_frame_strike_hold_active": info.get("contact_frame_strike_hold_active"),
                        "controller_body_clearance_active": info.get("controller_body_clearance_active"),
                        "contact_frame_apex_lift": info.get("contact_frame_apex_lift"),
                        "contact_frame_low_apex_recovery_lift": info.get(
                            "contact_frame_low_apex_recovery_lift"
                        ),
                        "contact_frame_low_apex_recovery_velocity": info.get(
                            "contact_frame_low_apex_recovery_velocity"
                        ),
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
                        "contact_frame_planner_contact_target_x": (
                            None
                            if info.get("contact_frame_planner_contact_target_xy") is None
                            else float(info["contact_frame_planner_contact_target_xy"][0])
                        ),
                        "contact_frame_planner_contact_target_y": (
                            None
                            if info.get("contact_frame_planner_contact_target_xy") is None
                            else float(info["contact_frame_planner_contact_target_xy"][1])
                        ),
                        "contact_frame_planner_target_apex_z": info.get("contact_frame_planner_target_apex_z"),
                        "contact_frame_planner_resolved_target_apex_z": info.get(
                            "contact_frame_planner_resolved_target_apex_z"
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
                        "racket_face_normal_x": (
                            None if racket_face_normal is None else float(racket_face_normal[0])
                        ),
                        "racket_face_normal_y": (
                            None if racket_face_normal is None else float(racket_face_normal[1])
                        ),
                        "racket_face_normal_z": (
                            None if racket_face_normal is None else float(racket_face_normal[2])
                        ),
                        "xy_alignment_error": info.get("contact_xy_alignment_error"),
                        "contact_ball_height_above_racket": info.get("contact_ball_height_above_racket"),
                        "projected_contact_apex_height_above_racket": info.get(
                            "projected_contact_apex_height_above_racket"
                        ),
                        "last_projected_contact_apex_height_above_racket": info.get(
                            "last_projected_contact_apex_height_above_racket"
                        ),
                        "last_contact_apex_shortfall": info.get("last_contact_apex_shortfall"),
                        "target_ball_height_above_racket": float(env.base_env.target_ball_height),
                        "target_tilt_0": (
                            None if info.get("target_tilt") is None else float(np.asarray(info["target_tilt"])[0])
                        ),
                        "target_tilt_1": (
                            None if info.get("target_tilt") is None else float(np.asarray(info["target_tilt"])[1])
                        ),
                    }
                    contact_row.update(next_intercept_metrics)
                    contact_row.update(contact_quality_metrics)
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
            stable_cycle_count = int(info.get("stable_cycle_count", useful_bounce_count))
            failure_reason = info.get("failure_reason")
            if failure_reason is None:
                failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
            failure_counts[str(failure_reason)] += 1
            robot_body_contact_name = info.get("robot_body_contact_name")
            if str(failure_reason) == "robot_body_contact":
                robot_body_contact_counts[str(robot_body_contact_name or "unknown")] += 1
            returns.append(episode_return)
            useful_bounces.append(useful_bounce_count)
            stable_cycles.append(stable_cycle_count)
            episode_rows.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "steps": step_count,
                    "contact_count": contact_count,
                    "first_contact_step": first_contact_step,
                    "useful_bounces": useful_bounce_count,
                    "stable_cycles": stable_cycle_count,
                    "failure_reason": failure_reason,
                    "robot_body_contact_name": robot_body_contact_name,
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
    stable_cycle_array = np.asarray(stable_cycles, dtype=float)
    summary = {
        "model_path": str(model_path.resolve()),
        "run_name": run_name,
        "episodes": args.episodes,
        "env_config": env.training_config() if False else env_kwargs,
        "episode_step_limit": episode_step_limit,
        "mean_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "mean_stable_cycles": float(stable_cycle_array.mean()) if stable_cycle_array.size else 0.0,
        "max_stable_cycles": int(stable_cycle_array.max()) if stable_cycle_array.size else 0,
        "episodes_with_one_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 1.0)) if bounce_array.size else 0,
        "one_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 1.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_two_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 2.0)) if bounce_array.size else 0,
        "two_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 2.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_three_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 3.0)) if bounce_array.size else 0,
        "three_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 3.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_ten_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 10.0)) if bounce_array.size else 0,
        "ten_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 10.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_twenty_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 20.0)) if bounce_array.size else 0,
        "twenty_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 20.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_thirty_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 30.0)) if bounce_array.size else 0,
        "thirty_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 30.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_one_or_more_stable_cycles": (
            int(np.count_nonzero(stable_cycle_array >= 1.0)) if stable_cycle_array.size else 0
        ),
        "one_or_more_stable_cycle_rate": (
            float(np.count_nonzero(stable_cycle_array >= 1.0) / stable_cycle_array.size)
            if stable_cycle_array.size else 0.0
        ),
        "episodes_with_two_or_more_stable_cycles": (
            int(np.count_nonzero(stable_cycle_array >= 2.0)) if stable_cycle_array.size else 0
        ),
        "two_or_more_stable_cycle_rate": (
            float(np.count_nonzero(stable_cycle_array >= 2.0) / stable_cycle_array.size)
            if stable_cycle_array.size else 0.0
        ),
        "episodes_with_three_or_more_stable_cycles": (
            int(np.count_nonzero(stable_cycle_array >= 3.0)) if stable_cycle_array.size else 0
        ),
        "three_or_more_stable_cycle_rate": (
            float(np.count_nonzero(stable_cycle_array >= 3.0) / stable_cycle_array.size)
            if stable_cycle_array.size else 0.0
        ),
        "episodes_with_ten_or_more_stable_cycles": (
            int(np.count_nonzero(stable_cycle_array >= 10.0)) if stable_cycle_array.size else 0
        ),
        "ten_or_more_stable_cycle_rate": (
            float(np.count_nonzero(stable_cycle_array >= 10.0) / stable_cycle_array.size)
            if stable_cycle_array.size else 0.0
        ),
        "episodes_with_twenty_or_more_stable_cycles": (
            int(np.count_nonzero(stable_cycle_array >= 20.0)) if stable_cycle_array.size else 0
        ),
        "twenty_or_more_stable_cycle_rate": (
            float(np.count_nonzero(stable_cycle_array >= 20.0) / stable_cycle_array.size)
            if stable_cycle_array.size else 0.0
        ),
        "episodes_with_thirty_or_more_stable_cycles": (
            int(np.count_nonzero(stable_cycle_array >= 30.0)) if stable_cycle_array.size else 0
        ),
        "thirty_or_more_stable_cycle_rate": (
            float(np.count_nonzero(stable_cycle_array >= 30.0) / stable_cycle_array.size)
            if stable_cycle_array.size else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "robot_body_contact_counts": dict(robot_body_contact_counts),
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
        "episode_next_intercept_summary": summarize_episode_next_intercepts(
            episode_rows,
            contact_rows,
        ),
        "episode_outgoing_velocity_summary": summarize_episode_outgoing_velocities(
            episode_rows,
            contact_rows,
        ),
        "terminal_contact_summary": summarize_terminal_contacts(contact_rows),
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
