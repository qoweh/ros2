from __future__ import annotations

import argparse
from pathlib import Path

from pingpong_rl2.training.cli_config import values_equal
from pingpong_rl2.training.presets import _ENV_PRESETS, _PRESET_MANAGED_ARG_DEFAULTS, _TILT_PROFILES
from pingpong_rl2.utils import resolve_input_path

def apply_env_preset(args: argparse.Namespace) -> str:
    # preset 값은 사용자가 직접 바꾼 CLI 값과 충돌하지 않을 때만 args에 주입한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/training/presets.py:1
    if args.preset is None:
        return "manual"
    if args.preset not in _ENV_PRESETS:
        raise ValueError(f"Unknown preset: {args.preset!r}.")

    preset_values = _ENV_PRESETS[args.preset]
    for arg_name, preset_value in preset_values.items():
        current_value = getattr(args, arg_name)
        default_value = _PRESET_MANAGED_ARG_DEFAULTS[arg_name]
        if values_equal(current_value, default_value):
            setattr(args, arg_name, preset_value)
            continue
        if not values_equal(current_value, preset_value):
            raise ValueError(
                f"--preset {args.preset!r} conflicts with explicit --{arg_name.replace('_', '-')}={current_value!r}."
            )
    return str(args.preset)



def resolve_tilt_profile(args: argparse.Namespace) -> str:
    # tilt action이 없는 mode는 tilt 관련 CLI 옵션을 비활성 프로필로 고정한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/action_modes.py:1
    if args.action_mode not in (
        "position_tilt",
        "position_strike_tilt",
        "position_strike_tilt_lift",
        "position_contact_frame",
        "position_contact_frame_velocity_residual",
        "position_contact_frame_velocity_tilt_residual",
        "position_contact_frame_velocity_tilt_lateral_residual",
        "position_contact_frame_velocity_tilt_lateral_apex_residual",
        "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
    ):
        if args.tracking_during_contact_scale is None:
            args.tracking_during_contact_scale = 0.0
        return "disabled"

    profile_name = "early" if args.tilt_profile == "auto" else args.tilt_profile
    if profile_name == "custom":
        if args.tracking_during_contact_scale is None:
            args.tracking_during_contact_scale = 0.0
        return profile_name

    # named profile은 action limit, target tilt limit, tilt regularization 기본값을 함께 채운다.
    # LINK: pingpong_rl2/src/pingpong_rl2/training/presets.py:213
    profile = _TILT_PROFILES[profile_name]
    if args.tilt_action_limit is None:
        args.tilt_action_limit = float(profile["tilt_action_limit"])
    if args.target_tilt_limit is None:
        args.target_tilt_limit = tuple(profile["target_tilt_limit"])
    if args.tilt_angle_penalty_weight is None:
        args.tilt_angle_penalty_weight = float(profile["tilt_angle_penalty_weight"])
    if args.tilt_action_delta_penalty_weight is None:
        args.tilt_action_delta_penalty_weight = float(profile["tilt_action_delta_penalty_weight"])
    if args.tracking_during_contact_scale is None:
        args.tracking_during_contact_scale = 0.0
    return profile_name


def tilt_limit_ratio(args: argparse.Namespace) -> float | None:
    # policy log_std 스케일링에서 tilt action limit과 실제 target tilt limit의 비율을 참고한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/training/policy_init.py:31
    if (
        args.action_mode
        not in (
            "position_tilt",
            "position_strike_tilt",
            "position_strike_tilt_lift",
            "position_contact_frame",
            "position_contact_frame_velocity_residual",
            "position_contact_frame_velocity_tilt_residual",
            "position_contact_frame_velocity_tilt_lateral_residual",
            "position_contact_frame_velocity_tilt_lateral_apex_residual",
            "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
        )
        or args.tilt_action_limit is None
        or args.target_tilt_limit is None
    ):
        return None
    return float(args.tilt_action_limit / max(min(args.target_tilt_limit), 1.0e-6))


def env_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    # 모든 환경이 공유하는 reset 분포, 목표 높이, action mode 기본 인자를 먼저 구성한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:53
    env_kwargs: dict[str, object] = {
        "action_mode": args.action_mode,
        "ball_height": args.ball_height,
        "target_ball_height": args.ball_height if args.target_ball_height is None else args.target_ball_height,
        "max_episode_steps": args.max_episode_steps,
        "reset_ball_height_range": args.reset_ball_height_range,
        "reset_ball_height_bounds": (
            None if args.reset_ball_height_bounds is None else tuple(args.reset_ball_height_bounds)
        ),
        "reset_xy_range": args.reset_xy_range,
        "reset_xy_sampling": args.reset_xy_sampling,
        "reset_velocity_xy_range": args.reset_velocity_xy_range,
        "reset_velocity_z_range": tuple(args.reset_velocity_z_range),
        "reset_ball_angular_velocity_range": args.reset_ball_angular_velocity_range,
        "target_offset_low": tuple(args.target_offset_low),
        "target_offset_high": tuple(args.target_offset_high),
        "success_velocity_threshold": args.success_velocity_threshold,
    }
    if args.scene_path is not None:
        env_kwargs["scene_path"] = str(resolve_input_path(Path(args.scene_path)))
    # action limit과 공통 보상/성공 조건은 명시된 값만 env_kwargs에 추가한다.
    if args.lateral_action_limit is not None:
        env_kwargs["lateral_action_limit"] = args.lateral_action_limit
    if args.vertical_action_limit is not None:
        env_kwargs["vertical_action_limit"] = args.vertical_action_limit
    if args.tilt_action_limit is not None:
        env_kwargs["tilt_action_limit"] = args.tilt_action_limit
    if args.followup_lift_action_limit is not None:
        env_kwargs["followup_lift_action_limit"] = args.followup_lift_action_limit
    if args.tracking_during_contact_scale is not None:
        env_kwargs["tracking_during_contact_scale"] = args.tracking_during_contact_scale
    if args.useful_contact_outgoing_x_penalty_weight is not None:
        env_kwargs["useful_contact_outgoing_x_penalty_weight"] = args.useful_contact_outgoing_x_penalty_weight
    if args.desired_outgoing_ball_velocity_x is not None:
        env_kwargs["desired_outgoing_ball_velocity_x"] = args.desired_outgoing_ball_velocity_x
    if args.keepup_target_xy_offset is not None:
        env_kwargs["keepup_target_xy_offset"] = tuple(args.keepup_target_xy_offset)
    if args.useful_contact_return_target_xy_reward_weight is not None:
        env_kwargs["useful_contact_return_target_xy_reward_weight"] = (
            args.useful_contact_return_target_xy_reward_weight
        )
    if args.return_target_xy_source is not None:
        env_kwargs["return_target_xy_source"] = args.return_target_xy_source
    if args.return_target_xy_tolerance is not None:
        env_kwargs["return_target_xy_tolerance"] = args.return_target_xy_tolerance
    if args.tilt_angle_penalty_weight is not None:
        env_kwargs["tilt_angle_penalty_weight"] = args.tilt_angle_penalty_weight
    if args.tilt_action_delta_penalty_weight is not None:
        env_kwargs["tilt_action_delta_penalty_weight"] = args.tilt_action_delta_penalty_weight
    if args.target_tilt_limit is not None:
        env_kwargs["target_tilt_limit"] = tuple(args.target_tilt_limit)
    if args.target_pitch_range is not None:
        env_kwargs["target_pitch_range"] = tuple(args.target_pitch_range)
    if args.initial_target_tilt is not None:
        env_kwargs["initial_target_tilt"] = tuple(args.initial_target_tilt)
    if args.strike_tilt_assist_limit is not None:
        env_kwargs["strike_tilt_assist_limit"] = tuple(args.strike_tilt_assist_limit)
    if args.strike_tilt_assist_deadband is not None:
        env_kwargs["strike_tilt_assist_deadband"] = args.strike_tilt_assist_deadband
    if args.strike_tilt_ramp_pitch is not None:
        env_kwargs["strike_tilt_ramp_pitch"] = args.strike_tilt_ramp_pitch
    if args.strike_tilt_ramp_xy_tolerance is not None:
        env_kwargs["strike_tilt_ramp_xy_tolerance"] = args.strike_tilt_ramp_xy_tolerance
    # follow-up strike 설정은 첫 성공 이후의 반복 keep-up 보정에만 영향을 준다.
    if args.followup_strike_target_tilt is not None:
        env_kwargs["followup_strike_target_tilt"] = tuple(args.followup_strike_target_tilt)
    if args.followup_strike_contact_offset_ratio is not None:
        env_kwargs["followup_strike_contact_offset_ratio"] = args.followup_strike_contact_offset_ratio
    if args.followup_strike_contact_offset_max is not None:
        env_kwargs["followup_strike_contact_offset_max"] = args.followup_strike_contact_offset_max
    if args.followup_strike_lift_boost is not None:
        env_kwargs["followup_strike_lift_boost"] = args.followup_strike_lift_boost
    if args.contact_frame_base_strike_z_boost is not None:
        env_kwargs["contact_frame_base_strike_z_boost"] = args.contact_frame_base_strike_z_boost
    if args.contact_frame_base_strike_z_offset is not None:
        env_kwargs["contact_frame_base_strike_z_offset"] = args.contact_frame_base_strike_z_offset
    if args.contact_frame_base_strike_time_horizon is not None:
        env_kwargs["contact_frame_base_strike_time_horizon"] = args.contact_frame_base_strike_time_horizon
    if args.contact_frame_base_tilt_residual is not None:
        env_kwargs["contact_frame_base_tilt_residual"] = tuple(args.contact_frame_base_tilt_residual)
    # contact-frame 계열은 world delta 대신 접촉 좌표계/속도 residual로 policy action을 해석한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:2144
    if args.contact_frame_apex_lift_gain is not None:
        env_kwargs["contact_frame_apex_lift_gain"] = args.contact_frame_apex_lift_gain
    if args.contact_frame_apex_lift_max is not None:
        env_kwargs["contact_frame_apex_lift_max"] = args.contact_frame_apex_lift_max
    if args.contact_frame_apex_lift_reference_velocity_z is not None:
        env_kwargs["contact_frame_apex_lift_reference_velocity_z"] = args.contact_frame_apex_lift_reference_velocity_z
    if args.contact_frame_apex_lift_restitution is not None:
        env_kwargs["contact_frame_apex_lift_restitution"] = args.contact_frame_apex_lift_restitution
    if args.contact_frame_low_apex_recovery_lift_gain is not None:
        env_kwargs["contact_frame_low_apex_recovery_lift_gain"] = args.contact_frame_low_apex_recovery_lift_gain
    if args.contact_frame_low_apex_recovery_lift_max is not None:
        env_kwargs["contact_frame_low_apex_recovery_lift_max"] = args.contact_frame_low_apex_recovery_lift_max
    if args.contact_frame_low_apex_recovery_velocity_gain is not None:
        env_kwargs["contact_frame_low_apex_recovery_velocity_gain"] = (
            args.contact_frame_low_apex_recovery_velocity_gain
        )
    if args.contact_frame_low_apex_recovery_velocity_max is not None:
        env_kwargs["contact_frame_low_apex_recovery_velocity_max"] = (
            args.contact_frame_low_apex_recovery_velocity_max
        )
    if args.contact_frame_velocity_lead_gain is not None:
        env_kwargs["contact_frame_velocity_lead_gain"] = args.contact_frame_velocity_lead_gain
    if args.contact_frame_velocity_lead_max is not None:
        env_kwargs["contact_frame_velocity_lead_max"] = args.contact_frame_velocity_lead_max
    if args.contact_frame_velocity_target_gain is not None:
        env_kwargs["contact_frame_velocity_target_gain"] = args.contact_frame_velocity_target_gain
    if args.contact_frame_velocity_target_max is not None:
        env_kwargs["contact_frame_velocity_target_max"] = args.contact_frame_velocity_target_max
    if args.contact_frame_velocity_scale_action_limit is not None:
        env_kwargs["contact_frame_velocity_scale_action_limit"] = args.contact_frame_velocity_scale_action_limit
    if args.contact_frame_outgoing_xy_action_limit is not None:
        env_kwargs["contact_frame_outgoing_xy_action_limit"] = args.contact_frame_outgoing_xy_action_limit
    if args.contact_frame_racket_vz_action_limit is not None:
        env_kwargs["contact_frame_racket_vz_action_limit"] = args.contact_frame_racket_vz_action_limit
    if args.contact_frame_racket_xy_action_limit is not None:
        env_kwargs["contact_frame_racket_xy_action_limit"] = args.contact_frame_racket_xy_action_limit
    if args.contact_frame_tilt_scale_action_limit is not None:
        env_kwargs["contact_frame_tilt_scale_action_limit"] = args.contact_frame_tilt_scale_action_limit
    if args.contact_frame_target_apex_z_action_limit is not None:
        env_kwargs["contact_frame_target_apex_z_action_limit"] = args.contact_frame_target_apex_z_action_limit
    if args.contact_frame_strike_plane_z_action_limit is not None:
        env_kwargs["contact_frame_strike_plane_z_action_limit"] = args.contact_frame_strike_plane_z_action_limit
    if args.contact_frame_tracking_xy_action_limit is not None:
        env_kwargs["contact_frame_tracking_xy_action_limit"] = args.contact_frame_tracking_xy_action_limit
    if args.contact_frame_intercept_velocity_gain is not None:
        env_kwargs["contact_frame_intercept_velocity_gain"] = args.contact_frame_intercept_velocity_gain
    if args.contact_frame_intercept_velocity_max is not None:
        env_kwargs["contact_frame_intercept_velocity_max"] = args.contact_frame_intercept_velocity_max
    if args.contact_frame_intercept_velocity_time_floor is not None:
        env_kwargs["contact_frame_intercept_velocity_time_floor"] = args.contact_frame_intercept_velocity_time_floor
    if args.contact_frame_planner_enabled:
        env_kwargs["contact_frame_planner_enabled"] = True
    if not args.contact_frame_planner_hold_during_descent:
        env_kwargs["contact_frame_planner_hold_during_descent"] = False
    if args.contact_frame_planner_min_intercept_time is not None:
        env_kwargs["contact_frame_planner_min_intercept_time"] = args.contact_frame_planner_min_intercept_time
    if args.contact_frame_planner_max_intercept_time is not None:
        env_kwargs["contact_frame_planner_max_intercept_time"] = args.contact_frame_planner_max_intercept_time
    if args.contact_frame_planner_target_apex_z_offset is not None:
        env_kwargs["contact_frame_planner_target_apex_z_offset"] = args.contact_frame_planner_target_apex_z_offset
    if args.contact_frame_planner_contact_offset_ratio is not None:
        env_kwargs["contact_frame_planner_contact_offset_ratio"] = args.contact_frame_planner_contact_offset_ratio
    if args.contact_frame_planner_contact_offset_max is not None:
        env_kwargs["contact_frame_planner_contact_offset_max"] = args.contact_frame_planner_contact_offset_max
    # planner 이후 strike hold/followthrough/brake/tilt 보정은 contact-frame controller의 보조 명령이다.
    if args.contact_frame_strike_hold_time is not None:
        env_kwargs["contact_frame_strike_hold_time"] = args.contact_frame_strike_hold_time
    if args.contact_frame_strike_hold_min_readiness is not None:
        env_kwargs["contact_frame_strike_hold_min_readiness"] = args.contact_frame_strike_hold_min_readiness
    if args.contact_frame_followthrough_gain is not None:
        env_kwargs["contact_frame_followthrough_gain"] = args.contact_frame_followthrough_gain
    if args.contact_frame_followthrough_time is not None:
        env_kwargs["contact_frame_followthrough_time"] = args.contact_frame_followthrough_time
    if args.contact_frame_followthrough_max is not None:
        env_kwargs["contact_frame_followthrough_max"] = args.contact_frame_followthrough_max
    if args.contact_frame_lateral_brake_gain is not None:
        env_kwargs["contact_frame_lateral_brake_gain"] = args.contact_frame_lateral_brake_gain
    if args.contact_frame_lateral_brake_max is not None:
        env_kwargs["contact_frame_lateral_brake_max"] = args.contact_frame_lateral_brake_max
    if args.contact_frame_lateral_brake_radius is not None:
        env_kwargs["contact_frame_lateral_brake_radius"] = args.contact_frame_lateral_brake_radius
    if args.contact_frame_trajectory_tilt_gain is not None:
        env_kwargs["contact_frame_trajectory_tilt_gain"] = args.contact_frame_trajectory_tilt_gain
    if args.contact_frame_trajectory_tilt_limit is not None:
        env_kwargs["contact_frame_trajectory_tilt_limit"] = tuple(args.contact_frame_trajectory_tilt_limit)
    if args.contact_frame_trajectory_tilt_deadband is not None:
        env_kwargs["contact_frame_trajectory_tilt_deadband"] = args.contact_frame_trajectory_tilt_deadband
    if args.contact_frame_tilt_ramp_time is not None:
        env_kwargs["contact_frame_tilt_ramp_time"] = args.contact_frame_tilt_ramp_time
    # controller override는 MuJoCo arm target을 실제 joint target으로 바꾸는 하위 제어기에 전달된다.
    # LINK: pingpong_rl2/src/pingpong_rl2/controllers/ee_pose_controller.py:1
    if args.controller_orientation_gain is not None:
        env_kwargs["controller_orientation_gain"] = args.controller_orientation_gain
    if args.controller_max_orientation_step is not None:
        env_kwargs["controller_max_orientation_step"] = args.controller_max_orientation_step
    if args.controller_velocity_gain is not None:
        env_kwargs["controller_velocity_gain"] = args.controller_velocity_gain
    if args.controller_velocity_feedback_gain is not None:
        env_kwargs["controller_velocity_feedback_gain"] = args.controller_velocity_feedback_gain
    if args.controller_max_velocity_step is not None:
        env_kwargs["controller_max_velocity_step"] = args.controller_max_velocity_step
    if args.controller_nullspace_posture_gain is not None:
        env_kwargs["controller_nullspace_posture_gain"] = args.controller_nullspace_posture_gain
    if args.controller_nullspace_posture_max_step is not None:
        env_kwargs["controller_nullspace_posture_max_step"] = args.controller_nullspace_posture_max_step
    if args.controller_nullspace_posture_target is not None:
        env_kwargs["controller_nullspace_posture_target"] = tuple(args.controller_nullspace_posture_target)
    if args.controller_body_clearance_gain is not None:
        env_kwargs["controller_body_clearance_gain"] = args.controller_body_clearance_gain
    if args.controller_body_clearance_margin is not None:
        env_kwargs["controller_body_clearance_margin"] = args.controller_body_clearance_margin
    if args.controller_body_clearance_vertical_margin is not None:
        env_kwargs["controller_body_clearance_vertical_margin"] = args.controller_body_clearance_vertical_margin
    if args.controller_body_clearance_max_step is not None:
        env_kwargs["controller_body_clearance_max_step"] = args.controller_body_clearance_max_step
    if args.controller_body_clearance_body_names is not None:
        env_kwargs["controller_body_clearance_body_names"] = tuple(args.controller_body_clearance_body_names)
    if args.tracking_strike_plane_offset is not None:
        env_kwargs["tracking_strike_plane_offset"] = args.tracking_strike_plane_offset
    if args.contact_frame_centering_tilt_limit is not None:
        env_kwargs["contact_frame_centering_tilt_limit"] = tuple(args.contact_frame_centering_tilt_limit)
    if args.contact_frame_centering_tilt_radius is not None:
        env_kwargs["contact_frame_centering_tilt_radius"] = args.contact_frame_centering_tilt_radius
    if args.contact_frame_centering_tilt_deadband is not None:
        env_kwargs["contact_frame_centering_tilt_deadband"] = args.contact_frame_centering_tilt_deadband
    if args.contact_frame_action_penalty_weight is not None:
        env_kwargs["contact_frame_action_penalty_weight"] = args.contact_frame_action_penalty_weight
    # 접촉 품질/다음 인터셉트 보상은 분석 스크립트가 보는 contact row 지표와 이름을 맞춘다.
    # LINK: pingpong_rl2/src/pingpong_rl2/analysis/rebound_metrics.py:54
    if args.next_intercept_xy_error_penalty_weight is not None:
        env_kwargs["next_intercept_xy_error_penalty_weight"] = args.next_intercept_xy_error_penalty_weight
    if args.post_contact_lateral_velocity_penalty_weight is not None:
        env_kwargs["post_contact_lateral_velocity_penalty_weight"] = (
            args.post_contact_lateral_velocity_penalty_weight
        )
    if args.contact_xy_error_penalty_weight is not None:
        env_kwargs["contact_xy_error_penalty_weight"] = args.contact_xy_error_penalty_weight
    if args.contact_racket_lateral_velocity_penalty_weight is not None:
        env_kwargs["contact_racket_lateral_velocity_penalty_weight"] = (
            args.contact_racket_lateral_velocity_penalty_weight
        )
    if args.contact_racket_lateral_velocity_tolerance is not None:
        env_kwargs["contact_racket_lateral_velocity_tolerance"] = args.contact_racket_lateral_velocity_tolerance
    if args.contact_racket_outward_velocity_penalty_weight is not None:
        env_kwargs["contact_racket_outward_velocity_penalty_weight"] = (
            args.contact_racket_outward_velocity_penalty_weight
        )
    if args.contact_racket_outward_velocity_tolerance is not None:
        env_kwargs["contact_racket_outward_velocity_tolerance"] = args.contact_racket_outward_velocity_tolerance
    if args.max_contact_racket_lateral_speed_for_success is not None:
        env_kwargs["max_contact_racket_lateral_speed_for_success"] = (
            args.max_contact_racket_lateral_speed_for_success
        )
    if args.nonuseful_contact_penalty_weight is not None:
        env_kwargs["nonuseful_contact_penalty_weight"] = args.nonuseful_contact_penalty_weight
    if args.contact_apex_under_target_penalty_weight is not None:
        env_kwargs["contact_apex_under_target_penalty_weight"] = (
            args.contact_apex_under_target_penalty_weight
        )
    if args.contact_apex_progress_reward_weight is not None:
        env_kwargs["contact_apex_progress_reward_weight"] = args.contact_apex_progress_reward_weight
    if args.contact_apex_recovery_progress_reward_weight is not None:
        env_kwargs["contact_apex_recovery_progress_reward_weight"] = (
            args.contact_apex_recovery_progress_reward_weight
        )
    if args.gate_contact_apex_progress_by_easy_next_ball:
        env_kwargs["gate_contact_apex_progress_by_easy_next_ball"] = True
    if args.contact_apex_progress_min_easy_next_ball_score is not None:
        env_kwargs["contact_apex_progress_min_easy_next_ball_score"] = (
            args.contact_apex_progress_min_easy_next_ball_score
        )
    if args.contact_apex_potential_reward_weight is not None:
        env_kwargs["contact_apex_potential_reward_weight"] = args.contact_apex_potential_reward_weight
    if args.contact_apex_potential_gamma is not None:
        env_kwargs["contact_apex_potential_gamma"] = args.contact_apex_potential_gamma
    if args.contact_apex_potential_cap is not None:
        env_kwargs["contact_apex_potential_cap"] = args.contact_apex_potential_cap
    if args.contact_lateral_stability_reward_weight is not None:
        env_kwargs["contact_lateral_stability_reward_weight"] = args.contact_lateral_stability_reward_weight
    if args.contact_lateral_stability_speed_tolerance is not None:
        env_kwargs["contact_lateral_stability_speed_tolerance"] = (
            args.contact_lateral_stability_speed_tolerance
        )
    if args.contact_lateral_stability_xy_tolerance is not None:
        env_kwargs["contact_lateral_stability_xy_tolerance"] = args.contact_lateral_stability_xy_tolerance
    if args.contact_lateral_stability_min_apex_ratio is not None:
        env_kwargs["contact_lateral_stability_min_apex_ratio"] = args.contact_lateral_stability_min_apex_ratio
    if args.stable_contact_reward_weight is not None:
        env_kwargs["stable_contact_reward_weight"] = args.stable_contact_reward_weight
    if args.stable_contact_min_apex_ratio is not None:
        env_kwargs["stable_contact_min_apex_ratio"] = args.stable_contact_min_apex_ratio
    if args.stable_cycle_reward_weight is not None:
        env_kwargs["stable_cycle_reward_weight"] = args.stable_cycle_reward_weight
    if args.stable_cycle_reward_cap != 4 or args.stable_cycle_reward_weight is not None:
        env_kwargs["stable_cycle_reward_cap"] = args.stable_cycle_reward_cap
    if args.stable_cycle_min_easy_next_ball_score is not None:
        env_kwargs["stable_cycle_min_easy_next_ball_score"] = args.stable_cycle_min_easy_next_ball_score
    if args.post_contact_return_assist_weight is not None:
        env_kwargs["post_contact_return_assist_weight"] = args.post_contact_return_assist_weight
    if args.post_contact_return_max_intercept_time is not None:
        env_kwargs["post_contact_return_max_intercept_time"] = args.post_contact_return_max_intercept_time
    if args.post_contact_return_z_offset is not None:
        env_kwargs["post_contact_return_z_offset"] = args.post_contact_return_z_offset
    if not args.post_contact_return_predict_during_rise:
        env_kwargs["post_contact_return_predict_during_rise"] = False
    if args.next_intercept_reachable_bonus_weight is not None:
        env_kwargs["next_intercept_reachable_bonus_weight"] = args.next_intercept_reachable_bonus_weight
    if args.easy_next_ball_reward_weight is not None:
        env_kwargs["easy_next_ball_reward_weight"] = args.easy_next_ball_reward_weight
    if args.next_intercept_success_radius is not None:
        env_kwargs["next_intercept_success_radius"] = args.next_intercept_success_radius
    if args.easy_next_ball_xy_radius is not None:
        env_kwargs["easy_next_ball_xy_radius"] = args.easy_next_ball_xy_radius
    if args.require_reachable_next_intercept_for_success:
        env_kwargs["require_reachable_next_intercept_for_success"] = True
    if args.require_apex_height_window_for_success:
        env_kwargs["require_apex_height_window_for_success"] = True
    if args.min_easy_next_ball_score_for_success is not None:
        env_kwargs["min_easy_next_ball_score_for_success"] = args.min_easy_next_ball_score_for_success
    if args.gate_nonuseful_easy_next_ball_by_apex:
        env_kwargs["gate_nonuseful_easy_next_ball_by_apex"] = True
    if args.terminate_on_nonuseful_contact:
        env_kwargs["terminate_on_nonuseful_contact"] = True
    if args.terminate_on_low_apex_contact:
        env_kwargs["terminate_on_low_apex_contact"] = True
    if args.low_apex_contact_height_threshold is not None:
        env_kwargs["low_apex_contact_height_threshold"] = args.low_apex_contact_height_threshold
    if args.low_apex_contact_grace_count != 0:
        env_kwargs["low_apex_contact_grace_count"] = args.low_apex_contact_grace_count
    if args.trajectory_match_reward_weight is not None:
        env_kwargs["trajectory_match_reward_weight"] = args.trajectory_match_reward_weight
    if args.trajectory_error_penalty_weight is not None:
        env_kwargs["trajectory_error_penalty_weight"] = args.trajectory_error_penalty_weight
    if args.reward_contact_quality_on_any_upward_contact:
        env_kwargs["reward_contact_quality_on_any_upward_contact"] = True
    if args.next_intercept_max_time is not None:
        env_kwargs["next_intercept_max_time"] = args.next_intercept_max_time
    # observation 확장 플래그는 observation_layout과 실제 observation 벡터 길이를 함께 바꾼다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/observation_layout.py:1
    if args.include_velocity_domain_observation:
        env_kwargs["include_velocity_domain_observation"] = True
    if args.include_task_phase_observation:
        env_kwargs["include_task_phase_observation"] = True
    if args.include_contact_context_observation:
        env_kwargs["include_contact_context_observation"] = True
    if args.include_next_intercept_observation:
        env_kwargs["include_next_intercept_observation"] = True
    if args.include_desired_outgoing_velocity_observation:
        env_kwargs["include_desired_outgoing_velocity_observation"] = True
    if args.desired_outgoing_xy_mode is not None:
        env_kwargs["desired_outgoing_xy_mode"] = args.desired_outgoing_xy_mode
    return env_kwargs
