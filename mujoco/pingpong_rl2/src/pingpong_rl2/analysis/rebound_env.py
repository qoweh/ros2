from __future__ import annotations

import argparse

# rebound 분석에서 학습 summary의 env_config 위에 바로 덮어쓸 수 있는 단순 스칼라 옵션들이다.
# LINK: mujoco/pingpong_rl2/scripts/run_ppo_rebound_analysis.py:214
DIRECT_ENV_OVERRIDE_FIELDS = (
    "min_easy_next_ball_score_for_success",
    "post_contact_return_z_offset",
    "contact_frame_velocity_target_gain",
    "contact_frame_velocity_target_max",
    "contact_frame_velocity_scale_action_limit",
    "contact_frame_outgoing_xy_action_limit",
    "contact_frame_racket_vz_action_limit",
    "contact_frame_racket_xy_action_limit",
    "contact_frame_tilt_scale_action_limit",
    "contact_frame_target_apex_z_action_limit",
    "contact_frame_strike_plane_z_action_limit",
    "tracking_strike_plane_offset",
    "contact_frame_intercept_velocity_gain",
    "contact_frame_intercept_velocity_max",
    "contact_frame_intercept_velocity_time_floor",
    "contact_frame_planner_min_intercept_time",
    "contact_frame_planner_max_intercept_time",
    "contact_frame_planner_target_apex_z_offset",
    "contact_frame_planner_contact_offset_ratio",
    "contact_frame_planner_contact_offset_max",
    "contact_frame_lateral_brake_gain",
    "contact_frame_lateral_brake_max",
    "contact_frame_lateral_brake_radius",
    "controller_velocity_gain",
    "controller_velocity_feedback_gain",
    "controller_max_velocity_step",
    "controller_nullspace_posture_gain",
    "controller_nullspace_posture_max_step",
    "controller_body_clearance_gain",
    "controller_body_clearance_margin",
    "controller_body_clearance_vertical_margin",
    "controller_body_clearance_max_step",
    "contact_frame_trajectory_tilt_gain",
    "contact_frame_trajectory_tilt_deadband",
    "contact_frame_centering_tilt_radius",
    "contact_frame_centering_tilt_deadband",
    "next_intercept_success_radius",
    "easy_next_ball_xy_radius",
    "next_intercept_xy_error_penalty_weight",
    "post_contact_lateral_velocity_penalty_weight",
    "contact_xy_error_penalty_weight",
    "contact_racket_outward_velocity_penalty_weight",
    "contact_racket_outward_velocity_tolerance",
    "nonuseful_contact_penalty_weight",
    "contact_apex_potential_reward_weight",
    "contact_apex_potential_gamma",
    "contact_apex_potential_cap",
    "contact_lateral_stability_min_apex_ratio",
    "stable_contact_min_apex_ratio",
)
TUPLE_ENV_OVERRIDE_FIELDS = (
    # list로 들어온 CLI nargs 값은 KeepUp env가 기대하는 tuple 형태로 바꿔 전달한다.
    "keepup_target_xy_offset",
    "controller_body_clearance_body_names",
    "contact_frame_trajectory_tilt_limit",
    "contact_frame_centering_tilt_limit",
)
TRUE_FLAG_ENV_OVERRIDES = (
    # store_true 계열은 켜진 값만 env_kwargs에 써서 학습 summary 기본값을 보존한다.
    "require_reachable_next_intercept_for_success",
    "contact_frame_planner_enabled",
    "terminate_on_nonuseful_contact",
)
FALSE_FLAG_ENV_OVERRIDES = (
    # disable 플래그는 False일 때만 명시 override로 간주한다.
    "post_contact_return_predict_during_rise",
    "contact_frame_planner_hold_during_descent",
)


def apply_rebound_env_overrides(
    args: argparse.Namespace,
    env_kwargs: dict[str, object],
) -> dict[str, object]:
    # 분석 CLI에서 명시한 반동/접촉 관련 옵션만 저장 모델의 env_kwargs 위에 덧씌운다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py:186
    for field_name in DIRECT_ENV_OVERRIDE_FIELDS:
        value = getattr(args, field_name)
        if value is not None:
            env_kwargs[field_name] = value

    for field_name in TUPLE_ENV_OVERRIDE_FIELDS:
        value = getattr(args, field_name)
        if value is not None:
            env_kwargs[field_name] = tuple(value)

    for field_name in TRUE_FLAG_ENV_OVERRIDES:
        if getattr(args, field_name):
            env_kwargs[field_name] = True

    for field_name in FALSE_FLAG_ENV_OVERRIDES:
        if not getattr(args, field_name):
            env_kwargs[field_name] = False

    return env_kwargs
