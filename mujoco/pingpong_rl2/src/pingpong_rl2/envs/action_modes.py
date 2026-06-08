from __future__ import annotations

# action_mode는 PPO가 내는 flat action vector를 어떻게 해석할지 정하는 schema다.
# 뒤쪽으로 갈수록 hand-coded contact-frame planner가 만든 기본 목표에 policy residual 슬롯을 더 많이 붙인다.
# 자세한 슬롯/흐름 설명은 docs/control_flow_and_terms.md를 참조한다.

ACTION_MODES = (
    "position",
    "position_strike",
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
CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_residual",
    "position_contact_frame_velocity_tilt_residual",
    "position_contact_frame_velocity_tilt_lateral_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
CONTACT_FRAME_TILT_SCALE_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_residual",
    "position_contact_frame_velocity_tilt_lateral_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
CONTACT_FRAME_LATERAL_VELOCITY_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_lateral_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
CONTACT_FRAME_APEX_TIMING_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_lateral_apex_residual",
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
CONTACT_FRAME_TRACKING_RESIDUAL_ACTION_MODES = (
    "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
)
CONTACT_FRAME_ACTION_MODES = (
    "position_contact_frame",
    *CONTACT_FRAME_VELOCITY_RESIDUAL_ACTION_MODES,
)
TILT_ACTION_MODES = (
    "position_tilt",
    "position_strike_tilt",
    "position_strike_tilt_lift",
    *CONTACT_FRAME_ACTION_MODES,
)
STRIKE_CONTRACT_ACTION_MODES = (
    "position_strike",
    "position_strike_tilt",
    "position_strike_tilt_lift",
    *CONTACT_FRAME_ACTION_MODES,
)
TILT_SLICE_3_TO_5_ACTION_MODES = ("position_strike_tilt_lift", *CONTACT_FRAME_ACTION_MODES)
CONTACT_ORACLE_MODES = ("none", "desired_outgoing_velocity")
RETURN_TARGET_XY_SOURCES = ("controller_anchor", "racket_home", "racket_position", "target_position")
DESIRED_OUTGOING_XY_MODES = ("next_intercept", "apex")
RESET_XY_SAMPLING_MODES = ("square", "disk")
