from __future__ import annotations

from pingpong_rl2.envs.action_modes import TILT_ACTION_MODES

POSITION_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("joint_positions", 7),
    ("joint_velocities", 7),
    ("racket_position", 3),
    ("racket_velocity", 3),
    ("target_position", 3),
    ("ball_position", 3),
    ("ball_velocity", 3),
    ("ball_relative_position", 3),
    ("predicted_intercept_relative_xy", 2),
    ("predicted_intercept_time", 1),
)

TASK_PHASE_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("phase_one_hot", 4),
)

CONTACT_CONTEXT_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("time_since_contact", 1),
    ("successful_bounce_count_clipped", 1),
)

NEXT_INTERCEPT_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("next_intercept_relative_xy", 2),
    ("next_intercept_time", 1),
    ("next_intercept_reachable", 1),
    ("next_intercept_recovery_distance", 1),
    ("next_intercept_recovery_readiness", 1),
)

DESIRED_OUTGOING_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("desired_outgoing_velocity", 3),
)

VELOCITY_DOMAIN_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("relative_velocity", 3),
    ("racket_face_normal", 3),
)

POSITION_TILT_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("target_tilt", 2),
)


def build_observation_layout(
    action_mode: str,
    include_velocity_domain_observation: bool,
    include_task_phase_observation: bool,
    include_contact_context_observation: bool,
    include_next_intercept_observation: bool,
    include_desired_outgoing_velocity_observation: bool,
) -> tuple[tuple[tuple[str, int], ...], dict[str, slice], int]:
    components = POSITION_OBSERVATION_COMPONENTS
    if include_task_phase_observation:
        components = components + TASK_PHASE_OBSERVATION_COMPONENTS
    if include_contact_context_observation:
        components = components + CONTACT_CONTEXT_OBSERVATION_COMPONENTS
    if include_next_intercept_observation:
        components = components + NEXT_INTERCEPT_OBSERVATION_COMPONENTS
    if include_desired_outgoing_velocity_observation:
        components = components + DESIRED_OUTGOING_OBSERVATION_COMPONENTS
    if include_velocity_domain_observation:
        components = components + VELOCITY_DOMAIN_OBSERVATION_COMPONENTS
    if action_mode in TILT_ACTION_MODES:
        if not include_velocity_domain_observation:
            components = components + (("racket_face_normal", 3),)
        components = components + POSITION_TILT_OBSERVATION_COMPONENTS

    observation_slices: dict[str, slice] = {}
    observation_offset = 0
    for component_name, component_size in components:
        observation_slices[component_name] = slice(observation_offset, observation_offset + component_size)
        observation_offset += component_size
    return components, observation_slices, observation_offset
