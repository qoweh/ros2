from __future__ import annotations

import math

import numpy as np

_APEX_TARGET_CHOICES = (
    "controller_anchor",
    "racket_home",
    "racket_position",
    "target_position",
)

_NEXT_INTERCEPT_MAX_TIME = 2.0
_EASY_NEXT_BALL_TARGET_TIME = 0.45
_EASY_NEXT_BALL_TIME_TOLERANCE = 0.30
_EASY_NEXT_BALL_TARGET_DESCENDING_SPEED = 1.25
_EASY_NEXT_BALL_MAX_LATERAL_SPEED = 1.0
_EASY_NEXT_BALL_SOFT_SPEED_LIMIT = 3.0
_UNLIMITED_ANALYSIS_STEP_LIMIT = 3_600


def solve_ballistic_times(
    start_z: float,
    velocity_z: float,
    target_z: float,
    gravity_z: float,
    *,
    max_time: float,
) -> list[float]:
    # 등가속도 z축 운동 방정식을 풀어 target_z를 통과하는 양의 시간 후보를 구한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:2562
    quadratic_a = 0.5 * gravity_z
    quadratic_b = velocity_z
    quadratic_c = start_z - target_z
    candidate_times: list[float] = []
    if abs(quadratic_a) < 1.0e-9:
        if abs(quadratic_b) > 1.0e-9:
            candidate_times.append(-quadratic_c / quadratic_b)
    else:
        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        if discriminant >= 0.0:
            sqrt_discriminant = float(np.sqrt(discriminant))
            denominator = 2.0 * quadratic_a
            candidate_times.extend(
                [
                    (-quadratic_b - sqrt_discriminant) / denominator,
                    (-quadratic_b + sqrt_discriminant) / denominator,
                ]
            )
    return sorted(time_value for time_value in candidate_times if 1.0e-6 <= time_value <= max_time)


def compute_next_intercept_metrics(
    *,
    contact_ball_position: np.ndarray | None,
    ball_velocity: np.ndarray | None,
    controller_anchor_position: np.ndarray | None,
    gravity_z: float,
    strike_plane_offset: float,
    strike_zone_xy_radius: float,
) -> dict[str, object]:
    # contact 직후 속도로 다음 descending strike plane 교차점과 easy-next-ball 점수를 추정한다.
    # LINK: pingpong_rl2/scripts/run_ppo_rebound_analysis.py:245
    metrics: dict[str, object] = {
        "next_intercept_target_z": None,
        "next_intercept_time": None,
        "next_intercept_x": None,
        "next_intercept_y": None,
        "next_intercept_xy_error": None,
        "next_intercept_reachable": None,
        "next_intercept_vertical_speed": None,
        "next_intercept_speed_norm": None,
        "next_intercept_xy_score": None,
        "next_intercept_time_score": None,
        "next_intercept_descending_score": None,
        "next_intercept_lateral_speed_penalty": None,
        "next_intercept_excessive_speed_penalty": None,
        "next_intercept_recovery_distance_penalty": None,
        "easy_next_ball_score": None,
    }
    if contact_ball_position is None or ball_velocity is None or controller_anchor_position is None:
        return metrics

    target_z = float(controller_anchor_position[2] + strike_plane_offset)
    metrics["next_intercept_target_z"] = target_z
    candidate_times = solve_ballistic_times(
        float(contact_ball_position[2]),
        float(ball_velocity[2]),
        target_z,
        gravity_z,
        max_time=_NEXT_INTERCEPT_MAX_TIME,
    )
    if not candidate_times:
        return metrics

    # 두 번 교차할 수 있으면 뒤쪽 시간을 사용해 다시 내려오는 다음 intercept를 본다.
    next_intercept_time = max(candidate_times)
    next_intercept_xy = np.asarray(contact_ball_position[:2] + next_intercept_time * ball_velocity[:2], dtype=float)
    next_intercept_xy_error = float(np.linalg.norm(next_intercept_xy - controller_anchor_position[:2]))
    next_intercept_reachable = next_intercept_xy_error <= strike_zone_xy_radius
    next_intercept_vertical_speed = float(ball_velocity[2] + gravity_z * next_intercept_time)
    next_intercept_speed_norm = float(
        np.linalg.norm(np.array([ball_velocity[0], ball_velocity[1], next_intercept_vertical_speed], dtype=float))
    )
    lateral_speed = float(math.hypot(float(ball_velocity[0]), float(ball_velocity[1])))
    xy_score = max(1.0 - next_intercept_xy_error / max(strike_zone_xy_radius, 1.0e-6), 0.0)
    time_score = max(
        1.0 - abs(next_intercept_time - _EASY_NEXT_BALL_TARGET_TIME) / _EASY_NEXT_BALL_TIME_TOLERANCE,
        0.0,
    )
    descending_score = max(
        1.0
        - abs(abs(next_intercept_vertical_speed) - _EASY_NEXT_BALL_TARGET_DESCENDING_SPEED)
        / _EASY_NEXT_BALL_TARGET_DESCENDING_SPEED,
        0.0,
    )
    lateral_speed_penalty = float(np.clip(lateral_speed / _EASY_NEXT_BALL_MAX_LATERAL_SPEED, 0.0, 1.0))
    excessive_speed_penalty = float(
        np.clip(
            max(next_intercept_speed_norm - _EASY_NEXT_BALL_SOFT_SPEED_LIMIT, 0.0)
            / _EASY_NEXT_BALL_SOFT_SPEED_LIMIT,
            0.0,
            1.0,
        )
    )
    recovery_distance_penalty = float(
        np.clip(next_intercept_xy_error / max(1.5 * strike_zone_xy_radius, 1.0e-6), 0.0, 1.0)
    )
    easy_next_ball_score = (
        # 점수는 XY 근접성, 타이밍, 하강 속도에서 보상을 받고 lateral/과속/복귀 거리에서 감점된다.
        xy_score
        + 0.75 * time_score
        + 0.5 * descending_score
        - 0.5 * lateral_speed_penalty
        - 0.25 * excessive_speed_penalty
        - 0.5 * recovery_distance_penalty
    )
    metrics.update(
        {
            "next_intercept_time": float(next_intercept_time),
            "next_intercept_x": float(next_intercept_xy[0]),
            "next_intercept_y": float(next_intercept_xy[1]),
            "next_intercept_xy_error": next_intercept_xy_error,
            "next_intercept_reachable": bool(next_intercept_reachable),
            "next_intercept_vertical_speed": next_intercept_vertical_speed,
            "next_intercept_speed_norm": next_intercept_speed_norm,
            "next_intercept_xy_score": float(xy_score),
            "next_intercept_time_score": float(time_score),
            "next_intercept_descending_score": float(descending_score),
            "next_intercept_lateral_speed_penalty": lateral_speed_penalty,
            "next_intercept_excessive_speed_penalty": excessive_speed_penalty,
            "next_intercept_recovery_distance_penalty": recovery_distance_penalty,
            "easy_next_ball_score": float(easy_next_ball_score),
        }
    )
    return metrics


def compute_contact_quality_metrics(
    *,
    ball_velocity: np.ndarray | None,
    racket_velocity: np.ndarray | None,
    racket_face_normal: np.ndarray | None,
) -> dict[str, object]:
    # racket face normal 기준 상대속도를 normal/tangential 성분으로 나눠 접촉 품질을 본다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py:365
    metrics: dict[str, object] = {
        "contact_relative_speed_norm": None,
        "contact_normal_relative_speed": None,
        "contact_tangential_relative_speed": None,
        "contact_tangential_relative_ratio": None,
    }
    if ball_velocity is None or racket_velocity is None or racket_face_normal is None:
        return metrics

    normal = np.asarray(racket_face_normal, dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-9:
        return metrics
    normal = normal / normal_norm
    relative_velocity = np.asarray(ball_velocity, dtype=float) - np.asarray(racket_velocity, dtype=float)
    relative_speed_norm = float(np.linalg.norm(relative_velocity))
    normal_relative_speed = float(np.dot(relative_velocity, normal))
    tangential_velocity = relative_velocity - normal_relative_speed * normal
    tangential_relative_speed = float(np.linalg.norm(tangential_velocity))
    tangential_relative_ratio = tangential_relative_speed / max(relative_speed_norm, 1.0e-6)
    metrics.update(
        {
            "contact_relative_speed_norm": relative_speed_norm,
            "contact_normal_relative_speed": normal_relative_speed,
            "contact_tangential_relative_speed": tangential_relative_speed,
            "contact_tangential_relative_ratio": tangential_relative_ratio,
        }
    )
    return metrics


def apex_target_xy_candidates(
    *,
    info: dict[str, object],
    racket_home_xy: np.ndarray,
    racket_position_xy: np.ndarray,
) -> dict[str, np.ndarray]:
    # projected apex XY error를 여러 target 기준으로 비교할 수 있게 후보 좌표를 모은다.
    # LINK: pingpong_rl2/scripts/run_ppo_rebound_analysis.py:297
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

