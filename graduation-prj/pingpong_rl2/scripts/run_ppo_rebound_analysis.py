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

_NEXT_INTERCEPT_MAX_TIME = 2.0
_EASY_NEXT_BALL_TARGET_TIME = 0.45
_EASY_NEXT_BALL_TIME_TOLERANCE = 0.30
_EASY_NEXT_BALL_TARGET_DESCENDING_SPEED = 1.25
_EASY_NEXT_BALL_MAX_LATERAL_SPEED = 1.0
_EASY_NEXT_BALL_SOFT_SPEED_LIMIT = 3.0


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
    parser.add_argument("--contact-frame-tilt-scale-action-limit", type=float, default=None)
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
    parser.add_argument("--nonuseful-contact-penalty-weight", type=float, default=None)
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


def solve_ballistic_times(
    start_z: float,
    velocity_z: float,
    target_z: float,
    gravity_z: float,
    *,
    max_time: float,
) -> list[float]:
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
            "stable_cycle_contact_rate": 0.0,
            "useful_contact_stable_cycle_rate": 0.0,
            "mean_ball_lateral_speed": 0.0,
            "mean_ball_lateral_to_vertical_ratio": 0.0,
            "mean_projected_contact_apex_height_above_racket": 0.0,
            "median_projected_contact_apex_height_above_racket": 0.0,
            "useful_contact_mean_projected_contact_apex_height_above_racket": 0.0,
            "upward_contact_count": 0,
            "upward_contact_projected_apex_below_0_16_rate": 0.0,
            "upward_contact_projected_apex_below_0_20_rate": 0.0,
            "upward_contact_projected_apex_below_target_rate": 0.0,
            "mean_projected_apex_xy_error": 0.0,
            "useful_contact_mean_projected_apex_xy_error": 0.0,
            "mean_outgoing_velocity_error_norm": 0.0,
            "useful_contact_mean_outgoing_velocity_error_norm": 0.0,
            "mean_outgoing_velocity_xy_error": 0.0,
            "useful_contact_mean_outgoing_velocity_xy_error": 0.0,
            "mean_outgoing_velocity_z_error": 0.0,
            "useful_contact_mean_outgoing_velocity_z_error": 0.0,
            "mean_predicted_apex_xy_error": 0.0,
            "useful_contact_mean_predicted_apex_xy_error": 0.0,
            "mean_next_intercept_time": 0.0,
            "useful_contact_mean_next_intercept_time": 0.0,
            "mean_next_intercept_xy_error": 0.0,
            "useful_contact_mean_next_intercept_xy_error": 0.0,
            "next_intercept_reachable_rate": 0.0,
            "useful_contact_next_intercept_reachable_rate": 0.0,
            "mean_easy_next_ball_score": 0.0,
            "useful_contact_mean_easy_next_ball_score": 0.0,
            "mean_contact_relative_speed_norm": 0.0,
            "useful_contact_mean_contact_relative_speed_norm": 0.0,
            "mean_contact_tangential_relative_speed": 0.0,
            "useful_contact_mean_contact_tangential_relative_speed": 0.0,
            "mean_contact_tangential_relative_ratio": 0.0,
            "useful_contact_mean_contact_tangential_relative_ratio": 0.0,
            "mean_contact_apex_progress_easy_next_ball_gate": 0.0,
            "mean_contact_lateral_stability_term": 0.0,
            "useful_contact_mean_contact_lateral_stability_term": 0.0,
            "mean_applied_action_normalized_norm": 0.0,
            "useful_contact_mean_applied_action_normalized_norm": 0.0,
            "mean_applied_tilt_action_norm": 0.0,
            "useful_contact_mean_applied_tilt_action_norm": 0.0,
            "mean_consecutive_stable_cycle_count": 0.0,
            "max_consecutive_stable_cycle_count": 0,
            "selected_apex_target": selected_apex_target,
            "next_intercept_target_source": "controller_anchor",
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

    def true_rate(key: str, rows: list[dict[str, object]]) -> float:
        values = [bool(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    useful_rows = [row for row in contact_rows if bool(row.get("is_useful_contact", False))]
    stable_cycle_rows = [row for row in contact_rows if bool(row.get("stable_cycle_observed", False))]
    ball_lateral_speed = float_series("ball_lateral_speed", contact_rows)
    ball_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", contact_rows)
    useful_lateral_speed = float_series("ball_lateral_speed", useful_rows)
    useful_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", useful_rows)
    projected_apex_height = float_series("projected_contact_apex_height_above_racket", contact_rows)
    useful_projected_apex_height = float_series("projected_contact_apex_height_above_racket", useful_rows)
    projected_apex_xy_error = float_series(selected_error_key, contact_rows)
    useful_projected_apex_xy_error = float_series(selected_error_key, useful_rows)
    outgoing_velocity_error_norm = float_series("outgoing_velocity_error_norm", contact_rows)
    useful_outgoing_velocity_error_norm = float_series("outgoing_velocity_error_norm", useful_rows)
    outgoing_velocity_xy_error = float_series("outgoing_velocity_xy_error", contact_rows)
    useful_outgoing_velocity_xy_error = float_series("outgoing_velocity_xy_error", useful_rows)
    outgoing_velocity_z_error = float_series("outgoing_velocity_z_error", contact_rows)
    useful_outgoing_velocity_z_error = float_series("outgoing_velocity_z_error", useful_rows)
    predicted_apex_xy_error = float_series("predicted_apex_xy_error", contact_rows)
    useful_predicted_apex_xy_error = float_series("predicted_apex_xy_error", useful_rows)
    next_intercept_time = float_series("next_intercept_time", contact_rows)
    useful_next_intercept_time = float_series("next_intercept_time", useful_rows)
    next_intercept_xy_error = float_series("next_intercept_xy_error", contact_rows)
    useful_next_intercept_xy_error = float_series("next_intercept_xy_error", useful_rows)
    easy_next_ball_score = float_series("easy_next_ball_score", contact_rows)
    useful_easy_next_ball_score = float_series("easy_next_ball_score", useful_rows)
    contact_relative_speed_norm = float_series("contact_relative_speed_norm", contact_rows)
    useful_contact_relative_speed_norm = float_series("contact_relative_speed_norm", useful_rows)
    tangential_relative_speed = float_series("contact_tangential_relative_speed", contact_rows)
    useful_tangential_relative_speed = float_series("contact_tangential_relative_speed", useful_rows)
    tangential_relative_ratio = float_series("contact_tangential_relative_ratio", contact_rows)
    useful_tangential_relative_ratio = float_series("contact_tangential_relative_ratio", useful_rows)
    consecutive_stable_cycle_count = float_series("consecutive_stable_cycle_count", contact_rows)
    last_contact_apex_shortfall = float_series("last_contact_apex_shortfall", contact_rows)
    recovery_lift = float_series("contact_frame_low_apex_recovery_lift", contact_rows)
    recovery_velocity = float_series("contact_frame_low_apex_recovery_velocity", contact_rows)
    recovery_progress_term = float_series("contact_apex_recovery_progress_term", contact_rows)
    apex_progress_gate = float_series("contact_apex_progress_easy_next_ball_gate", contact_rows)
    lateral_stability_term = float_series("contact_lateral_stability_term", contact_rows)
    useful_lateral_stability_term = float_series("contact_lateral_stability_term", useful_rows)
    action_normalized_norm = float_series("applied_action_normalized_norm", contact_rows)
    useful_action_normalized_norm = float_series("applied_action_normalized_norm", useful_rows)
    tilt_action_norm = float_series("applied_tilt_action_norm", contact_rows)
    useful_tilt_action_norm = float_series("applied_tilt_action_norm", useful_rows)
    upward_rows = [
        row
        for row in contact_rows
        if row.get("actual_outgoing_velocity_z") is not None and float(row["actual_outgoing_velocity_z"]) > 0.5
    ]
    upward_projected_apex_height = float_series("projected_contact_apex_height_above_racket", upward_rows)
    target_apex_height_values = float_series("target_ball_height_above_racket", contact_rows)
    target_apex_height = float(target_apex_height_values.mean()) if target_apex_height_values.size else None

    def below_rate(values: np.ndarray, threshold: float) -> float:
        if not values.size:
            return 0.0
        return float(np.count_nonzero(values < threshold) / values.size)

    summary = {
        "total_contacts": len(contact_rows),
        "useful_contact_rate": len(useful_rows) / len(contact_rows),
        "stable_cycle_contact_rate": len(stable_cycle_rows) / len(contact_rows),
        "useful_contact_stable_cycle_rate": true_rate("stable_cycle_observed", useful_rows),
        "selected_apex_target": selected_apex_target,
        "next_intercept_target_source": "controller_anchor",
        "mean_ball_lateral_speed": float(ball_lateral_speed.mean()) if ball_lateral_speed.size else 0.0,
        "mean_ball_lateral_to_vertical_ratio": (
            float(ball_lateral_ratio.mean()) if ball_lateral_ratio.size else 0.0
        ),
        "mean_projected_contact_apex_height_above_racket": (
            float(projected_apex_height.mean()) if projected_apex_height.size else 0.0
        ),
        "median_projected_contact_apex_height_above_racket": (
            float(np.median(projected_apex_height)) if projected_apex_height.size else 0.0
        ),
        "useful_contact_mean_projected_contact_apex_height_above_racket": (
            float(useful_projected_apex_height.mean()) if useful_projected_apex_height.size else 0.0
        ),
        "upward_contact_count": len(upward_rows),
        "upward_contact_projected_apex_below_0_16_rate": below_rate(upward_projected_apex_height, 0.16),
        "upward_contact_projected_apex_below_0_20_rate": below_rate(upward_projected_apex_height, 0.20),
        "upward_contact_projected_apex_below_target_rate": (
            0.0 if target_apex_height is None else below_rate(upward_projected_apex_height, target_apex_height)
        ),
        "mean_last_contact_apex_shortfall": (
            float(last_contact_apex_shortfall.mean()) if last_contact_apex_shortfall.size else 0.0
        ),
        "mean_contact_frame_low_apex_recovery_lift": float(recovery_lift.mean()) if recovery_lift.size else 0.0,
        "max_contact_frame_low_apex_recovery_lift": float(recovery_lift.max()) if recovery_lift.size else 0.0,
        "mean_contact_frame_low_apex_recovery_velocity": (
            float(recovery_velocity.mean()) if recovery_velocity.size else 0.0
        ),
        "max_contact_frame_low_apex_recovery_velocity": (
            float(recovery_velocity.max()) if recovery_velocity.size else 0.0
        ),
        "mean_contact_apex_recovery_progress_term": (
            float(recovery_progress_term.mean()) if recovery_progress_term.size else 0.0
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
        "mean_outgoing_velocity_error_norm": (
            float(outgoing_velocity_error_norm.mean()) if outgoing_velocity_error_norm.size else 0.0
        ),
        "useful_contact_mean_outgoing_velocity_error_norm": (
            float(useful_outgoing_velocity_error_norm.mean()) if useful_outgoing_velocity_error_norm.size else 0.0
        ),
        "mean_outgoing_velocity_xy_error": (
            float(outgoing_velocity_xy_error.mean()) if outgoing_velocity_xy_error.size else 0.0
        ),
        "useful_contact_mean_outgoing_velocity_xy_error": (
            float(useful_outgoing_velocity_xy_error.mean()) if useful_outgoing_velocity_xy_error.size else 0.0
        ),
        "mean_outgoing_velocity_z_error": (
            float(outgoing_velocity_z_error.mean()) if outgoing_velocity_z_error.size else 0.0
        ),
        "useful_contact_mean_outgoing_velocity_z_error": (
            float(useful_outgoing_velocity_z_error.mean()) if useful_outgoing_velocity_z_error.size else 0.0
        ),
        "mean_predicted_apex_xy_error": (
            float(predicted_apex_xy_error.mean()) if predicted_apex_xy_error.size else 0.0
        ),
        "useful_contact_mean_predicted_apex_xy_error": (
            float(useful_predicted_apex_xy_error.mean()) if useful_predicted_apex_xy_error.size else 0.0
        ),
        "mean_next_intercept_time": float(next_intercept_time.mean()) if next_intercept_time.size else 0.0,
        "useful_contact_mean_next_intercept_time": (
            float(useful_next_intercept_time.mean()) if useful_next_intercept_time.size else 0.0
        ),
        "mean_next_intercept_xy_error": (
            float(next_intercept_xy_error.mean()) if next_intercept_xy_error.size else 0.0
        ),
        "useful_contact_mean_next_intercept_xy_error": (
            float(useful_next_intercept_xy_error.mean()) if useful_next_intercept_xy_error.size else 0.0
        ),
        "next_intercept_reachable_rate": true_rate("next_intercept_reachable", contact_rows),
        "useful_contact_next_intercept_reachable_rate": true_rate("next_intercept_reachable", useful_rows),
        "mean_easy_next_ball_score": (
            float(easy_next_ball_score.mean()) if easy_next_ball_score.size else 0.0
        ),
        "useful_contact_mean_easy_next_ball_score": (
            float(useful_easy_next_ball_score.mean()) if useful_easy_next_ball_score.size else 0.0
        ),
        "mean_contact_relative_speed_norm": (
            float(contact_relative_speed_norm.mean()) if contact_relative_speed_norm.size else 0.0
        ),
        "useful_contact_mean_contact_relative_speed_norm": (
            float(useful_contact_relative_speed_norm.mean()) if useful_contact_relative_speed_norm.size else 0.0
        ),
        "mean_contact_tangential_relative_speed": (
            float(tangential_relative_speed.mean()) if tangential_relative_speed.size else 0.0
        ),
        "useful_contact_mean_contact_tangential_relative_speed": (
            float(useful_tangential_relative_speed.mean()) if useful_tangential_relative_speed.size else 0.0
        ),
        "mean_contact_tangential_relative_ratio": (
            float(tangential_relative_ratio.mean()) if tangential_relative_ratio.size else 0.0
        ),
        "useful_contact_mean_contact_tangential_relative_ratio": (
            float(useful_tangential_relative_ratio.mean()) if useful_tangential_relative_ratio.size else 0.0
        ),
        "mean_contact_apex_progress_easy_next_ball_gate": (
            float(apex_progress_gate.mean()) if apex_progress_gate.size else 0.0
        ),
        "mean_contact_lateral_stability_term": (
            float(lateral_stability_term.mean()) if lateral_stability_term.size else 0.0
        ),
        "useful_contact_mean_contact_lateral_stability_term": (
            float(useful_lateral_stability_term.mean()) if useful_lateral_stability_term.size else 0.0
        ),
        "mean_applied_action_normalized_norm": (
            float(action_normalized_norm.mean()) if action_normalized_norm.size else 0.0
        ),
        "useful_contact_mean_applied_action_normalized_norm": (
            float(useful_action_normalized_norm.mean()) if useful_action_normalized_norm.size else 0.0
        ),
        "mean_applied_tilt_action_norm": float(tilt_action_norm.mean()) if tilt_action_norm.size else 0.0,
        "useful_contact_mean_applied_tilt_action_norm": (
            float(useful_tilt_action_norm.mean()) if useful_tilt_action_norm.size else 0.0
        ),
        "mean_consecutive_stable_cycle_count": (
            float(consecutive_stable_cycle_count.mean()) if consecutive_stable_cycle_count.size else 0.0
        ),
        "max_consecutive_stable_cycle_count": (
            int(consecutive_stable_cycle_count.max()) if consecutive_stable_cycle_count.size else 0
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


def summarize_terminal_contacts(contact_rows: list[dict[str, object]]) -> dict[str, object]:
    if not contact_rows:
        return {
            "episodes_with_contacts": 0,
            "mean_terminal_projected_contact_apex_height_above_racket": 0.0,
            "median_terminal_projected_contact_apex_height_above_racket": 0.0,
            "mean_terminal_actual_outgoing_velocity_z": 0.0,
            "mean_terminal_desired_outgoing_velocity_z": 0.0,
            "mean_terminal_outgoing_velocity_z_error": 0.0,
            "terminal_upward_projected_apex_below_0_16_rate": 0.0,
            "terminal_upward_projected_apex_below_0_20_rate": 0.0,
        }

    last_contact_by_episode: dict[int, dict[str, object]] = {}
    for row in contact_rows:
        last_contact_by_episode[int(row["episode"])] = row
    terminal_rows = list(last_contact_by_episode.values())

    def float_series(key: str, rows: list[dict[str, object]]) -> np.ndarray:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return np.asarray(values, dtype=float)

    def below_rate(values: np.ndarray, threshold: float) -> float:
        if not values.size:
            return 0.0
        return float(np.count_nonzero(values < threshold) / values.size)

    terminal_apex_height = float_series("projected_contact_apex_height_above_racket", terminal_rows)
    terminal_actual_z = float_series("actual_outgoing_velocity_z", terminal_rows)
    terminal_desired_z = float_series("desired_outgoing_velocity_z", terminal_rows)
    terminal_z_error = float_series("outgoing_velocity_z_error", terminal_rows)
    terminal_upward_rows = [
        row
        for row in terminal_rows
        if row.get("actual_outgoing_velocity_z") is not None and float(row["actual_outgoing_velocity_z"]) > 0.5
    ]
    terminal_upward_apex_height = float_series("projected_contact_apex_height_above_racket", terminal_upward_rows)
    return {
        "episodes_with_contacts": len(terminal_rows),
        "mean_terminal_projected_contact_apex_height_above_racket": (
            float(terminal_apex_height.mean()) if terminal_apex_height.size else 0.0
        ),
        "median_terminal_projected_contact_apex_height_above_racket": (
            float(np.median(terminal_apex_height)) if terminal_apex_height.size else 0.0
        ),
        "mean_terminal_actual_outgoing_velocity_z": (
            float(terminal_actual_z.mean()) if terminal_actual_z.size else 0.0
        ),
        "mean_terminal_desired_outgoing_velocity_z": (
            float(terminal_desired_z.mean()) if terminal_desired_z.size else 0.0
        ),
        "mean_terminal_outgoing_velocity_z_error": (
            float(terminal_z_error.mean()) if terminal_z_error.size else 0.0
        ),
        "terminal_upward_projected_apex_below_0_16_rate": below_rate(terminal_upward_apex_height, 0.16),
        "terminal_upward_projected_apex_below_0_20_rate": below_rate(terminal_upward_apex_height, 0.20),
    }


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


def summarize_episode_next_intercepts(
    episode_rows: list[dict[str, object]],
    contact_rows: list[dict[str, object]],
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

    def group_values(metric_key: str, *, min_useful_bounces: int) -> list[float]:
        values: list[float] = []
        for episode, useful_bounce_count in episode_useful_bounces.items():
            if useful_bounce_count < min_useful_bounces:
                continue
            first_contact = first_contact_by_episode.get(episode)
            if first_contact is None or first_contact.get(metric_key) is None:
                continue
            values.append(float(first_contact[metric_key]))
        return values

    def group_values_max(metric_key: str, *, max_useful_bounces: int) -> list[float]:
        values: list[float] = []
        for episode, useful_bounce_count in episode_useful_bounces.items():
            if useful_bounce_count > max_useful_bounces:
                continue
            first_contact = first_contact_by_episode.get(episode)
            if first_contact is None or first_contact.get(metric_key) is None:
                continue
            values.append(float(first_contact[metric_key]))
        return values

    def reachable_rate(*, min_useful_bounces: int | None = None, max_useful_bounces: int | None = None) -> float | None:
        values: list[bool] = []
        for episode, useful_bounce_count in episode_useful_bounces.items():
            if min_useful_bounces is not None and useful_bounce_count < min_useful_bounces:
                continue
            if max_useful_bounces is not None and useful_bounce_count > max_useful_bounces:
                continue
            first_contact = first_contact_by_episode.get(episode)
            if first_contact is None or first_contact.get("next_intercept_reachable") is None:
                continue
            values.append(bool(first_contact["next_intercept_reachable"]))
        if not values:
            return None
        return float(sum(values) / len(values))

    first_contact_two_or_more_error = group_values("next_intercept_xy_error", min_useful_bounces=2)
    first_contact_fewer_than_two_error = group_values_max("next_intercept_xy_error", max_useful_bounces=1)
    first_contact_one_or_more_error = group_values("next_intercept_xy_error", min_useful_bounces=1)
    first_contact_zero_error = group_values_max("next_intercept_xy_error", max_useful_bounces=0)
    first_contact_two_or_more_score = group_values("easy_next_ball_score", min_useful_bounces=2)
    first_contact_fewer_than_two_score = group_values_max("easy_next_ball_score", max_useful_bounces=1)
    first_contact_one_or_more_score = group_values("easy_next_ball_score", min_useful_bounces=1)
    first_contact_zero_score = group_values_max("easy_next_ball_score", max_useful_bounces=0)

    mean_two_or_more_error = float(np.mean(first_contact_two_or_more_error)) if first_contact_two_or_more_error else None
    mean_fewer_than_two_error = (
        float(np.mean(first_contact_fewer_than_two_error)) if first_contact_fewer_than_two_error else None
    )
    mean_one_or_more_error = float(np.mean(first_contact_one_or_more_error)) if first_contact_one_or_more_error else None
    mean_zero_error = float(np.mean(first_contact_zero_error)) if first_contact_zero_error else None
    mean_two_or_more_score = float(np.mean(first_contact_two_or_more_score)) if first_contact_two_or_more_score else None
    mean_fewer_than_two_score = (
        float(np.mean(first_contact_fewer_than_two_score)) if first_contact_fewer_than_two_score else None
    )
    mean_one_or_more_score = float(np.mean(first_contact_one_or_more_score)) if first_contact_one_or_more_score else None
    mean_zero_score = float(np.mean(first_contact_zero_score)) if first_contact_zero_score else None

    return {
        "episodes_with_one_or_more_useful_bounces": sum(value >= 1 for value in episode_useful_bounces.values()),
        "episodes_with_two_or_more_useful_bounces": sum(value >= 2 for value in episode_useful_bounces.values()),
        "first_contact_mean_next_intercept_xy_error_two_or_more_useful_bounces": mean_two_or_more_error,
        "first_contact_mean_next_intercept_xy_error_fewer_than_two_useful_bounces": mean_fewer_than_two_error,
        "two_or_more_useful_bounces_next_intercept_xy_error_gap": (
            None
            if mean_two_or_more_error is None or mean_fewer_than_two_error is None
            else mean_two_or_more_error - mean_fewer_than_two_error
        ),
        "first_contact_mean_next_intercept_xy_error_one_or_more_useful_bounces": mean_one_or_more_error,
        "first_contact_mean_next_intercept_xy_error_zero_useful_bounces": mean_zero_error,
        "one_or_more_useful_bounces_next_intercept_xy_error_gap": (
            None
            if mean_one_or_more_error is None or mean_zero_error is None
            else mean_one_or_more_error - mean_zero_error
        ),
        "first_contact_mean_easy_next_ball_score_two_or_more_useful_bounces": mean_two_or_more_score,
        "first_contact_mean_easy_next_ball_score_fewer_than_two_useful_bounces": mean_fewer_than_two_score,
        "two_or_more_useful_bounces_easy_next_ball_score_gap": (
            None
            if mean_two_or_more_score is None or mean_fewer_than_two_score is None
            else mean_two_or_more_score - mean_fewer_than_two_score
        ),
        "first_contact_mean_easy_next_ball_score_one_or_more_useful_bounces": mean_one_or_more_score,
        "first_contact_mean_easy_next_ball_score_zero_useful_bounces": mean_zero_score,
        "one_or_more_useful_bounces_easy_next_ball_score_gap": (
            None
            if mean_one_or_more_score is None or mean_zero_score is None
            else mean_one_or_more_score - mean_zero_score
        ),
        "first_contact_next_intercept_reachable_rate_one_or_more_useful_bounces": reachable_rate(min_useful_bounces=1),
        "first_contact_next_intercept_reachable_rate_zero_useful_bounces": reachable_rate(max_useful_bounces=0),
    }


def summarize_episode_outgoing_velocities(
    episode_rows: list[dict[str, object]],
    contact_rows: list[dict[str, object]],
) -> dict[str, object]:
    episode_useful_bounces: dict[int, int] = {
        int(row["episode"]): int(row.get("useful_bounces", 0))
        for row in episode_rows
    }

    def collect_contact_values(
        metric_key: str,
        *,
        min_useful_bounces: int | None = None,
        max_useful_bounces: int | None = None,
        useful_only: bool = False,
    ) -> list[float]:
        values: list[float] = []
        for row in contact_rows:
            episode = int(row["episode"])
            useful_bounce_count = episode_useful_bounces.get(episode, 0)
            if min_useful_bounces is not None and useful_bounce_count < min_useful_bounces:
                continue
            if max_useful_bounces is not None and useful_bounce_count > max_useful_bounces:
                continue
            if useful_only and not bool(row.get("is_useful_contact", False)):
                continue
            if row.get(metric_key) is None:
                continue
            values.append(float(row[metric_key]))
        return values

    all_error = collect_contact_values("outgoing_velocity_error_norm")
    useful_error = collect_contact_values("outgoing_velocity_error_norm", useful_only=True)
    two_or_more_episode_error = collect_contact_values("outgoing_velocity_error_norm", min_useful_bounces=2)
    zero_episode_error = collect_contact_values("outgoing_velocity_error_norm", max_useful_bounces=0)
    two_or_more_episode_apex_error = collect_contact_values("predicted_apex_xy_error", min_useful_bounces=2)
    zero_episode_apex_error = collect_contact_values("predicted_apex_xy_error", max_useful_bounces=0)

    mean_two_or_more_episode_error = (
        float(np.mean(two_or_more_episode_error)) if two_or_more_episode_error else None
    )
    mean_zero_episode_error = float(np.mean(zero_episode_error)) if zero_episode_error else None
    return {
        "episodes_with_one_or_more_useful_bounces": sum(value >= 1 for value in episode_useful_bounces.values()),
        "episodes_with_two_or_more_useful_bounces": sum(value >= 2 for value in episode_useful_bounces.values()),
        "all_contact_mean_outgoing_velocity_error_norm": float(np.mean(all_error)) if all_error else None,
        "useful_contact_mean_outgoing_velocity_error_norm": float(np.mean(useful_error)) if useful_error else None,
        "two_or_more_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm": (
            mean_two_or_more_episode_error
        ),
        "zero_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm": mean_zero_episode_error,
        "two_or_more_useful_bounces_outgoing_velocity_error_norm_gap": (
            None
            if mean_two_or_more_episode_error is None or mean_zero_episode_error is None
            else mean_two_or_more_episode_error - mean_zero_episode_error
        ),
        "two_or_more_useful_bounce_episode_contact_mean_predicted_apex_xy_error": (
            float(np.mean(two_or_more_episode_apex_error)) if two_or_more_episode_apex_error else None
        ),
        "zero_useful_bounce_episode_contact_mean_predicted_apex_xy_error": (
            float(np.mean(zero_episode_apex_error)) if zero_episode_apex_error else None
        ),
    }


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
        scene_path=args.scene_path,
        ball_height=args.ball_height,
        target_ball_height=args.target_ball_height,
        max_episode_steps=args.max_episode_steps,
        reset_xy_range=args.reset_xy_range,
        reset_velocity_xy_range=args.reset_velocity_xy_range,
        reset_velocity_z_range=args.reset_velocity_z_range,
        success_velocity_threshold=args.success_velocity_threshold,
    )
    if args.require_reachable_next_intercept_for_success:
        env_kwargs["require_reachable_next_intercept_for_success"] = True
    if args.min_easy_next_ball_score_for_success is not None:
        env_kwargs["min_easy_next_ball_score_for_success"] = args.min_easy_next_ball_score_for_success
    if args.keepup_target_xy_offset is not None:
        env_kwargs["keepup_target_xy_offset"] = tuple(args.keepup_target_xy_offset)
    if args.post_contact_return_z_offset is not None:
        env_kwargs["post_contact_return_z_offset"] = args.post_contact_return_z_offset
    if not args.post_contact_return_predict_during_rise:
        env_kwargs["post_contact_return_predict_during_rise"] = False
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
    if args.contact_frame_tilt_scale_action_limit is not None:
        env_kwargs["contact_frame_tilt_scale_action_limit"] = args.contact_frame_tilt_scale_action_limit
    if args.tracking_strike_plane_offset is not None:
        env_kwargs["tracking_strike_plane_offset"] = args.tracking_strike_plane_offset
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
    if args.contact_frame_trajectory_tilt_gain is not None:
        env_kwargs["contact_frame_trajectory_tilt_gain"] = args.contact_frame_trajectory_tilt_gain
    if args.contact_frame_trajectory_tilt_limit is not None:
        env_kwargs["contact_frame_trajectory_tilt_limit"] = tuple(args.contact_frame_trajectory_tilt_limit)
    if args.contact_frame_trajectory_tilt_deadband is not None:
        env_kwargs["contact_frame_trajectory_tilt_deadband"] = args.contact_frame_trajectory_tilt_deadband
    if args.contact_frame_centering_tilt_limit is not None:
        env_kwargs["contact_frame_centering_tilt_limit"] = tuple(args.contact_frame_centering_tilt_limit)
    if args.contact_frame_centering_tilt_radius is not None:
        env_kwargs["contact_frame_centering_tilt_radius"] = args.contact_frame_centering_tilt_radius
    if args.contact_frame_centering_tilt_deadband is not None:
        env_kwargs["contact_frame_centering_tilt_deadband"] = args.contact_frame_centering_tilt_deadband
    if args.next_intercept_success_radius is not None:
        env_kwargs["next_intercept_success_radius"] = args.next_intercept_success_radius
    if args.easy_next_ball_xy_radius is not None:
        env_kwargs["easy_next_ball_xy_radius"] = args.easy_next_ball_xy_radius
    if args.next_intercept_xy_error_penalty_weight is not None:
        env_kwargs["next_intercept_xy_error_penalty_weight"] = args.next_intercept_xy_error_penalty_weight
    if args.post_contact_lateral_velocity_penalty_weight is not None:
        env_kwargs["post_contact_lateral_velocity_penalty_weight"] = (
            args.post_contact_lateral_velocity_penalty_weight
        )
    if args.contact_xy_error_penalty_weight is not None:
        env_kwargs["contact_xy_error_penalty_weight"] = args.contact_xy_error_penalty_weight
    if args.nonuseful_contact_penalty_weight is not None:
        env_kwargs["nonuseful_contact_penalty_weight"] = args.nonuseful_contact_penalty_weight
    if args.terminate_on_nonuseful_contact:
        env_kwargs["terminate_on_nonuseful_contact"] = True
    env = PingPongKeepUpGymEnv(**env_kwargs)
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
                        "contact_apex_progress_easy_next_ball_gate": info.get(
                            "contact_apex_progress_easy_next_ball_gate"
                        ),
                        "contact_lateral_stability_term": reward_terms.get("contact_lateral_stability_term"),
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
                        "contact_frame_trajectory_tilt_scale": info.get("contact_frame_trajectory_tilt_scale"),
                        "contact_frame_centering_tilt_scale": info.get("contact_frame_centering_tilt_scale"),
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
