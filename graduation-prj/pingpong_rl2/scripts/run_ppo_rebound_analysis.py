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
            "mean_ball_lateral_speed": 0.0,
            "mean_ball_lateral_to_vertical_ratio": 0.0,
            "mean_projected_apex_xy_error": 0.0,
            "useful_contact_mean_projected_apex_xy_error": 0.0,
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
    ball_lateral_speed = float_series("ball_lateral_speed", contact_rows)
    ball_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", contact_rows)
    useful_lateral_speed = float_series("ball_lateral_speed", useful_rows)
    useful_lateral_ratio = float_series("ball_lateral_to_vertical_ratio", useful_rows)
    projected_apex_xy_error = float_series(selected_error_key, contact_rows)
    useful_projected_apex_xy_error = float_series(selected_error_key, useful_rows)
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
    summary = {
        "total_contacts": len(contact_rows),
        "useful_contact_rate": len(useful_rows) / len(contact_rows),
        "selected_apex_target": selected_apex_target,
        "next_intercept_target_source": "controller_anchor",
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
    gravity_z = float(env.base_env.sim.model.opt.gravity[2])
    gravity_magnitude = max(abs(gravity_z), 1.0e-6)
    strike_plane_offset = float(env.base_env._tracking_strike_plane_offset())
    strike_zone_xy_radius = float(env.base_env.strike_zone_xy_radius)
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
                        "contact_ball_position_x": contact_ball_position_x,
                        "contact_ball_position_y": contact_ball_position_y,
                        "contact_ball_position_z": contact_ball_position_z,
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
                        "racket_speed_norm": info.get("contact_racket_speed_norm"),
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
        "episodes_with_one_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 1.0)) if bounce_array.size else 0,
        "one_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 1.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "episodes_with_two_or_more_useful_bounces": int(np.count_nonzero(bounce_array >= 2.0)) if bounce_array.size else 0,
        "two_or_more_useful_bounce_rate": (
            float(np.count_nonzero(bounce_array >= 2.0) / bounce_array.size) if bounce_array.size else 0.0
        ),
        "failure_counts": dict(failure_counts),
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