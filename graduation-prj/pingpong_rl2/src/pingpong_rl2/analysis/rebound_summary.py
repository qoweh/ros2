from __future__ import annotations

import numpy as np

from pingpong_rl2.analysis.rebound_metrics import _APEX_TARGET_CHOICES

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
            "mean_contact_apex_potential_term": 0.0,
            "mean_contact_frame_lateral_brake_speed": 0.0,
            "max_contact_frame_lateral_brake_speed": 0.0,
            "mean_contact_racket_outward_speed": 0.0,
            "max_contact_racket_outward_speed": 0.0,
            "mean_contact_racket_outward_velocity_penalty": 0.0,
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
    apex_potential_term = float_series("contact_apex_potential_term", contact_rows)
    apex_progress_gate = float_series("contact_apex_progress_easy_next_ball_gate", contact_rows)
    lateral_brake_x = float_series("contact_frame_lateral_brake_velocity_x", contact_rows)
    lateral_brake_y = float_series("contact_frame_lateral_brake_velocity_y", contact_rows)
    lateral_brake_speed = (
        np.sqrt(lateral_brake_x * lateral_brake_x + lateral_brake_y * lateral_brake_y)
        if lateral_brake_x.size and lateral_brake_y.size and lateral_brake_x.size == lateral_brake_y.size
        else np.asarray([], dtype=float)
    )
    racket_outward_speed = float_series("contact_racket_outward_speed", contact_rows)
    racket_outward_velocity_penalty = float_series("contact_racket_outward_velocity_penalty", contact_rows)
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
        "mean_contact_frame_lateral_brake_speed": (
            float(lateral_brake_speed.mean()) if lateral_brake_speed.size else 0.0
        ),
        "max_contact_frame_lateral_brake_speed": (
            float(lateral_brake_speed.max()) if lateral_brake_speed.size else 0.0
        ),
        "mean_contact_racket_outward_speed": (
            float(racket_outward_speed.mean()) if racket_outward_speed.size else 0.0
        ),
        "max_contact_racket_outward_speed": (
            float(racket_outward_speed.max()) if racket_outward_speed.size else 0.0
        ),
        "mean_contact_racket_outward_velocity_penalty": (
            float(racket_outward_velocity_penalty.mean()) if racket_outward_velocity_penalty.size else 0.0
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
        "mean_contact_apex_potential_term": (
            float(apex_potential_term.mean()) if apex_potential_term.size else 0.0
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


