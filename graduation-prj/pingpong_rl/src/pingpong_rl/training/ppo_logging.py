from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


EPISODE_FIELDS: tuple[str, ...] = (
    "episode_index",
    "terminated",
    "truncated",
    "success_reason",
    "episode_success_reason",
    "failure_reason",
    "time_limit_reached",
    "episode_steps",
    "contact_count",
    "successful_bounce_count",
    "first_contact_step",
    "reward_total_sum",
    "reward_height_sum",
    "reward_distance_sum",
    "reward_contact_sum",
    "reward_success_sum",
    "reward_failure_sum",
)

STEP_FIELDS: tuple[str, ...] = (
    "episode_index",
    "step",
    "reward_total",
    "reward_height",
    "reward_distance",
    "reward_contact",
    "reward_success",
    "reward_failure",
    "terminated",
    "truncated",
    "time_limit_reached",
    "success_reason",
    "episode_success_reason",
    "failure_reason",
    "successful_bounce_count",
    "racket_contact",
    "contact_observed_during_step",
    "contact_event_during_step",
    "contact_substep",
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    "ball_speed_norm",
    "contact_ball_velocity_x",
    "contact_ball_velocity_y",
    "contact_ball_velocity_z",
    "contact_ball_speed_norm",
)

CONTACT_FIELDS: tuple[str, ...] = (
    "episode_index",
    "contact_step",
    "contact_substep",
    "ball_velocity_x",
    "ball_velocity_y",
    "ball_velocity_z",
    "ball_speed_norm",
    "success_reason",
    "failure_reason",
    "terminated",
    "truncated",
    "time_limit_reached",
)


def _normalize_reason(value: object) -> str:
    return "" if value is None else str(value)


def _quantile_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "p50": None, "p75": None, "p90": None, "max": None}

    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _episode_metric_stats(rows: Sequence[dict[str, object]], field_name: str) -> dict[str, float | None]:
    return _quantile_stats([float(row[field_name]) for row in rows])


def build_training_summary(
    episode_rows: Sequence[dict[str, object]],
    contact_rows: Sequence[dict[str, object]],
    config: dict[str, object],
) -> dict[str, object]:
    success_counter = Counter(str(row["episode_success_reason"]) for row in episode_rows if row["episode_success_reason"])
    failure_counter = Counter(str(row["failure_reason"]) for row in episode_rows if row["failure_reason"])
    contact_velocity_x = [float(row["ball_velocity_x"]) for row in contact_rows if row["ball_velocity_x"] is not None]
    contact_velocity_y = [float(row["ball_velocity_y"]) for row in contact_rows if row["ball_velocity_y"] is not None]
    contact_velocity_z = [float(row["ball_velocity_z"]) for row in contact_rows if row["ball_velocity_z"] is not None]
    contact_speed_norm = [float(row["ball_speed_norm"]) for row in contact_rows if row["ball_speed_norm"] is not None]
    zero_contact_reward_episodes = sum(float(row["reward_contact_sum"]) == 0.0 for row in episode_rows)
    zero_success_reward_episodes = sum(float(row["reward_success_sum"]) == 0.0 for row in episode_rows)
    height_dominant_episodes = sum(
        float(row["reward_height_sum"]) > float(row["reward_contact_sum"]) + float(row["reward_success_sum"])
        for row in episode_rows
    )
    survival_without_success_episodes = sum(
        (not bool(row["truncated"])) and row["episode_success_reason"] == "" and float(row["reward_height_sum"]) > 0.0
        for row in episode_rows
    )
    return {
        "config": config,
        "episode_counts": {
            "episodes": len(episode_rows),
            "terminated": sum(bool(row["terminated"]) for row in episode_rows),
            "truncated": sum(bool(row["truncated"]) for row in episode_rows),
            "successes": int(sum(success_counter.values())),
            "failures": int(sum(failure_counter.values())),
            "contacts": len(contact_rows),
            "episodes_with_bounces": sum(int(row["successful_bounce_count"]) > 0 for row in episode_rows),
        },
        "success_reason_counts": dict(success_counter),
        "failure_reason_counts": dict(failure_counter),
        "bounce_count_stats": _episode_metric_stats(episode_rows, "successful_bounce_count"),
        "reward_sum_stats": {
            "reward_total_sum": _episode_metric_stats(episode_rows, "reward_total_sum"),
            "reward_height_sum": _episode_metric_stats(episode_rows, "reward_height_sum"),
            "reward_distance_sum": _episode_metric_stats(episode_rows, "reward_distance_sum"),
            "reward_contact_sum": _episode_metric_stats(episode_rows, "reward_contact_sum"),
            "reward_success_sum": _episode_metric_stats(episode_rows, "reward_success_sum"),
            "reward_failure_sum": _episode_metric_stats(episode_rows, "reward_failure_sum"),
        },
        "reward_dominance": {
            "zero_contact_reward_episodes": zero_contact_reward_episodes,
            "zero_success_reward_episodes": zero_success_reward_episodes,
            "height_dominant_episodes": height_dominant_episodes,
            "survival_without_success_episodes": survival_without_success_episodes,
        },
        "contact_velocity_stats": {
            "ball_velocity_x": _quantile_stats(contact_velocity_x),
            "ball_velocity_y": _quantile_stats(contact_velocity_y),
            "ball_velocity_z": _quantile_stats(contact_velocity_z),
            "ball_speed_norm": _quantile_stats(contact_speed_norm),
        },
    }


class PPOLoggingCallback(BaseCallback):
    def __init__(
        self,
        output_dir: Path,
        run_name: str,
        summary_config: dict[str, object],
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.summary_config = dict(summary_config)
        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.episodes_path = self.output_dir / f"{self.run_name}_episodes.csv"
        self.steps_path = self.output_dir / f"{self.run_name}_steps.csv"
        self.contacts_path = self.output_dir / f"{self.run_name}_contacts.csv"
        self.summary_path = self.output_dir / f"{self.run_name}_training_summary.json"
        self._episode_rows: list[dict[str, object]] = []
        self._contact_rows: list[dict[str, object]] = []

    def _empty_episode_state(self) -> dict[str, object]:
        return {
            "contact_count": 0,
            "successful_bounce_count": 0,
            "first_contact_step": None,
            "reward_total_sum": 0.0,
            "reward_height_sum": 0.0,
            "reward_distance_sum": 0.0,
            "reward_contact_sum": 0.0,
            "reward_success_sum": 0.0,
            "reward_failure_sum": 0.0,
        }

    def _on_training_start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self._episode_file = self.episodes_path.open("w", newline="", encoding="utf-8")
        self._step_file = self.steps_path.open("w", newline="", encoding="utf-8")
        self._contact_file = self.contacts_path.open("w", newline="", encoding="utf-8")
        self._episode_writer = csv.DictWriter(self._episode_file, fieldnames=list(EPISODE_FIELDS))
        self._step_writer = csv.DictWriter(self._step_file, fieldnames=list(STEP_FIELDS))
        self._contact_writer = csv.DictWriter(self._contact_file, fieldnames=list(CONTACT_FIELDS))
        self._episode_writer.writeheader()
        self._step_writer.writeheader()
        self._contact_writer.writeheader()
        self._summary_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        self._episode_index = 0
        self._episode_states = [self._empty_episode_state() for _ in range(self.training_env.num_envs)]

    def _step_row(self, episode_index: int, info: dict[str, object]) -> dict[str, object]:
        return {
            "episode_index": episode_index,
            "step": int(info["episode_steps"]),
            "reward_total": float(info["reward_total"]),
            "reward_height": float(info["reward_height"]),
            "reward_distance": float(info["reward_distance"]),
            "reward_contact": float(info["reward_contact"]),
            "reward_success": float(info["reward_success"]),
            "reward_failure": float(info["reward_failure"]),
            "terminated": bool(info["terminated"]),
            "truncated": bool(info["truncated"]),
            "time_limit_reached": bool(info["time_limit_reached"]),
            "success_reason": _normalize_reason(info["success_reason"]),
            "episode_success_reason": _normalize_reason(info.get("episode_success_reason")),
            "failure_reason": _normalize_reason(info["failure_reason"]),
            "successful_bounce_count": int(info.get("successful_bounce_count", 0)),
            "racket_contact": bool(info["racket_contact"]),
            "contact_observed_during_step": bool(info["contact_observed_during_step"]),
            "contact_event_during_step": bool(info.get("contact_event_during_step", False)),
            "contact_substep": info["contact_substep"],
            "ball_velocity_x": float(info["ball_velocity_x"]),
            "ball_velocity_y": float(info["ball_velocity_y"]),
            "ball_velocity_z": float(info["ball_velocity_z"]),
            "ball_speed_norm": float(info["ball_speed_norm"]),
            "contact_ball_velocity_x": info["contact_ball_velocity_x"],
            "contact_ball_velocity_y": info["contact_ball_velocity_y"],
            "contact_ball_velocity_z": info["contact_ball_velocity_z"],
            "contact_ball_speed_norm": info["contact_ball_speed_norm"],
        }

    def _contact_row(self, episode_index: int, info: dict[str, object]) -> dict[str, object]:
        return {
            "episode_index": episode_index,
            "contact_step": int(info["episode_steps"]),
            "contact_substep": info["contact_substep"],
            "ball_velocity_x": info["contact_ball_velocity_x"],
            "ball_velocity_y": info["contact_ball_velocity_y"],
            "ball_velocity_z": info["contact_ball_velocity_z"],
            "ball_speed_norm": info["contact_ball_speed_norm"],
            "success_reason": _normalize_reason(info["success_reason"]),
            "failure_reason": _normalize_reason(info["failure_reason"]),
            "terminated": bool(info["terminated"]),
            "truncated": bool(info["truncated"]),
            "time_limit_reached": bool(info["time_limit_reached"]),
        }

    def _episode_row(self, episode_index: int, info: dict[str, object], episode_state: dict[str, object]) -> dict[str, object]:
        return {
            "episode_index": episode_index,
            "terminated": bool(info["terminated"]),
            "truncated": bool(info["truncated"]),
            "success_reason": _normalize_reason(info["success_reason"]),
            "episode_success_reason": _normalize_reason(info.get("episode_success_reason")),
            "failure_reason": _normalize_reason(info["failure_reason"]),
            "time_limit_reached": bool(info["time_limit_reached"]),
            "episode_steps": int(info["episode_steps"]),
            "contact_count": int(episode_state["contact_count"]),
            "successful_bounce_count": int(episode_state["successful_bounce_count"]),
            "first_contact_step": episode_state["first_contact_step"] or "",
            "reward_total_sum": float(episode_state["reward_total_sum"]),
            "reward_height_sum": float(episode_state["reward_height_sum"]),
            "reward_distance_sum": float(episode_state["reward_distance_sum"]),
            "reward_contact_sum": float(episode_state["reward_contact_sum"]),
            "reward_success_sum": float(episode_state["reward_success_sum"]),
            "reward_failure_sum": float(episode_state["reward_failure_sum"]),
        }

    def _log_tensorboard(self, episode_row: dict[str, object]) -> None:
        episode_index = int(episode_row["episode_index"])
        for field_name in (
            "reward_total_sum",
            "reward_height_sum",
            "reward_distance_sum",
            "reward_contact_sum",
            "reward_success_sum",
            "reward_failure_sum",
            "episode_steps",
            "contact_count",
            "successful_bounce_count",
        ):
            self._summary_writer.add_scalar(field_name, float(episode_row[field_name]), episode_index)
        self._summary_writer.add_scalar("terminated", float(bool(episode_row["terminated"])), episode_index)
        self._summary_writer.add_scalar("truncated", float(bool(episode_row["truncated"])), episode_index)
        self._summary_writer.add_scalar(
            "time_limit_reached",
            float(bool(episode_row["time_limit_reached"])),
            episode_index,
        )
        if episode_row["episode_success_reason"]:
            self._summary_writer.add_scalar(
                f"success_reason/{episode_row['episode_success_reason']}",
                1.0,
                episode_index,
            )
        if episode_row["failure_reason"]:
            self._summary_writer.add_scalar(
                f"failure_reason/{episode_row['failure_reason']}",
                1.0,
                episode_index,
            )

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for env_index, (info, done) in enumerate(zip(infos, dones)):
            episode_state = self._episode_states[env_index]
            step_episode_index = self._episode_index + 1
            self._step_writer.writerow(self._step_row(step_episode_index, info))
            episode_state["reward_total_sum"] += float(info["reward_total"])
            episode_state["reward_height_sum"] += float(info["reward_height"])
            episode_state["reward_distance_sum"] += float(info["reward_distance"])
            episode_state["reward_contact_sum"] += float(info["reward_contact"])
            episode_state["reward_success_sum"] += float(info["reward_success"])
            episode_state["reward_failure_sum"] += float(info["reward_failure"])
            episode_state["successful_bounce_count"] = max(
                int(episode_state["successful_bounce_count"]),
                int(info.get("successful_bounce_count", 0)),
            )
            if bool(info.get("contact_event_during_step", False)):
                episode_state["contact_count"] += 1
                if episode_state["first_contact_step"] is None:
                    episode_state["first_contact_step"] = int(info["episode_steps"])
                contact_row = self._contact_row(step_episode_index, info)
                self._contact_writer.writerow(contact_row)
                self._contact_rows.append(contact_row)
            if not done:
                continue

            self._episode_index += 1
            episode_row = self._episode_row(self._episode_index, info, episode_state)
            self._episode_writer.writerow(episode_row)
            self._episode_rows.append(episode_row)
            self._log_tensorboard(episode_row)
            self._episode_states[env_index] = self._empty_episode_state()
            self._episode_file.flush()
            self._step_file.flush()
            self._contact_file.flush()
        return True

    def _on_training_end(self) -> None:
        summary = build_training_summary(self._episode_rows, self._contact_rows, self.summary_config)
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._summary_writer.flush()
        self._summary_writer.close()
        self._episode_file.close()
        self._step_file.close()
        self._contact_file.close()