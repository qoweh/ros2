from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class ResetDistributionCurriculumCallback(BaseCallback):
    def __init__(
        self,
        *,
        start_xy_range: float,
        end_xy_range: float,
        start_velocity_xy_range: float,
        end_velocity_xy_range: float,
        start_velocity_z_range: Sequence[float],
        end_velocity_z_range: Sequence[float],
        start_ball_angular_velocity_range: float,
        end_ball_angular_velocity_range: float,
        total_timesteps: int,
        curriculum_fraction: float,
        update_interval: int,
    ) -> None:
        super().__init__(verbose=0)
        if start_xy_range < 0.0 or end_xy_range < 0.0:
            raise ValueError("reset xy curriculum ranges must be non-negative.")
        if start_velocity_xy_range < 0.0 or end_velocity_xy_range < 0.0:
            raise ValueError("reset velocity xy curriculum ranges must be non-negative.")
        if start_ball_angular_velocity_range < 0.0 or end_ball_angular_velocity_range < 0.0:
            raise ValueError("reset ball angular velocity curriculum ranges must be non-negative.")
        start_velocity_z_range = (float(start_velocity_z_range[0]), float(start_velocity_z_range[1]))
        end_velocity_z_range = (float(end_velocity_z_range[0]), float(end_velocity_z_range[1]))
        if start_velocity_z_range[0] > start_velocity_z_range[1]:
            raise ValueError(
                "reset_velocity_z_curriculum_start must be ordered as (low, high), got "
                f"{start_velocity_z_range}."
            )
        if end_velocity_z_range[0] > end_velocity_z_range[1]:
            raise ValueError(
                "reset_velocity_z_curriculum_end must be ordered as (low, high), got "
                f"{end_velocity_z_range}."
            )
        if total_timesteps < 0:
            raise ValueError(f"total_timesteps must be non-negative, got {total_timesteps}.")
        if curriculum_fraction <= 0.0:
            raise ValueError(f"reset_xy_curriculum_fraction must be positive, got {curriculum_fraction}.")
        if update_interval < 1:
            raise ValueError(f"reset_xy_curriculum_update_interval must be positive, got {update_interval}.")
        self.start_xy_range = float(start_xy_range)
        self.end_xy_range = float(end_xy_range)
        self.start_velocity_xy_range = float(start_velocity_xy_range)
        self.end_velocity_xy_range = float(end_velocity_xy_range)
        self.start_velocity_z_range = start_velocity_z_range
        self.end_velocity_z_range = end_velocity_z_range
        self.start_ball_angular_velocity_range = float(start_ball_angular_velocity_range)
        self.end_ball_angular_velocity_range = float(end_ball_angular_velocity_range)
        self.total_timesteps = int(total_timesteps)
        self.curriculum_fraction = float(curriculum_fraction)
        self.update_interval = int(update_interval)
        self._base_num_timesteps: int | None = None
        self._last_applied_distribution: dict[str, object] | None = None

    def _progress(self) -> float:
        if self.total_timesteps <= 0:
            return 1.0
        run_timesteps = max(int(self.num_timesteps) - int(self._base_num_timesteps or 0), 0)
        curriculum_timesteps = max(int(round(self.total_timesteps * self.curriculum_fraction)), 1)
        return float(np.clip(run_timesteps / curriculum_timesteps, 0.0, 1.0))

    @staticmethod
    def _lerp(start: float, end: float, progress: float) -> float:
        return float(start + (end - start) * progress)

    def target_distribution(self) -> dict[str, object]:
        progress = self._progress()
        velocity_z_range = (
            self._lerp(self.start_velocity_z_range[0], self.end_velocity_z_range[0], progress),
            self._lerp(self.start_velocity_z_range[1], self.end_velocity_z_range[1], progress),
        )
        return {
            "reset_xy_range": self._lerp(self.start_xy_range, self.end_xy_range, progress),
            "reset_velocity_xy_range": self._lerp(
                self.start_velocity_xy_range,
                self.end_velocity_xy_range,
                progress,
            ),
            "reset_velocity_z_range": velocity_z_range,
            "reset_ball_angular_velocity_range": self._lerp(
                self.start_ball_angular_velocity_range,
                self.end_ball_angular_velocity_range,
                progress,
            ),
        }

    @staticmethod
    def _same_distribution(left: dict[str, object], right: dict[str, object]) -> bool:
        for key in left:
            left_value = left[key]
            right_value = right.get(key)
            if isinstance(left_value, tuple):
                if right_value is None:
                    return False
                if any(abs(float(a) - float(b)) >= 1.0e-6 for a, b in zip(left_value, right_value)):
                    return False
            elif right_value is None or abs(float(left_value) - float(right_value)) >= 1.0e-6:
                return False
        return True

    def _apply_distribution(self, distribution: dict[str, object]) -> None:
        if self._last_applied_distribution is not None and self._same_distribution(
            distribution,
            self._last_applied_distribution,
        ):
            return
        self.training_env.env_method("set_reset_distribution", **distribution)
        self._last_applied_distribution = dict(distribution)

    def _on_training_start(self) -> None:
        if self._base_num_timesteps is None:
            self._base_num_timesteps = int(self.model.num_timesteps)
        self._apply_distribution(self.target_distribution())

    def _on_step(self) -> bool:
        if self.n_calls % self.update_interval == 0:
            self._apply_distribution(self.target_distribution())
        return True


def build_reset_xy_curriculum_callback(args: argparse.Namespace) -> ResetDistributionCurriculumCallback | None:
    if not args.reset_xy_curriculum_enabled:
        return None
    start_xy_range = args.reset_xy_range if args.reset_xy_curriculum_start is None else args.reset_xy_curriculum_start
    end_xy_range = args.reset_xy_range if args.reset_xy_curriculum_end is None else args.reset_xy_curriculum_end
    start_velocity_xy_range = (
        args.reset_velocity_xy_range
        if args.reset_velocity_xy_curriculum_start is None
        else args.reset_velocity_xy_curriculum_start
    )
    end_velocity_xy_range = (
        args.reset_velocity_xy_range
        if args.reset_velocity_xy_curriculum_end is None
        else args.reset_velocity_xy_curriculum_end
    )
    start_velocity_z_range = (
        args.reset_velocity_z_range
        if args.reset_velocity_z_curriculum_start is None
        else args.reset_velocity_z_curriculum_start
    )
    end_velocity_z_range = (
        args.reset_velocity_z_range
        if args.reset_velocity_z_curriculum_end is None
        else args.reset_velocity_z_curriculum_end
    )
    start_ball_angular_velocity_range = (
        args.reset_ball_angular_velocity_range
        if args.reset_ball_angular_velocity_curriculum_start is None
        else args.reset_ball_angular_velocity_curriculum_start
    )
    end_ball_angular_velocity_range = (
        args.reset_ball_angular_velocity_range
        if args.reset_ball_angular_velocity_curriculum_end is None
        else args.reset_ball_angular_velocity_curriculum_end
    )
    return ResetDistributionCurriculumCallback(
        start_xy_range=float(start_xy_range),
        end_xy_range=float(end_xy_range),
        start_velocity_xy_range=float(start_velocity_xy_range),
        end_velocity_xy_range=float(end_velocity_xy_range),
        start_velocity_z_range=start_velocity_z_range,
        end_velocity_z_range=end_velocity_z_range,
        start_ball_angular_velocity_range=float(start_ball_angular_velocity_range),
        end_ball_angular_velocity_range=float(end_ball_angular_velocity_range),
        total_timesteps=int(args.total_timesteps),
        curriculum_fraction=float(args.reset_xy_curriculum_fraction),
        update_interval=int(args.reset_xy_curriculum_update_interval),
    )


