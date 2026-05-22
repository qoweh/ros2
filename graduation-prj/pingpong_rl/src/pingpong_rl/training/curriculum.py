from __future__ import annotations

from dataclasses import dataclass

from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    progress_start: float
    env_updates: dict[str, object]


CURRICULA: dict[str, tuple[CurriculumStage, ...]] = {
    "keepup_v1": (
        CurriculumStage(
            name="bootstrap",
            progress_start=0.0,
            env_updates={
                "reset_ball_height_range": 0.0,
                "reset_xy_range": 0.0,
                "reset_velocity_xy_range": 0.0,
                "reset_velocity_z_range": (0.0, 0.0),
                "success_velocity_threshold": 0.35,
                "tracking_assist_weight": 0.45,
                "tilt_tracking_assist_weight": 0.0,
                "tracking_alignment_reward_weight": 2.0,
                "contact_centering_reward_weight": 0.5,
                "racket_tilt_penalty_weight": 0.0,
                "joint_velocity_penalty_weight": 0.0,
                "action_smoothness_penalty_weight": 0.0,
                "action_filter_alpha": 0.0,
            },
        ),
        CurriculumStage(
            name="stabilize",
            progress_start=0.35,
            env_updates={
                "reset_ball_height_range": 0.025,
                "reset_xy_range": 0.01,
                "reset_velocity_xy_range": 0.005,
                "reset_velocity_z_range": (-0.01, 0.01),
                "success_velocity_threshold": 0.45,
                "tracking_assist_weight": 0.30,
                "tilt_tracking_assist_weight": 0.0,
                "tracking_alignment_reward_weight": 1.5,
                "contact_centering_reward_weight": 1.0,
                "racket_tilt_penalty_weight": 0.25,
                "joint_velocity_penalty_weight": 0.0015,
                "action_smoothness_penalty_weight": 0.03,
                "action_filter_alpha": 0.06,
            },
        ),
        CurriculumStage(
            name="refine",
            progress_start=0.75,
            env_updates={
                "reset_ball_height_range": 0.05,
                "reset_xy_range": 0.02,
                "reset_velocity_xy_range": 0.01,
                "reset_velocity_z_range": (-0.02, 0.02),
                "success_velocity_threshold": 0.5,
                "tracking_assist_weight": 0.15,
                "tilt_tracking_assist_weight": 0.0,
                "tracking_alignment_reward_weight": 0.75,
                "contact_centering_reward_weight": 1.25,
                "racket_tilt_penalty_weight": 0.6,
                "joint_velocity_penalty_weight": 0.004,
                "action_smoothness_penalty_weight": 0.05,
                "action_filter_alpha": 0.12,
            },
        ),
    ),
}


def curriculum_names() -> tuple[str, ...]:
    return ("none", *tuple(sorted(CURRICULA)))


class CurriculumCallback(BaseCallback):
    def __init__(self, curriculum_name: str, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if curriculum_name not in CURRICULA:
            raise ValueError(f"Unknown curriculum_name={curriculum_name!r}.")
        if total_timesteps < 1:
            raise ValueError(f"total_timesteps must be positive, got {total_timesteps}.")

        self.curriculum_name = curriculum_name
        self.total_timesteps = int(total_timesteps)
        self._stages = CURRICULA[curriculum_name]
        self._current_stage_index = -1

    def _on_training_start(self) -> None:
        self._maybe_apply_stage(progress=0.0)

    def _on_step(self) -> bool:
        progress = min(max(self.num_timesteps / max(self.total_timesteps, 1), 0.0), 1.0)
        self._maybe_apply_stage(progress=progress)
        return True

    def _maybe_apply_stage(self, progress: float) -> None:
        target_stage_index = 0
        for index, stage in enumerate(self._stages):
            if progress >= stage.progress_start:
                target_stage_index = index

        if target_stage_index == self._current_stage_index:
            return

        stage = self._stages[target_stage_index]
        self.training_env.env_method("apply_curriculum_stage", stage.name, stage.env_updates)
        self._current_stage_index = target_stage_index
        if self.verbose > 0:
            print(
                f"curriculum_name={self.curriculum_name} curriculum_stage={stage.name} "
                f"progress={progress:.3f} updates={stage.env_updates}"
            )