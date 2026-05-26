from __future__ import annotations

import json
from pathlib import Path

from pingpong_rl2.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_POSITION_TILT_RUN_NAME,
    DEFAULT_PPO_RUN_NAME,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    SMOKE_PPO_POSITION_TILT_RUN_NAME,
    SMOKE_PPO_RUN_NAME,
    default_ppo_model_candidates,
)
from pingpong_rl2.utils.paths import PPO_RUNS_ROOT, resolve_input_path


def default_run_name_for_action_mode(action_mode: str, smoke: bool = False) -> str:
    if smoke:
        return SMOKE_PPO_POSITION_TILT_RUN_NAME if action_mode == "position_tilt" else SMOKE_PPO_RUN_NAME
    return DEFAULT_PPO_POSITION_TILT_RUN_NAME if action_mode == "position_tilt" else DEFAULT_PPO_RUN_NAME


def compose_run_name(base_run_name: str, run_version: str | None = None) -> str:
    version = None if run_version is None else run_version.strip()
    return base_run_name if not version else f"{base_run_name}_{version}"


def resolve_requested_run_name(
    run_name: str | None,
    run_version: str | None = None,
    *,
    action_mode: str = "position",
    smoke: bool = False,
) -> str:
    base_run_name = default_run_name_for_action_mode(action_mode, smoke=smoke) if run_name is None else run_name
    return compose_run_name(base_run_name, run_version)


def model_path_for_run_name(run_name: str, ppo_runs_root: Path = PPO_RUNS_ROOT) -> Path:
    return ppo_runs_root / run_name / f"{run_name}_model.zip"


def training_summary_path_for_run_name(run_name: str, ppo_runs_root: Path = PPO_RUNS_ROOT) -> Path:
    return ppo_runs_root / run_name / f"{run_name}_training_summary.json"


def infer_run_name_from_model_path(model_path: Path) -> str:
    model_stem = model_path.stem
    return model_stem[:-6] if model_stem.endswith("_model") else model_stem


def resolve_saved_model_path(model_path: Path | None = None, run_name: str | None = None) -> Path:
    if model_path is not None:
        return resolve_input_path(model_path)
    if run_name is not None:
        return model_path_for_run_name(run_name)
    candidates = default_ppo_model_candidates(PPO_RUNS_ROOT)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_training_summary(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_env_config_for_model(model_path: Path) -> dict[str, object] | None:
    run_name = infer_run_name_from_model_path(model_path)
    summary_path = model_path.parent / f"{run_name}_training_summary.json"
    summary = load_training_summary(summary_path)
    if summary is None:
        return None
    env_config = summary.get("env_config")
    return dict(env_config) if isinstance(env_config, dict) else None


def resolve_env_kwargs_for_model(
    model_path: Path | None = None,
    *,
    ball_height: float | None = None,
    max_episode_steps: int | None = None,
    reset_xy_range: float | None = None,
    reset_velocity_xy_range: float | None = None,
    reset_velocity_z_range: tuple[float, float] | list[float] | None = None,
    success_velocity_threshold: float | None = None,
) -> dict[str, object]:
    env_kwargs: dict[str, object] = {
        "action_mode": "position",
        "ball_height": DEFAULT_BALL_HEIGHT,
        "target_ball_height": DEFAULT_BALL_HEIGHT,
        "max_episode_steps": DEFAULT_MAX_EPISODE_STEPS,
        "reset_xy_range": DEFAULT_RESET_XY_RANGE,
        "reset_velocity_xy_range": DEFAULT_RESET_VELOCITY_XY_RANGE,
        "reset_velocity_z_range": tuple(DEFAULT_RESET_VELOCITY_Z_RANGE),
        "success_velocity_threshold": DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    }
    if model_path is not None:
        summary_env_config = load_env_config_for_model(model_path)
        if summary_env_config is not None:
            env_kwargs.update(summary_env_config)

    if ball_height is not None:
        env_kwargs["ball_height"] = float(ball_height)
        env_kwargs["target_ball_height"] = float(ball_height)
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = int(max_episode_steps)
    if reset_xy_range is not None:
        env_kwargs["reset_xy_range"] = float(reset_xy_range)
    if reset_velocity_xy_range is not None:
        env_kwargs["reset_velocity_xy_range"] = float(reset_velocity_xy_range)
    if reset_velocity_z_range is not None:
        env_kwargs["reset_velocity_z_range"] = tuple(float(value) for value in reset_velocity_z_range)
    if success_velocity_threshold is not None:
        env_kwargs["success_velocity_threshold"] = float(success_velocity_threshold)
    return env_kwargs