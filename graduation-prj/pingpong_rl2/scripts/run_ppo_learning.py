from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from math import ceil
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import torch as th

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.controllers import HeuristicKeepUpPolicy
from pingpong_rl2.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_BATCH_SIZE,
    DEFAULT_PPO_GAMMA,
    DEFAULT_PPO_LEARNING_RATE,
    DEFAULT_PPO_N_STEPS,
    DEFAULT_PPO_RUN_NAME,
    DEFAULT_PPO_TOTAL_TIMESTEPS,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    SMOKE_PPO_BATCH_SIZE,
    SMOKE_PPO_N_STEPS,
    SMOKE_PPO_RUN_NAME,
    SMOKE_PPO_TOTAL_TIMESTEPS,
)
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.training import make_sb3_async_vector_env
from pingpong_rl2.utils import (
    PPO_RUNS_ROOT,
    resolve_input_path,
    resolve_output_path,
    resolve_requested_run_name,
)

_TILT_PROFILES: dict[str, dict[str, object]] = {
    "early": {
        "tilt_action_limit": 0.015,
        "target_tilt_limit": (0.06, 0.06),
        "tilt_angle_penalty_weight": 0.06,
        "tilt_action_delta_penalty_weight": 0.12,
    },
    "mid": {
        "tilt_action_limit": 0.025,
        "target_tilt_limit": (0.09, 0.09),
        "tilt_angle_penalty_weight": 0.05,
        "tilt_action_delta_penalty_weight": 0.10,
    },
    "late": {
        "tilt_action_limit": 0.035,
        "target_tilt_limit": (0.12, 0.12),
        "tilt_angle_penalty_weight": 0.04,
        "tilt_action_delta_penalty_weight": 0.08,
    },
    "final": {
        "tilt_action_limit": 0.04,
        "target_tilt_limit": (0.12, 0.12),
        "tilt_angle_penalty_weight": 0.03,
        "tilt_action_delta_penalty_weight": 0.06,
    },
}

_ENV_PRESETS: dict[str, dict[str, object]] = {
    "baseline_position": {
        "action_mode": "position",
    },
    "strike_position": {
        "action_mode": "position_strike",
    },
    "strike_velocity_obs": {
        "action_mode": "position_strike",
        "include_velocity_domain_observation": True,
    },
    "tilt_experiment": {
        "action_mode": "position_tilt",
        "tilt_profile": "auto",
    },
    "final_candidate": {
        "action_mode": "position_strike",
        "strike_tilt_ramp_pitch": -0.03,
        "strike_tilt_ramp_xy_tolerance": 0.04,
        "post_contact_return_assist_weight": 0.5,
        "post_contact_return_max_intercept_time": 0.6,
    },
    "phase_contract_candidate": {
        "action_mode": "position_strike",
        "strike_tilt_ramp_pitch": -0.03,
        "strike_tilt_ramp_xy_tolerance": 0.04,
        "post_contact_return_assist_weight": 0.5,
        "post_contact_return_max_intercept_time": 0.6,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
        "next_intercept_reachable_bonus_weight": 0.2,
    },
    "followup_strike_candidate": {
        "action_mode": "position_strike",
        "strike_tilt_ramp_pitch": -0.03,
        "strike_tilt_ramp_xy_tolerance": 0.04,
        "followup_strike_target_tilt": (-0.03, 0.0),
        "post_contact_return_assist_weight": 0.5,
        "post_contact_return_max_intercept_time": 0.6,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
        "next_intercept_reachable_bonus_weight": 0.2,
    },
    "contact_primitive_candidate": {
        "action_mode": "position_strike_tilt",
        "tilt_profile": "early",
        "strike_tilt_ramp_pitch": -0.06,
        "strike_tilt_ramp_xy_tolerance": 0.04,
        "followup_strike_target_tilt": (-0.06, 0.0),
        "followup_strike_lift_boost": 0.02,
        "post_contact_return_assist_weight": 0.5,
        "post_contact_return_max_intercept_time": 0.6,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
    },
    "contact_lift_candidate": {
        "action_mode": "position_strike_tilt_lift",
        "tilt_profile": "early",
        "strike_tilt_ramp_pitch": -0.06,
        "strike_tilt_ramp_xy_tolerance": 0.04,
        "followup_strike_target_tilt": (-0.06, 0.0),
        "followup_strike_lift_boost": 0.02,
        "followup_lift_action_limit": 0.02,
        "post_contact_return_assist_weight": 0.5,
        "post_contact_return_max_intercept_time": 0.6,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
    },
    "contact_frame_candidate": {
        "action_mode": "position_contact_frame",
        "tilt_profile": "early",
        "strike_tilt_ramp_pitch": -0.06,
        "strike_tilt_ramp_xy_tolerance": 0.04,
        "followup_strike_target_tilt": (-0.06, 0.0),
        "contact_frame_base_strike_z_boost": 0.024,
        "contact_frame_base_strike_z_offset": 0.01,
        "contact_frame_base_tilt_residual": (-0.02, 0.0),
        "contact_frame_action_penalty_weight": 0.05,
        "log_std_init": -3.0,
        "zero_init_action_mean": True,
        "post_contact_return_assist_weight": 0.8,
        "post_contact_return_max_intercept_time": 0.6,
        "include_task_phase_observation": True,
        "include_contact_context_observation": True,
        "include_next_intercept_observation": True,
    },
}

_PRESET_MANAGED_ARG_DEFAULTS: dict[str, object] = {
    "action_mode": "position",
    "tilt_profile": "auto",
    "followup_lift_action_limit": None,
    "strike_tilt_ramp_pitch": None,
    "strike_tilt_ramp_xy_tolerance": None,
    "post_contact_return_assist_weight": None,
    "post_contact_return_max_intercept_time": None,
    "include_velocity_domain_observation": False,
    "include_task_phase_observation": False,
    "include_contact_context_observation": False,
    "include_next_intercept_observation": False,
    "include_desired_outgoing_velocity_observation": False,
    "trajectory_match_reward_weight": None,
    "next_intercept_reachable_bonus_weight": None,
    "followup_strike_target_tilt": None,
    "followup_strike_contact_offset_ratio": None,
    "followup_strike_contact_offset_max": None,
    "followup_strike_lift_boost": None,
    "contact_frame_base_strike_z_boost": None,
    "contact_frame_base_strike_z_offset": None,
    "contact_frame_base_strike_time_horizon": None,
    "contact_frame_base_tilt_residual": None,
    "contact_frame_action_penalty_weight": None,
    "log_std_init": None,
    "zero_init_action_mean": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal pingpong_rl2 PPO baseline.")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=tuple(_ENV_PRESETS.keys()),
        help="Optional experiment preset that applies a fixed env configuration before any manual overrides.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--run-version",
        type=str,
        default=None,
        help="Optional suffix appended as <run-name>_<run-version> so A/B runs stay in separate directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint zip to continue from. Default behavior resumes <run-name>_model.zip when it exists.",
    )
    parser.add_argument(
        "--reset-model",
        action="store_true",
        help="Start a fresh model even when the target run directory already has a saved checkpoint.",
    )
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_PPO_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_PPO_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PPO_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PPO_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_PPO_GAMMA)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--bootstrap-heuristic-episodes",
        type=int,
        default=0,
        help="Optional number of heuristic rollout episodes to collect before PPO learning for actor warm-start.",
    )
    parser.add_argument(
        "--bootstrap-min-useful-bounces",
        type=int,
        default=1,
        help="Minimum useful bounce count required for a heuristic episode to be kept in the bootstrap dataset.",
    )
    parser.add_argument(
        "--bootstrap-max-samples",
        type=int,
        default=0,
        help="Optional hard cap on accepted bootstrap samples. Set 0 for no cap.",
    )
    parser.add_argument(
        "--bootstrap-epochs",
        type=int,
        default=0,
        help="Number of supervised actor pretraining epochs on the accepted heuristic dataset.",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=256,
        help="Batch size used for heuristic actor pretraining.",
    )
    parser.add_argument(
        "--bootstrap-learning-rate",
        type=float,
        default=1.0e-3,
        help="Learning rate used for heuristic actor pretraining.",
    )
    parser.add_argument(
        "--bootstrap-sample-mode",
        type=str,
        default="episode",
        choices=("episode", "post_success", "post_success_reachable"),
        help=(
            "Which heuristic samples to keep: full qualifying episodes, only post-success steps, "
            "or only post-success steps whose next ball is still reachable."
        ),
    )
    parser.add_argument(
        "--bootstrap-followup-epochs",
        type=int,
        default=0,
        help="Optional extra actor pretraining epochs on a follow-up-focused heuristic dataset after the base bootstrap pass.",
    )
    parser.add_argument(
        "--bootstrap-followup-sample-mode",
        type=str,
        default="post_success_reachable",
        choices=("post_success", "post_success_reachable"),
        help="Sample filter for the optional follow-up bootstrap pass.",
    )
    parser.add_argument(
        "--bootstrap-followup-min-useful-bounces",
        type=int,
        default=None,
        help="Optional useful-bounce threshold override for the follow-up bootstrap pass. Defaults to --bootstrap-min-useful-bounces.",
    )
    parser.add_argument(
        "--bootstrap-followup-learning-rate",
        type=float,
        default=None,
        help="Optional learning rate override for the follow-up bootstrap pass. Defaults to --bootstrap-learning-rate.",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default="position",
        choices=(
            "position",
            "position_strike",
            "position_tilt",
            "position_strike_tilt",
            "position_strike_tilt_lift",
            "position_contact_frame",
        ),
    )
    parser.add_argument(
        "--tilt-profile",
        type=str,
        default="auto",
        choices=("auto", "custom", "early", "mid", "late", "final"),
        help="Convenience preset for position_tilt limits and regularization. 'auto' resolves to 'early'.",
    )
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--reset-xy-range", type=float, default=DEFAULT_RESET_XY_RANGE)
    parser.add_argument("--reset-velocity-xy-range", type=float, default=DEFAULT_RESET_VELOCITY_XY_RANGE)
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_RESET_VELOCITY_Z_RANGE,
    )
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    )
    parser.add_argument("--lateral-action-limit", type=float, default=None)
    parser.add_argument("--vertical-action-limit", type=float, default=None)
    parser.add_argument("--tilt-action-limit", type=float, default=None)
    parser.add_argument("--followup-lift-action-limit", type=float, default=None)
    parser.add_argument("--tracking-during-contact-scale", type=float, default=None)
    parser.add_argument("--useful-contact-outgoing-x-penalty-weight", type=float, default=None)
    parser.add_argument("--desired-outgoing-ball-velocity-x", type=float, default=None)
    parser.add_argument("--useful-contact-return-target-xy-reward-weight", type=float, default=None)
    parser.add_argument(
        "--return-target-xy-source",
        type=str,
        choices=("controller_anchor", "racket_home", "racket_position", "target_position"),
        default=None,
    )
    parser.add_argument("--return-target-xy-tolerance", type=float, default=None)
    parser.add_argument("--tilt-angle-penalty-weight", type=float, default=None)
    parser.add_argument("--tilt-action-delta-penalty-weight", type=float, default=None)
    parser.add_argument(
        "--target-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument(
        "--target-pitch-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Optional target pitch clamp applied after tilt integration. Use this for inward-only rebound A/B runs.",
    )
    parser.add_argument(
        "--initial-target-tilt",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
        help="Optional initial target tilt applied at env reset. Useful for breaking the zero-tilt symmetry in tilt A/B runs.",
    )
    parser.add_argument(
        "--strike-tilt-assist-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
        help="Optional pre-contact tilt assist limit for position_strike. The assist ramps toward center-seeking tilt and returns to neutral after contact.",
    )
    parser.add_argument(
        "--strike-tilt-assist-deadband",
        type=float,
        default=None,
        help="Deadband in meters below which position_strike tilt assist stays neutral.",
    )
    parser.add_argument(
        "--strike-tilt-ramp-pitch",
        type=float,
        default=None,
        help="Optional fixed pitch target for position_strike that ramps in only during pre-contact strike preparation and returns to neutral after contact.",
    )
    parser.add_argument(
        "--strike-tilt-ramp-xy-tolerance",
        type=float,
        default=None,
        help="Maximum XY alignment error allowed before the position_strike pitch ramp stays neutral.",
    )
    parser.add_argument(
        "--followup-strike-target-tilt",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
        help="Optional persistent target tilt used after the first useful bounce so follow-up strikes keep a nonzero inward face.",
    )
    parser.add_argument(
        "--followup-strike-contact-offset-ratio",
        type=float,
        default=None,
        help="Optional fraction of anchor correction used to bias follow-up descending strike contact points toward the strike zone center.",
    )
    parser.add_argument(
        "--followup-strike-contact-offset-max",
        type=float,
        default=None,
        help="Maximum meters of follow-up descending strike contact bias toward the strike zone center.",
    )
    parser.add_argument(
        "--followup-strike-lift-boost",
        type=float,
        default=None,
        help="Optional extra follow-up lift boost applied only after the first useful bounce.",
    )
    parser.add_argument("--contact-frame-base-strike-z-boost", type=float, default=None)
    parser.add_argument("--contact-frame-base-strike-z-offset", type=float, default=None)
    parser.add_argument("--contact-frame-base-strike-time-horizon", type=float, default=None)
    parser.add_argument(
        "--contact-frame-base-tilt-residual",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-action-penalty-weight", type=float, default=None)
    parser.add_argument("--post-contact-return-assist-weight", type=float, default=None)
    parser.add_argument("--post-contact-return-max-intercept-time", type=float, default=None)
    parser.add_argument("--next-intercept-reachable-bonus-weight", type=float, default=None)
    parser.add_argument("--easy-next-ball-reward-weight", type=float, default=None)
    parser.add_argument(
        "--trajectory-match-reward-weight",
        type=float,
        default=None,
        help="Optional contact-event reward bonus for matching the desired outgoing ball velocity.",
    )
    parser.add_argument("--next-intercept-max-time", type=float, default=None)
    parser.add_argument(
        "--include-velocity-domain-observation",
        action="store_true",
        help="Add relative velocity and racket face normal to the observation for velocity-domain rebound experiments.",
    )
    parser.add_argument(
        "--include-task-phase-observation",
        action="store_true",
        help="Add prepare/strike/return/recovery phase one-hot observation for repeated keep-up experiments.",
    )
    parser.add_argument(
        "--include-contact-context-observation",
        action="store_true",
        help="Add time-since-contact and clipped bounce-count observation signals.",
    )
    parser.add_argument(
        "--include-next-intercept-observation",
        action="store_true",
        help="Add next descending intercept and recovery-readiness observation signals.",
    )
    parser.add_argument(
        "--include-desired-outgoing-velocity-observation",
        action="store_true",
        help="Add the desired outgoing ball velocity target to the observation for trajectory-matching experiments.",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10_000,
        help="Timesteps between periodic checkpoint saves and interim evaluations. Set 0 to disable interim checkpointing.",
    )
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=10,
        help="Number of deterministic episodes used for periodic checkpoint evaluation.",
    )
    parser.add_argument(
        "--early-stop-patience-evals",
        type=int,
        default=0,
        help="Optional number of consecutive non-improving checkpoint evaluations before training stops early. Set 0 to disable.",
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=None,
        help="Optional initial log std for PPO Gaussian actions. Useful for small residual action spaces.",
    )
    parser.add_argument(
        "--zero-init-action-mean",
        action="store_true",
        help="Initialize the PPO action mean head to zero for residual action spaces.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_env_preset(args: argparse.Namespace) -> str:
    if args.preset is None:
        return "manual"

    preset_values = _ENV_PRESETS[args.preset]
    for arg_name, preset_value in preset_values.items():
        current_value = getattr(args, arg_name)
        default_value = _PRESET_MANAGED_ARG_DEFAULTS[arg_name]
        if current_value == default_value:
            setattr(args, arg_name, preset_value)
            continue
        if current_value != preset_value:
            raise ValueError(
                f"--preset {args.preset!r} conflicts with explicit --{arg_name.replace('_', '-')}={current_value!r}."
            )
    return str(args.preset)


def build_run_dir(run_name: str, output_dir: Path | None) -> Path:
    if output_dir is None:
        run_dir = PPO_RUNS_ROOT / run_name
    else:
        run_dir = resolve_output_path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def default_model_path(run_dir: Path, run_name: str) -> Path:
    return run_dir / f"{run_name}_model.zip"


def resolve_starting_checkpoint(args: argparse.Namespace, run_dir: Path, resolved_run_name: str) -> tuple[str, Path | None]:
    if args.reset_model and args.resume_from is not None:
        raise ValueError("--reset-model and --resume-from cannot be used together.")

    if args.reset_model:
        return "new", None

    if args.resume_from is not None:
        checkpoint_path = resolve_input_path(args.resume_from)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        return "resume", checkpoint_path

    checkpoint_path = default_model_path(run_dir, resolved_run_name)
    if checkpoint_path.is_file():
        return "resume", checkpoint_path
    return "new", None


def build_session_monitor_path(run_dir: Path) -> Path:
    session_index = 1
    while True:
        candidate = run_dir / f"monitor_{session_index:03d}.monitor.csv"
        if not candidate.exists():
            return candidate
        session_index += 1


def build_checkpoint_dir(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def resolve_tilt_profile(args: argparse.Namespace) -> str:
    if args.action_mode not in ("position_tilt", "position_strike_tilt", "position_strike_tilt_lift", "position_contact_frame"):
        if args.tracking_during_contact_scale is None:
            args.tracking_during_contact_scale = 0.0
        return "disabled"

    profile_name = "early" if args.tilt_profile == "auto" else args.tilt_profile
    if profile_name == "custom":
        if args.tracking_during_contact_scale is None:
            args.tracking_during_contact_scale = 0.0
        return profile_name

    profile = _TILT_PROFILES[profile_name]
    if args.tilt_action_limit is None:
        args.tilt_action_limit = float(profile["tilt_action_limit"])
    if args.target_tilt_limit is None:
        args.target_tilt_limit = tuple(profile["target_tilt_limit"])
    if args.tilt_angle_penalty_weight is None:
        args.tilt_angle_penalty_weight = float(profile["tilt_angle_penalty_weight"])
    if args.tilt_action_delta_penalty_weight is None:
        args.tilt_action_delta_penalty_weight = float(profile["tilt_action_delta_penalty_weight"])
    if args.tracking_during_contact_scale is None:
        args.tracking_during_contact_scale = 0.0
    return profile_name


def tilt_limit_ratio(args: argparse.Namespace) -> float | None:
    if args.action_mode not in ("position_tilt", "position_strike_tilt", "position_strike_tilt_lift", "position_contact_frame") or args.tilt_action_limit is None or args.target_tilt_limit is None:
        return None
    return float(args.tilt_action_limit / max(min(args.target_tilt_limit), 1.0e-6))


def env_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    env_kwargs: dict[str, object] = {
        "action_mode": args.action_mode,
        "ball_height": args.ball_height,
        "target_ball_height": args.ball_height,
        "max_episode_steps": args.max_episode_steps,
        "reset_xy_range": args.reset_xy_range,
        "reset_velocity_xy_range": args.reset_velocity_xy_range,
        "reset_velocity_z_range": tuple(args.reset_velocity_z_range),
        "success_velocity_threshold": args.success_velocity_threshold,
    }
    if args.lateral_action_limit is not None:
        env_kwargs["lateral_action_limit"] = args.lateral_action_limit
    if args.vertical_action_limit is not None:
        env_kwargs["vertical_action_limit"] = args.vertical_action_limit
    if args.tilt_action_limit is not None:
        env_kwargs["tilt_action_limit"] = args.tilt_action_limit
    if args.followup_lift_action_limit is not None:
        env_kwargs["followup_lift_action_limit"] = args.followup_lift_action_limit
    if args.tracking_during_contact_scale is not None:
        env_kwargs["tracking_during_contact_scale"] = args.tracking_during_contact_scale
    if args.useful_contact_outgoing_x_penalty_weight is not None:
        env_kwargs["useful_contact_outgoing_x_penalty_weight"] = args.useful_contact_outgoing_x_penalty_weight
    if args.desired_outgoing_ball_velocity_x is not None:
        env_kwargs["desired_outgoing_ball_velocity_x"] = args.desired_outgoing_ball_velocity_x
    if args.useful_contact_return_target_xy_reward_weight is not None:
        env_kwargs["useful_contact_return_target_xy_reward_weight"] = (
            args.useful_contact_return_target_xy_reward_weight
        )
    if args.return_target_xy_source is not None:
        env_kwargs["return_target_xy_source"] = args.return_target_xy_source
    if args.return_target_xy_tolerance is not None:
        env_kwargs["return_target_xy_tolerance"] = args.return_target_xy_tolerance
    if args.tilt_angle_penalty_weight is not None:
        env_kwargs["tilt_angle_penalty_weight"] = args.tilt_angle_penalty_weight
    if args.tilt_action_delta_penalty_weight is not None:
        env_kwargs["tilt_action_delta_penalty_weight"] = args.tilt_action_delta_penalty_weight
    if args.target_tilt_limit is not None:
        env_kwargs["target_tilt_limit"] = tuple(args.target_tilt_limit)
    if args.target_pitch_range is not None:
        env_kwargs["target_pitch_range"] = tuple(args.target_pitch_range)
    if args.initial_target_tilt is not None:
        env_kwargs["initial_target_tilt"] = tuple(args.initial_target_tilt)
    if args.strike_tilt_assist_limit is not None:
        env_kwargs["strike_tilt_assist_limit"] = tuple(args.strike_tilt_assist_limit)
    if args.strike_tilt_assist_deadband is not None:
        env_kwargs["strike_tilt_assist_deadband"] = args.strike_tilt_assist_deadband
    if args.strike_tilt_ramp_pitch is not None:
        env_kwargs["strike_tilt_ramp_pitch"] = args.strike_tilt_ramp_pitch
    if args.strike_tilt_ramp_xy_tolerance is not None:
        env_kwargs["strike_tilt_ramp_xy_tolerance"] = args.strike_tilt_ramp_xy_tolerance
    if args.followup_strike_target_tilt is not None:
        env_kwargs["followup_strike_target_tilt"] = tuple(args.followup_strike_target_tilt)
    if args.followup_strike_contact_offset_ratio is not None:
        env_kwargs["followup_strike_contact_offset_ratio"] = args.followup_strike_contact_offset_ratio
    if args.followup_strike_contact_offset_max is not None:
        env_kwargs["followup_strike_contact_offset_max"] = args.followup_strike_contact_offset_max
    if args.followup_strike_lift_boost is not None:
        env_kwargs["followup_strike_lift_boost"] = args.followup_strike_lift_boost
    if args.contact_frame_base_strike_z_boost is not None:
        env_kwargs["contact_frame_base_strike_z_boost"] = args.contact_frame_base_strike_z_boost
    if args.contact_frame_base_strike_z_offset is not None:
        env_kwargs["contact_frame_base_strike_z_offset"] = args.contact_frame_base_strike_z_offset
    if args.contact_frame_base_strike_time_horizon is not None:
        env_kwargs["contact_frame_base_strike_time_horizon"] = args.contact_frame_base_strike_time_horizon
    if args.contact_frame_base_tilt_residual is not None:
        env_kwargs["contact_frame_base_tilt_residual"] = tuple(args.contact_frame_base_tilt_residual)
    if args.contact_frame_action_penalty_weight is not None:
        env_kwargs["contact_frame_action_penalty_weight"] = args.contact_frame_action_penalty_weight
    if args.post_contact_return_assist_weight is not None:
        env_kwargs["post_contact_return_assist_weight"] = args.post_contact_return_assist_weight
    if args.post_contact_return_max_intercept_time is not None:
        env_kwargs["post_contact_return_max_intercept_time"] = args.post_contact_return_max_intercept_time
    if args.next_intercept_reachable_bonus_weight is not None:
        env_kwargs["next_intercept_reachable_bonus_weight"] = args.next_intercept_reachable_bonus_weight
    if args.easy_next_ball_reward_weight is not None:
        env_kwargs["easy_next_ball_reward_weight"] = args.easy_next_ball_reward_weight
    if args.trajectory_match_reward_weight is not None:
        env_kwargs["trajectory_match_reward_weight"] = args.trajectory_match_reward_weight
    if args.next_intercept_max_time is not None:
        env_kwargs["next_intercept_max_time"] = args.next_intercept_max_time
    if args.include_velocity_domain_observation:
        env_kwargs["include_velocity_domain_observation"] = True
    if args.include_task_phase_observation:
        env_kwargs["include_task_phase_observation"] = True
    if args.include_contact_context_observation:
        env_kwargs["include_contact_context_observation"] = True
    if args.include_next_intercept_observation:
        env_kwargs["include_next_intercept_observation"] = True
    if args.include_desired_outgoing_velocity_observation:
        env_kwargs["include_desired_outgoing_velocity_observation"] = True
    return env_kwargs


def evaluate_model(model: PPO, env_kwargs: dict[str, object], episodes: int, seed: int) -> dict[str, object]:
    env = PingPongKeepUpGymEnv(**env_kwargs)
    returns: list[float] = []
    useful_bounces: list[int] = []
    failure_counts: Counter[str] = Counter()
    for episode_index in range(episodes):
        observation, _ = env.reset(seed=seed + episode_index)
        episode_return = 0.0
        info: dict[str, object] = {}
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        returns.append(episode_return)
        useful_bounces.append(int(info.get("successful_bounce_count", 0)))
        failure_reason = info.get("failure_reason")
        if failure_reason is None:
            failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
        failure_counts[str(failure_reason)] += 1
    env.close()
    returns_array = np.asarray(returns, dtype=float)
    bounce_array = np.asarray(useful_bounces, dtype=float)
    one_or_more_useful = int(np.count_nonzero(bounce_array >= 1.0)) if bounce_array.size else 0
    two_or_more_useful = int(np.count_nonzero(bounce_array >= 2.0)) if bounce_array.size else 0
    three_or_more_useful = int(np.count_nonzero(bounce_array >= 3.0)) if bounce_array.size else 0
    ball_out_of_bounds_count = int(failure_counts.get("ball_out_of_bounds", 0))
    return {
        "episodes": episodes,
        "mean_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "episodes_with_one_or_more_useful_bounces": one_or_more_useful,
        "one_or_more_useful_bounce_rate": (one_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_two_or_more_useful_bounces": two_or_more_useful,
        "two_or_more_useful_bounce_rate": (two_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_three_or_more_useful_bounces": three_or_more_useful,
        "three_or_more_useful_bounce_rate": (three_or_more_useful / episodes) if episodes > 0 else 0.0,
        "ball_out_of_bounds_rate": (ball_out_of_bounds_count / episodes) if episodes > 0 else 0.0,
        "failure_counts": dict(failure_counts),
    }


def evaluation_sort_key(evaluation: dict[str, object]) -> tuple[float, float, int, float, float]:
    failure_counts = evaluation.get("failure_counts", {})
    if not isinstance(failure_counts, dict):
        failure_counts = {}
    return (
        float(evaluation.get("three_or_more_useful_bounce_rate", 0.0)),
        float(evaluation.get("two_or_more_useful_bounce_rate", 0.0)),
        float(evaluation.get("mean_useful_bounces", 0.0)),
        int(evaluation.get("max_useful_bounces", 0)),
        -float(failure_counts.get("ball_out_of_bounds", 0)),
        float(evaluation.get("mean_return", 0.0)),
    )


def collect_heuristic_bootstrap_dataset(
    *,
    env_kwargs: dict[str, object],
    episodes: int,
    seed: int,
    min_useful_bounces: int,
    max_samples: int,
    sample_mode: str,
) -> dict[str, object]:
    if episodes <= 0:
        return {
            "requested_episodes": episodes,
            "accepted_episodes": 0,
            "accepted_samples": 0,
            "mean_episode_useful_bounces": 0.0,
            "sample_mode": sample_mode,
            "observations": np.empty((0, 0), dtype=np.float32),
            "actions": np.empty((0, 0), dtype=np.float32),
        }
    if env_kwargs.get("action_mode") not in {"position_strike", "position_strike_tilt", "position_strike_tilt_lift", "position_contact_frame"}:
        raise ValueError(
            "Heuristic bootstrap currently requires action_mode='position_strike', 'position_strike_tilt', 'position_strike_tilt_lift', or 'position_contact_frame'."
        )
    if sample_mode not in {"episode", "post_success", "post_success_reachable"}:
        raise ValueError(f"Unsupported bootstrap sample mode: {sample_mode}")

    env = PingPongKeepUpGymEnv(**env_kwargs)
    policy = HeuristicKeepUpPolicy()
    accepted_observations: list[np.ndarray] = []
    accepted_actions: list[np.ndarray] = []
    accepted_episode_useful_bounces: list[int] = []
    accepted_episode_count = 0
    qualifying_episode_count = 0
    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            policy.reset()
            episode_samples: list[dict[str, object]] = []
            info: dict[str, object] = {}
            while True:
                action = policy.predict(env.base_env).astype(np.float32, copy=False)
                next_observation, _, terminated, truncated, info = env.step(action)
                episode_samples.append(
                    {
                        "observation": np.asarray(observation, dtype=np.float32).copy(),
                        "action": np.asarray(action, dtype=np.float32).copy(),
                        "successful_bounce_count": int(info.get("successful_bounce_count", 0)),
                        "next_intercept_reachable": bool(info.get("next_intercept_reachable", False)),
                    }
                )
                observation = next_observation
                if terminated or truncated:
                    break

            useful_bounces = int(info.get("successful_bounce_count", 0))
            if useful_bounces < min_useful_bounces:
                continue
            qualifying_episode_count += 1

            if sample_mode == "episode":
                selected_samples = episode_samples
            elif sample_mode == "post_success":
                selected_samples = [
                    sample for sample in episode_samples if int(sample["successful_bounce_count"]) > 0
                ]
            else:
                selected_samples = [
                    sample
                    for sample in episode_samples
                    if int(sample["successful_bounce_count"]) > 0 and bool(sample["next_intercept_reachable"])
                ]
            if not selected_samples:
                continue

            accepted_episode_count += 1
            accepted_episode_useful_bounces.append(useful_bounces)
            accepted_observations.extend(
                np.asarray(sample["observation"], dtype=np.float32) for sample in selected_samples
            )
            accepted_actions.extend(
                np.asarray(sample["action"], dtype=np.float32) for sample in selected_samples
            )
            if max_samples > 0 and len(accepted_observations) >= max_samples:
                accepted_observations = accepted_observations[:max_samples]
                accepted_actions = accepted_actions[:max_samples]
                break
    finally:
        env.close()

    if not accepted_observations:
        observation_shape = (0, env.base_env.observation_size)
        action_shape = (0, env.action_space.shape[0])
        return {
            "requested_episodes": episodes,
            "accepted_episodes": accepted_episode_count,
            "qualifying_episodes": qualifying_episode_count,
            "accepted_samples": 0,
            "mean_episode_useful_bounces": 0.0,
            "sample_mode": sample_mode,
            "observations": np.empty(observation_shape, dtype=np.float32),
            "actions": np.empty(action_shape, dtype=np.float32),
        }

    observations_array = np.asarray(accepted_observations, dtype=np.float32)
    actions_array = np.asarray(accepted_actions, dtype=np.float32)
    return {
        "requested_episodes": episodes,
        "accepted_episodes": accepted_episode_count,
        "qualifying_episodes": qualifying_episode_count,
        "accepted_samples": int(observations_array.shape[0]),
        "mean_episode_useful_bounces": float(np.mean(accepted_episode_useful_bounces)),
        "sample_mode": sample_mode,
        "observations": observations_array,
        "actions": actions_array,
    }


def bootstrap_actor_from_dataset(
    *,
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    if epochs <= 0 or observations.size == 0 or actions.size == 0:
        return {
            "epochs": epochs,
            "samples": int(observations.shape[0]) if observations.ndim == 2 else 0,
            "mean_loss": None,
            "last_loss": None,
        }

    optimizer = th.optim.Adam(model.policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    loss_history: list[float] = []
    model.policy.train()
    for _ in range(epochs):
        permutation = rng.permutation(observations.shape[0])
        batch_count = max(1, ceil(observations.shape[0] / batch_size))
        for batch_index in range(batch_count):
            batch_slice = permutation[batch_index * batch_size:(batch_index + 1) * batch_size]
            if batch_slice.size == 0:
                continue
            observation_tensor = th.as_tensor(observations[batch_slice], device=model.device)
            action_tensor = th.as_tensor(actions[batch_slice], device=model.device)
            distribution = model.policy.get_distribution(observation_tensor)
            predicted_action_tensor = distribution.get_actions(deterministic=True)
            loss = th.nn.functional.mse_loss(predicted_action_tensor, action_tensor)
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()
            loss_history.append(float(loss.detach().cpu().item()))

    return {
        "epochs": epochs,
        "samples": int(observations.shape[0]),
        "mean_loss": float(np.mean(loss_history)) if loss_history else None,
        "last_loss": loss_history[-1] if loss_history else None,
    }


def save_periodic_checkpoints(
    *,
    model: PPO,
    run_dir: Path,
    resolved_run_name: str,
    env_kwargs: dict[str, object],
    total_timesteps: int,
    checkpoint_interval: int,
    checkpoint_eval_episodes: int,
    early_stop_patience_evals: int,
    initial_reset_num_timesteps: bool,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object] | None, int, bool]:
    if checkpoint_interval < 0:
        raise ValueError(f"checkpoint-interval must be non-negative, got {checkpoint_interval}.")
    if checkpoint_eval_episodes < 1:
        raise ValueError(
            f"checkpoint-eval-episodes must be positive, got {checkpoint_eval_episodes}."
        )
    if early_stop_patience_evals < 0:
        raise ValueError(
            f"early-stop-patience-evals must be non-negative, got {early_stop_patience_evals}."
        )

    checkpoint_dir = build_checkpoint_dir(run_dir)
    effective_interval = total_timesteps if checkpoint_interval == 0 else checkpoint_interval
    completed_timesteps = 0
    no_improvement_evals = 0
    stopped_early = False
    best_evaluation: dict[str, object] | None = None
    best_checkpoint_record: dict[str, object] | None = None
    checkpoint_history: list[dict[str, object]] = []

    while completed_timesteps < total_timesteps:
        learn_chunk = min(effective_interval, total_timesteps - completed_timesteps)
        model.learn(
            total_timesteps=learn_chunk,
            progress_bar=False,
            reset_num_timesteps=initial_reset_num_timesteps and completed_timesteps == 0,
        )
        completed_timesteps += learn_chunk

        checkpoint_path = checkpoint_dir / f"{resolved_run_name}_step_{completed_timesteps:07d}_model"
        model.save(str(checkpoint_path))
        checkpoint_evaluation = evaluate_model(
            model,
            env_kwargs=env_kwargs,
            episodes=checkpoint_eval_episodes,
            seed=seed + 20_000,
        )
        checkpoint_record = {
            "timesteps": completed_timesteps,
            "model_path": str(checkpoint_path.with_suffix(".zip").resolve()),
            "evaluation": checkpoint_evaluation,
        }
        checkpoint_history.append(checkpoint_record)

        if best_evaluation is None or evaluation_sort_key(checkpoint_evaluation) > evaluation_sort_key(best_evaluation):
            best_evaluation = checkpoint_evaluation
            best_checkpoint_record = checkpoint_record
            best_model_path = run_dir / f"{resolved_run_name}_best_model"
            model.save(str(best_model_path))
            no_improvement_evals = 0
        else:
            no_improvement_evals += 1

        if early_stop_patience_evals > 0 and no_improvement_evals >= early_stop_patience_evals:
            stopped_early = True
            break

    return checkpoint_dir, checkpoint_history, best_checkpoint_record, completed_timesteps, stopped_early


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.total_timesteps = SMOKE_PPO_TOTAL_TIMESTEPS
        args.n_steps = SMOKE_PPO_N_STEPS
        args.batch_size = SMOKE_PPO_BATCH_SIZE
        args.n_envs = min(args.n_envs, 2)
        if args.checkpoint_interval > args.total_timesteps:
            args.checkpoint_interval = args.total_timesteps
        if args.checkpoint_eval_episodes > 2:
            args.checkpoint_eval_episodes = 2
        if args.eval_episodes > 2:
            args.eval_episodes = 2
        if args.bootstrap_heuristic_episodes > 12:
            args.bootstrap_heuristic_episodes = 12
    resolved_preset = apply_env_preset(args)
    resolved_run_name = resolve_requested_run_name(
        args.run_name,
        args.run_version,
        action_mode=args.action_mode,
        smoke=args.smoke,
    )
    resolved_tilt_profile = resolve_tilt_profile(args)
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError(f"batch-size must be <= n_steps * n_envs ({rollout_size}), got {args.batch_size}.")

    run_dir = build_run_dir(resolved_run_name, args.output_dir)
    training_mode, starting_checkpoint = resolve_starting_checkpoint(args, run_dir, resolved_run_name)
    env_kwargs = env_kwargs_from_args(args)
    config_env = PingPongKeepUpGymEnv(**env_kwargs)
    try:
        resolved_env_config = config_env.training_config()
    finally:
        config_env.close()
    vec_env = make_sb3_async_vector_env(num_envs=args.n_envs, env_kwargs=env_kwargs, seed=args.seed)
    monitor_path = build_session_monitor_path(run_dir)
    monitored_env = VecMonitor(venv=vec_env, filename=str(monitor_path))

    if starting_checkpoint is None:
        policy_kwargs = None if args.log_std_init is None else {"log_std_init": float(args.log_std_init)}
        model = PPO(
            "MlpPolicy",
            monitored_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            device=args.device,
            policy_kwargs=policy_kwargs,
        )
        if args.zero_init_action_mean:
            th.nn.init.zeros_(model.policy.action_net.weight)
            th.nn.init.zeros_(model.policy.action_net.bias)
    else:
        model = PPO.load(
            str(starting_checkpoint),
            env=monitored_env,
            device=args.device,
        )
    bootstrap_summary: dict[str, object] | None = None
    try:
        if starting_checkpoint is None and args.bootstrap_heuristic_episodes > 0 and args.bootstrap_epochs > 0:
            bootstrap_dataset = collect_heuristic_bootstrap_dataset(
                env_kwargs=env_kwargs,
                episodes=args.bootstrap_heuristic_episodes,
                seed=args.seed + 40_000,
                min_useful_bounces=args.bootstrap_min_useful_bounces,
                max_samples=args.bootstrap_max_samples,
                sample_mode=args.bootstrap_sample_mode,
            )
            bootstrap_train_summary = bootstrap_actor_from_dataset(
                model=model,
                observations=bootstrap_dataset["observations"],
                actions=bootstrap_dataset["actions"],
                epochs=args.bootstrap_epochs,
                batch_size=args.bootstrap_batch_size,
                learning_rate=args.bootstrap_learning_rate,
                seed=args.seed + 50_000,
            )
            bootstrap_summary = {
                "base": {
                    "requested_episodes": bootstrap_dataset["requested_episodes"],
                    "accepted_episodes": bootstrap_dataset["accepted_episodes"],
                    "qualifying_episodes": bootstrap_dataset["qualifying_episodes"],
                    "accepted_samples": bootstrap_dataset["accepted_samples"],
                    "min_useful_bounces": args.bootstrap_min_useful_bounces,
                    "sample_mode": bootstrap_dataset["sample_mode"],
                    "mean_episode_useful_bounces": bootstrap_dataset["mean_episode_useful_bounces"],
                    **bootstrap_train_summary,
                }
            }
            if args.bootstrap_followup_epochs > 0:
                followup_dataset = collect_heuristic_bootstrap_dataset(
                    env_kwargs=env_kwargs,
                    episodes=args.bootstrap_heuristic_episodes,
                    seed=args.seed + 45_000,
                    min_useful_bounces=(
                        args.bootstrap_min_useful_bounces
                        if args.bootstrap_followup_min_useful_bounces is None
                        else args.bootstrap_followup_min_useful_bounces
                    ),
                    max_samples=args.bootstrap_max_samples,
                    sample_mode=args.bootstrap_followup_sample_mode,
                )
                followup_train_summary = bootstrap_actor_from_dataset(
                    model=model,
                    observations=followup_dataset["observations"],
                    actions=followup_dataset["actions"],
                    epochs=args.bootstrap_followup_epochs,
                    batch_size=args.bootstrap_batch_size,
                    learning_rate=(
                        args.bootstrap_learning_rate
                        if args.bootstrap_followup_learning_rate is None
                        else args.bootstrap_followup_learning_rate
                    ),
                    seed=args.seed + 55_000,
                )
                bootstrap_summary["followup"] = {
                    "requested_episodes": followup_dataset["requested_episodes"],
                    "accepted_episodes": followup_dataset["accepted_episodes"],
                    "qualifying_episodes": followup_dataset["qualifying_episodes"],
                    "accepted_samples": followup_dataset["accepted_samples"],
                    "min_useful_bounces": (
                        args.bootstrap_min_useful_bounces
                        if args.bootstrap_followup_min_useful_bounces is None
                        else args.bootstrap_followup_min_useful_bounces
                    ),
                    "sample_mode": followup_dataset["sample_mode"],
                    "mean_episode_useful_bounces": followup_dataset["mean_episode_useful_bounces"],
                    **followup_train_summary,
                }
            else:
                bootstrap_summary["followup"] = None
        checkpoint_dir, checkpoint_history, best_checkpoint_record, completed_timesteps, stopped_early = save_periodic_checkpoints(
            model=model,
            run_dir=run_dir,
            resolved_run_name=resolved_run_name,
            env_kwargs=env_kwargs,
            total_timesteps=args.total_timesteps,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            early_stop_patience_evals=args.early_stop_patience_evals,
            initial_reset_num_timesteps=starting_checkpoint is None,
            seed=args.seed,
        )
        model_path = run_dir / f"{resolved_run_name}_model"
        model.save(str(model_path))
        evaluation = evaluate_model(model, env_kwargs=env_kwargs, episodes=args.eval_episodes, seed=args.seed + 10_000)
    finally:
        monitored_env.close()

    checkpoint_history_path = run_dir / f"{resolved_run_name}_checkpoint_evaluations.json"
    checkpoint_history_path.write_text(json.dumps(checkpoint_history, indent=2), encoding="utf-8")

    summary = {
        "run_name": resolved_run_name,
        "training_mode": training_mode,
        "starting_checkpoint": None if starting_checkpoint is None else str(starting_checkpoint.resolve()),
        "model_path": str((run_dir / f"{resolved_run_name}_model.zip").resolve()),
        "monitor_path": str(monitor_path.resolve()),
        "completed_timesteps": completed_timesteps,
        "config": {
            "preset": resolved_preset,
            "run_version": args.run_version,
            "resolved_tilt_profile": resolved_tilt_profile,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "seed": args.seed,
            "device": args.device,
            "log_std_init": args.log_std_init,
            "zero_init_action_mean": args.zero_init_action_mean,
            "checkpoint_interval": args.checkpoint_interval,
            "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
            "early_stop_patience_evals": args.early_stop_patience_evals,
            **env_kwargs,
        },
        "env_config": resolved_env_config,
        "checkpointing": {
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "checkpoint_history_path": str(checkpoint_history_path.resolve()),
            "checkpoint_count": len(checkpoint_history),
            "stopped_early": stopped_early,
            "best_checkpoint": best_checkpoint_record,
            "best_model_path": (
                None
                if best_checkpoint_record is None
                else str((run_dir / f"{resolved_run_name}_best_model.zip").resolve())
            ),
        },
        "bootstrap": bootstrap_summary,
        "evaluation": evaluation,
    }
    summary_path = run_dir / f"{resolved_run_name}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"resolved_run_name={resolved_run_name}")
    print(f"resolved_preset={resolved_preset}")
    print(f"resolved_tilt_profile={resolved_tilt_profile}")
    print(f"training_mode={training_mode}")
    print(f"completed_timesteps={completed_timesteps}")
    if starting_checkpoint is not None:
        print(f"starting_checkpoint={starting_checkpoint}")
        if args.resume_from is None and not args.reset_model:
            print("resume_note=existing_checkpoint_in_run_dir")
    resolved_tilt_limit_ratio = tilt_limit_ratio(args)
    if resolved_tilt_limit_ratio is not None:
        print(f"tilt_limit_ratio={resolved_tilt_limit_ratio:.3f}")
        if resolved_tilt_limit_ratio > 0.33:
            print("tilt_limit_warning=tilt_action_limit is large relative to target_tilt_limit and may encourage chatter")
    print(f"run_dir={run_dir}")
    print(f"model_path={run_dir / f'{resolved_run_name}_model.zip'}")
    print(f"checkpoint_history_path={checkpoint_history_path}")
    if best_checkpoint_record is not None:
        print(f"best_model_path={run_dir / f'{resolved_run_name}_best_model.zip'}")
        print(f"best_checkpoint_timesteps={best_checkpoint_record['timesteps']}")
    print(f"monitor_path={monitor_path}")
    print(f"summary_path={summary_path}")
    if bootstrap_summary is not None:
        print(
            "bootstrap "
            f"base_accepted_episodes={bootstrap_summary['base']['accepted_episodes']} "
            f"base_accepted_samples={bootstrap_summary['base']['accepted_samples']} "
            f"base_mean_loss={bootstrap_summary['base']['mean_loss']}"
        )
        if bootstrap_summary["followup"] is not None:
            print(
                "bootstrap_followup "
                f"accepted_episodes={bootstrap_summary['followup']['accepted_episodes']} "
                f"accepted_samples={bootstrap_summary['followup']['accepted_samples']} "
                f"mean_loss={bootstrap_summary['followup']['mean_loss']}"
            )
    print(
        "evaluation "
        f"mean_return={evaluation['mean_return']:.3f} "
        f"mean_useful_bounces={evaluation['mean_useful_bounces']:.3f} "
        f"max_useful_bounces={evaluation['max_useful_bounces']} "
        f"two_or_more_rate={evaluation['two_or_more_useful_bounce_rate']:.3f} "
        f"three_or_more_rate={evaluation['three_or_more_useful_bounce_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
