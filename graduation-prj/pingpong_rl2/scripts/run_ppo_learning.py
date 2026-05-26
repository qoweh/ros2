from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal pingpong_rl2 PPO baseline.")
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
    parser.add_argument("--action-mode", type=str, default="position", choices=("position", "position_tilt"))
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
    parser.add_argument(
        "--target-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


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
    if args.target_tilt_limit is not None:
        env_kwargs["target_tilt_limit"] = tuple(args.target_tilt_limit)
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
    return {
        "episodes": episodes,
        "mean_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "failure_counts": dict(failure_counts),
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.total_timesteps = SMOKE_PPO_TOTAL_TIMESTEPS
        args.n_steps = SMOKE_PPO_N_STEPS
        args.batch_size = SMOKE_PPO_BATCH_SIZE
        args.n_envs = min(args.n_envs, 2)
    resolved_run_name = resolve_requested_run_name(
        args.run_name,
        args.run_version,
        action_mode=args.action_mode,
        smoke=args.smoke,
    )
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
        )
    else:
        model = PPO.load(
            str(starting_checkpoint),
            env=monitored_env,
            device=args.device,
        )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=False,
            reset_num_timesteps=starting_checkpoint is None,
        )
        model_path = run_dir / f"{resolved_run_name}_model"
        model.save(str(model_path))
        evaluation = evaluate_model(model, env_kwargs=env_kwargs, episodes=args.eval_episodes, seed=args.seed + 10_000)
    finally:
        monitored_env.close()

    summary = {
        "run_name": resolved_run_name,
        "training_mode": training_mode,
        "starting_checkpoint": None if starting_checkpoint is None else str(starting_checkpoint.resolve()),
        "model_path": str((run_dir / f"{resolved_run_name}_model.zip").resolve()),
        "monitor_path": str(monitor_path.resolve()),
        "config": {
            "run_version": args.run_version,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "seed": args.seed,
            "device": args.device,
            **env_kwargs,
        },
        "env_config": resolved_env_config,
        "evaluation": evaluation,
    }
    summary_path = run_dir / f"{resolved_run_name}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"resolved_run_name={resolved_run_name}")
    print(f"training_mode={training_mode}")
    if starting_checkpoint is not None:
        print(f"starting_checkpoint={starting_checkpoint}")
    print(f"run_dir={run_dir}")
    print(f"model_path={run_dir / f'{resolved_run_name}_model.zip'}")
    print(f"monitor_path={monitor_path}")
    print(f"summary_path={summary_path}")
    print(
        "evaluation "
        f"mean_return={evaluation['mean_return']:.3f} "
        f"mean_useful_bounces={evaluation['mean_useful_bounces']:.3f} "
        f"max_useful_bounces={evaluation['max_useful_bounces']}"
    )


if __name__ == "__main__":
    main()
