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
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    SMOKE_PPO_BATCH_SIZE,
    SMOKE_PPO_N_STEPS,
    SMOKE_PPO_RUN_NAME,
    SMOKE_PPO_TOTAL_TIMESTEPS,
)
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.training import make_sb3_async_vector_env
from pingpong_rl2.utils import PPO_RUNS_ROOT, resolve_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal pingpong_rl2 PPO baseline.")
    parser.add_argument("--run-name", type=str, default=DEFAULT_PPO_RUN_NAME)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_PPO_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_PPO_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PPO_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PPO_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_PPO_GAMMA)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--reset-xy-range", type=float, default=DEFAULT_RESET_XY_RANGE)
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
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


def env_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "ball_height": args.ball_height,
        "target_ball_height": args.ball_height,
        "max_episode_steps": args.max_episode_steps,
        "reset_xy_range": args.reset_xy_range,
        "success_velocity_threshold": args.success_velocity_threshold,
    }


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
        args.run_name = SMOKE_PPO_RUN_NAME if args.run_name == DEFAULT_PPO_RUN_NAME else args.run_name
        args.total_timesteps = SMOKE_PPO_TOTAL_TIMESTEPS
        args.n_steps = SMOKE_PPO_N_STEPS
        args.batch_size = SMOKE_PPO_BATCH_SIZE
        args.n_envs = min(args.n_envs, 2)
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError(f"batch-size must be <= n_steps * n_envs ({rollout_size}), got {args.batch_size}.")

    run_dir = build_run_dir(args.run_name, args.output_dir)
    env_kwargs = env_kwargs_from_args(args)
    vec_env = make_sb3_async_vector_env(num_envs=args.n_envs, env_kwargs=env_kwargs, seed=args.seed)
    monitored_env = VecMonitor(venv=vec_env, filename=str(run_dir / "monitor.csv"))

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
    try:
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        model_path = run_dir / f"{args.run_name}_model"
        model.save(str(model_path))
        evaluation = evaluate_model(model, env_kwargs=env_kwargs, episodes=args.eval_episodes, seed=args.seed + 10_000)
    finally:
        monitored_env.close()

    summary = {
        "run_name": args.run_name,
        "model_path": str((run_dir / f"{args.run_name}_model.zip").resolve()),
        "config": {
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
        "evaluation": evaluation,
    }
    summary_path = run_dir / f"{args.run_name}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"run_dir={run_dir}")
    print(f"model_path={run_dir / f'{args.run_name}_model.zip'}")
    print(f"summary_path={summary_path}")
    print(
        "evaluation "
        f"mean_return={evaluation['mean_return']:.3f} "
        f"mean_useful_bounces={evaluation['mean_useful_bounces']:.3f} "
        f"max_useful_bounces={evaluation['max_useful_bounces']}"
    )


if __name__ == "__main__":
    main()