from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_RUN_NAME,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    default_ppo_model_candidates,
)
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.utils import PPO_RUNS_ROOT, resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved pingpong_rl2 PPO policy headlessly.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT)
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
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return resolve_input_path(model_path)
    candidates = default_ppo_model_candidates(PPO_RUNS_ROOT)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved PPO model not found: {model_path}")

    env = PingPongKeepUpGymEnv(
        ball_height=args.ball_height,
        target_ball_height=args.ball_height,
        max_episode_steps=args.max_episode_steps,
        reset_xy_range=args.reset_xy_range,
        reset_velocity_xy_range=args.reset_velocity_xy_range,
        reset_velocity_z_range=tuple(args.reset_velocity_z_range),
        success_velocity_threshold=args.success_velocity_threshold,
    )
    env_config = env.training_config()
    model = PPO.load(str(model_path))
    returns: list[float] = []
    useful_bounces: list[int] = []
    failure_counts: Counter[str] = Counter()
    summaries: list[dict[str, object]] = []

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode - 1)
            episode_return = 0.0
            info: dict[str, object] = {}
            step_count = 0
            while True:
                action, _ = model.predict(observation, deterministic=not args.stochastic)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                step_count += 1
                if terminated or truncated:
                    break
            failure_reason = info.get("failure_reason")
            if failure_reason is None:
                failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
            useful_bounce_count = int(info.get("successful_bounce_count", 0))
            failure_counts[str(failure_reason)] += 1
            returns.append(episode_return)
            useful_bounces.append(useful_bounce_count)
            episode_summary = {
                "episode": episode,
                "return": episode_return,
                "steps": step_count,
                "useful_bounces": useful_bounce_count,
                "failure_reason": failure_reason,
            }
            summaries.append(episode_summary)
            print(
                f"episode={episode} steps={step_count} return={episode_return:.3f} "
                f"useful_bounces={useful_bounce_count} failure_reason={failure_reason}"
            )
    finally:
        env.close()

    returns_array = np.asarray(returns, dtype=float)
    bounce_array = np.asarray(useful_bounces, dtype=float)
    summary = {
        "model_path": str(model_path.resolve()),
        "run_name": DEFAULT_PPO_RUN_NAME,
        "episodes": args.episodes,
        "env_config": env_config,
        "mean_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "failure_counts": dict(failure_counts),
        "episodes_detail": summaries,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
