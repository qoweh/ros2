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
    DEFAULT_PPO_RUN_NAME,
)
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.utils import (
    infer_run_name_from_model_path,
    resolve_env_kwargs_for_model,
    resolve_requested_run_name,
    resolve_saved_model_path,
)

_UNLIMITED_EVALUATION_STEP_LIMIT = 3_600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved pingpong_rl2 PPO policy headlessly.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-version", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--ball-height", type=float, default=None)
    parser.add_argument("--reset-ball-height-range", type=float, default=None)
    parser.add_argument(
        "--reset-ball-height-bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--reset-xy-range", type=float, default=None)
    parser.add_argument("--reset-xy-sampling", type=str, choices=("square", "disk"), default=None)
    parser.add_argument("--reset-velocity-xy-range", type=float, default=None)
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--reset-ball-angular-velocity-range", type=float, default=None)
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--episode-step-limit",
        type=int,
        default=None,
        help="Evaluation-only safety cap for unlimited envs. Defaults to 3600 steps when max_episode_steps is unlimited.",
    )
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    # 모델 위치를 확정하고, 학습 summary JSON에 저장된 env kwargs를 기본값으로 복원한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py:144
    # LINK: pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py:186
    args = parse_args()
    resolved_run_name = None
    if args.run_name is not None:
        resolved_run_name = resolve_requested_run_name(args.run_name, args.run_version)
    model_path = resolve_saved_model_path(args.model_path, resolved_run_name)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved PPO model not found: {model_path}")

    run_name = infer_run_name_from_model_path(model_path)
    env_kwargs = resolve_env_kwargs_for_model(
        model_path,
        ball_height=args.ball_height,
        max_episode_steps=args.max_episode_steps,
        reset_ball_height_range=args.reset_ball_height_range,
        reset_ball_height_bounds=args.reset_ball_height_bounds,
        reset_xy_range=args.reset_xy_range,
        reset_xy_sampling=args.reset_xy_sampling,
        reset_velocity_xy_range=args.reset_velocity_xy_range,
        reset_velocity_z_range=args.reset_velocity_z_range,
        reset_ball_angular_velocity_range=args.reset_ball_angular_velocity_range,
        success_velocity_threshold=args.success_velocity_threshold,
    )

    # 평가 전용 안전 cap이다. 학습 환경이 무제한 horizon이면 기본 3600 step에서 끊는다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:17
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:53
    env = PingPongKeepUpGymEnv(**env_kwargs)
    env_config = env.training_config()
    if args.episode_step_limit is None:
        episode_step_limit = _UNLIMITED_EVALUATION_STEP_LIMIT if env.base_env.max_episode_steps is None else None
    else:
        episode_step_limit = None if args.episode_step_limit <= 0 else int(args.episode_step_limit)
    model = PPO.load(str(model_path))
    returns: list[float] = []
    contact_counts: list[int] = []
    useful_bounces: list[int] = []
    stable_cycles: list[int] = []
    failure_counts: Counter[str] = Counter()
    summaries: list[dict[str, object]] = []

    try:
        # 단일 env에서 episode를 반복 실행하며 PPO action, reward, terminal info만 모은다.
        # LINK: pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:57
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
                if not terminated and not truncated and episode_step_limit is not None and step_count >= episode_step_limit:
                    truncated = True
                    info = dict(info)
                    info["truncated"] = True
                    info["evaluation_step_limit"] = episode_step_limit
                if terminated or truncated:
                    break
            failure_reason = info.get("failure_reason")
            if failure_reason is None:
                failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
            contact_count = int(info.get("contact_count", 0))
            useful_bounce_count = int(info.get("successful_bounce_count", 0))
            stable_cycle_count = int(info.get("stable_cycle_count", useful_bounce_count))
            failure_counts[str(failure_reason)] += 1
            returns.append(episode_return)
            contact_counts.append(contact_count)
            useful_bounces.append(useful_bounce_count)
            stable_cycles.append(stable_cycle_count)
            episode_summary = {
                "episode": episode,
                "return": episode_return,
                "steps": step_count,
                "contact_count": contact_count,
                "useful_bounces": useful_bounce_count,
                "stable_cycles": stable_cycle_count,
                "failure_reason": failure_reason,
            }
            summaries.append(episode_summary)
            print(
                f"episode={episode} steps={step_count} return={episode_return:.3f} "
                f"contacts={contact_count} "
                f"useful_bounces={useful_bounce_count} "
                f"stable_cycles={stable_cycle_count} failure_reason={failure_reason}"
            )
    finally:
        env.close()

    # 발표/보고서에 바로 넣기 좋은 scalar rate와 episode별 세부 결과로 압축한다.
    returns_array = np.asarray(returns, dtype=float)
    contact_array = np.asarray(contact_counts, dtype=float)
    bounce_array = np.asarray(useful_bounces, dtype=float)
    stable_cycle_array = np.asarray(stable_cycles, dtype=float)
    summary = {
        "model_path": str(model_path.resolve()),
        "run_name": run_name,
        "episodes": args.episodes,
        "env_config": env_config,
        "episode_step_limit": episode_step_limit,
        "mean_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "mean_contacts": float(contact_array.mean()) if contact_array.size else 0.0,
        "max_contacts": int(contact_array.max()) if contact_array.size else 0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "mean_stable_cycles": float(stable_cycle_array.mean()) if stable_cycle_array.size else 0.0,
        "max_stable_cycles": int(stable_cycle_array.max()) if stable_cycle_array.size else 0,
        "one_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 1.0)) if bounce_array.size else 0.0,
        "two_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 2.0)) if bounce_array.size else 0.0,
        "three_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 3.0)) if bounce_array.size else 0.0,
        "ten_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 10.0)) if bounce_array.size else 0.0,
        "twenty_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 20.0)) if bounce_array.size else 0.0,
        "thirty_or_more_useful_bounce_rate": float(np.mean(bounce_array >= 30.0)) if bounce_array.size else 0.0,
        "one_or_more_stable_cycle_rate": (
            float(np.mean(stable_cycle_array >= 1.0)) if stable_cycle_array.size else 0.0
        ),
        "two_or_more_stable_cycle_rate": (
            float(np.mean(stable_cycle_array >= 2.0)) if stable_cycle_array.size else 0.0
        ),
        "three_or_more_stable_cycle_rate": (
            float(np.mean(stable_cycle_array >= 3.0)) if stable_cycle_array.size else 0.0
        ),
        "ten_or_more_stable_cycle_rate": (
            float(np.mean(stable_cycle_array >= 10.0)) if stable_cycle_array.size else 0.0
        ),
        "twenty_or_more_stable_cycle_rate": (
            float(np.mean(stable_cycle_array >= 20.0)) if stable_cycle_array.size else 0.0
        ),
        "thirty_or_more_stable_cycle_rate": (
            float(np.mean(stable_cycle_array >= 30.0)) if stable_cycle_array.size else 0.0
        ),
        "failure_counts": dict(failure_counts),
        "episodes_detail": summaries,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
