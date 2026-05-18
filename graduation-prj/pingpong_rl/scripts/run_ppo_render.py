from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
from stable_baselines3 import PPO

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_RUN_NAME,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    default_ppo_model_candidates,
)
from pingpong_rl.envs import PingPongEEDeltaGymEnv
from pingpong_rl.utils import PPO_RUNS_ROOT, resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a saved PPO policy in the MuJoCo passive viewer.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to a saved Stable-Baselines3 PPO zip model. Default: ppo_baseline model if present.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to replay before exit.")
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="Env time limit.")
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT, help="Spawn height above racket_center.")
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
        help="Success threshold forwarded to the env. This script does not tune it automatically.",
    )
    parser.add_argument(
        "--hold-final-seconds",
        type=float,
        default=1.5,
        help="How long to keep the final frame visible after the last episode finishes.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of using deterministic policy evaluation.",
    )
    return parser.parse_args()


def _resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        return resolve_input_path(model_path)

    candidates = default_ppo_model_candidates(PPO_RUNS_ROOT)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _episode_summary(index: int, episode_return: float, episode_steps: int, info: dict[str, object]) -> str:
    return (
        f"episode={index} steps={episode_steps} return={episode_return:.4f} "
        f"terminated={bool(info.get('terminated', False))} "
        f"truncated={bool(info.get('truncated', False))} "
        f"success_reason={info.get('success_reason')} "
        f"failure_reason={info.get('failure_reason')}"
    )


def main() -> None:
    args = parse_args()
    model_path = _resolve_model_path(args.model_path)
    if args.episodes < 1:
        raise ValueError(f"episodes must be positive, got {args.episodes}.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved PPO model not found: {model_path}")

    env = PingPongEEDeltaGymEnv(
        max_episode_steps=args.max_episode_steps,
        ball_height=args.ball_height,
        success_velocity_threshold=args.success_velocity_threshold,
    )
    model = PPO.load(str(model_path))
    obs, _ = env.reset()
    sim = env.base_env.sim
    frame_sleep = sim.model.opt.timestep * sim.n_substeps

    print(f"render_model={model_path}")
    print(
        f"episodes={args.episodes} deterministic={not args.stochastic} "
        f"max_episode_steps={args.max_episode_steps} ball_height={args.ball_height}"
    )
    print(f"default_run_name={DEFAULT_PPO_RUN_NAME}")
    print("Close the MuJoCo viewer window to stop early.")

    episode_index = 1
    episode_return = 0.0
    episode_steps = 0

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.sync()
            while viewer.is_running():
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_steps += 1

                viewer.sync()
                time.sleep(frame_sleep)

                if not (terminated or truncated):
                    continue

                print(_episode_summary(episode_index, episode_return, episode_steps, info))
                if episode_index >= args.episodes:
                    hold_until = time.time() + max(args.hold_final_seconds, 0.0)
                    while viewer.is_running() and time.time() < hold_until:
                        viewer.sync()
                        time.sleep(frame_sleep)
                    break

                episode_index += 1
                episode_return = 0.0
                episode_steps = 0
                obs, _ = env.reset()
                viewer.sync()
    finally:
        env.close()


if __name__ == "__main__":
    main()