from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.controllers import HeuristicKeepUpPolicy
from pingpong_rl2.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
)
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.utils import resolve_env_kwargs_for_model, resolve_requested_run_name, resolve_saved_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fresh pingpong_rl2 evaluation episodes in the MuJoCo viewer.")
    parser.add_argument("--mode", type=str, default="policy", choices=("policy", "zero_action", "heuristic"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-version", type=str, default=None)
    parser.add_argument(
        "--best-model",
        action="store_true",
        help="When used with --run-name/--run-version, load <run>_best_model.zip from the training summary instead of the final model.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--ball-height", type=float, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--reset-xy-range", type=float, default=None)
    parser.add_argument("--reset-velocity-xy-range", type=float, default=None)
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--hold-final-seconds", type=float, default=1.5)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_run_name = None if args.run_name is None else resolve_requested_run_name(args.run_name, args.run_version)
    configured_model_path: Path | None = None
    if args.mode == "policy" or args.model_path is not None or resolved_run_name is not None:
        configured_model_path = resolve_saved_model_path(
            args.model_path,
            resolved_run_name,
            prefer_best_model=args.best_model,
        )

    env_kwargs = resolve_env_kwargs_for_model(
        configured_model_path,
        ball_height=args.ball_height,
        max_episode_steps=args.max_episode_steps,
        reset_xy_range=args.reset_xy_range,
        reset_velocity_xy_range=args.reset_velocity_xy_range,
        reset_velocity_z_range=args.reset_velocity_z_range,
    )
    if args.mode == "heuristic" and configured_model_path is None:
        env_kwargs.update(
            {
                "action_mode": "position_strike",
                "strike_tilt_ramp_pitch": -0.03,
                "strike_tilt_ramp_xy_tolerance": 0.04,
                "post_contact_return_assist_weight": 0.5,
                "post_contact_return_max_intercept_time": 0.6,
                "include_task_phase_observation": True,
                "include_contact_context_observation": True,
                "include_next_intercept_observation": True,
            }
        )
    env = PingPongKeepUpGymEnv(**env_kwargs)
    model = None
    heuristic_policy = None
    if args.mode == "policy":
        model_path = resolve_saved_model_path(
            args.model_path,
            resolved_run_name,
            prefer_best_model=args.best_model,
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved PPO model not found: {model_path}")
        model = PPO.load(str(model_path))
        print(f"render_model={model_path}")
    elif args.mode == "heuristic":
        heuristic_policy = HeuristicKeepUpPolicy()
        print("render_mode=heuristic")
    else:
        print("render_mode=zero_action")

    observation, _ = env.reset(seed=args.seed)
    sim = env.base_env.sim
    frame_sleep = sim.model.opt.timestep * sim.n_substeps
    episode_index = 1
    episode_return = 0.0
    episode_steps = 0

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.sync()
            while viewer.is_running():
                if model is None:
                    if heuristic_policy is None:
                        action = np.zeros(env.action_space.shape, dtype=np.float32)
                    else:
                        action = heuristic_policy.predict(env.base_env).astype(np.float32, copy=False)
                else:
                    action, _ = model.predict(observation, deterministic=not args.stochastic)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_steps += 1
                viewer.sync()
                time.sleep(frame_sleep)
                if not (terminated or truncated):
                    continue
                print(
                    f"episode={episode_index} steps={episode_steps} return={episode_return:.3f} "
                    f"contacts={info.get('contact_count', 0)} "
                    f"useful_bounces={info.get('successful_bounce_count', 0)} "
                    f"failure_reason={info.get('failure_reason')}"
                )
                if episode_index >= args.episodes:
                    hold_until = time.time() + max(args.hold_final_seconds, 0.0)
                    while viewer.is_running() and time.time() < hold_until:
                        viewer.sync()
                        time.sleep(frame_sleep)
                    break
                episode_index += 1
                episode_return = 0.0
                episode_steps = 0
                if heuristic_policy is not None:
                    heuristic_policy.reset()
                observation, _ = env.reset(seed=args.seed + episode_index - 1)
                viewer.sync()
    finally:
        env.close()


if __name__ == "__main__":
    main()
