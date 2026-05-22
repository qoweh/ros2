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

from pingpong_rl2.defaults import DEFAULT_BALL_HEIGHT, DEFAULT_MAX_EPISODE_STEPS, default_ppo_model_candidates
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.utils import PPO_RUNS_ROOT, resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a pingpong_rl2 policy or zero-action rollout in the MuJoCo viewer.")
    parser.add_argument("--mode", type=str, default="policy", choices=("policy", "zero_action"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--hold-final-seconds", type=float, default=1.5)
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
    env = PingPongKeepUpGymEnv(ball_height=args.ball_height, target_ball_height=args.ball_height, max_episode_steps=args.max_episode_steps)
    model = None
    if args.mode == "policy":
        model_path = resolve_model_path(args.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved PPO model not found: {model_path}")
        model = PPO.load(str(model_path))
        print(f"render_model={model_path}")
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
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
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
                observation, _ = env.reset(seed=args.seed + episode_index - 1)
                viewer.sync()
    finally:
        env.close()


if __name__ == "__main__":
    main()