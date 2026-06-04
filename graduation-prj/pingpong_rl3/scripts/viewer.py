from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl3.envs import TwoBallKeepUpGymEnv
from pingpong_rl3.utils import resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pingpong_rl3 two-ball keep-up episodes.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "keep2_v1.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hold-final-seconds", type=float, default=1.0)
    return parser.parse_args()


def load_env_kwargs(config_path: Path) -> dict[str, Any]:
    with resolve_input_path(config_path).open("r", encoding="utf-8") as file:
        return dict(json.load(file).get("env", {}))


def main() -> None:
    args = parse_args()
    env = TwoBallKeepUpGymEnv(**load_env_kwargs(args.config))
    model = None if args.model_path is None else PPO.load(str(resolve_input_path(args.model_path)), device=args.device)
    observation, _ = env.reset(seed=args.seed)
    sim = env.base_env.sim
    frame_sleep = sim.model.opt.timestep * sim.n_substeps
    episode_index = 1
    episode_return = 0.0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.sync()
        while viewer.is_running():
            if model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            viewer.sync()
            time.sleep(frame_sleep)
            if not (terminated or truncated):
                continue
            print(
                f"episode={episode_index} return={episode_return:.3f} "
                f"steps={info.get('elapsed_steps')} contacts={info.get('contact_count')} "
                f"useful_bounces={info.get('useful_bounces')} failure_reason={info.get('failure_reason')}"
            )
            if episode_index >= args.episodes:
                hold_until = time.time() + max(args.hold_final_seconds, 0.0)
                while viewer.is_running() and time.time() < hold_until:
                    viewer.sync()
                    time.sleep(frame_sleep)
                break
            episode_index += 1
            episode_return = 0.0
            observation, _ = env.reset(seed=args.seed + episode_index - 1)
            viewer.sync()
    env.close()


if __name__ == "__main__":
    main()
