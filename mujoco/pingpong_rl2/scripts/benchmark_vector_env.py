from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.training import make_gym_vector_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pingpong_rl2 vector environment stepping throughput.")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--vector-mode", type=str, default="async", choices=("async", "sync"))
    return parser.parse_args()


def main() -> None:
    # 학습 전에 vector env step 처리량을 재서 n_envs/vector_mode 선택을 점검한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/vector_env.py:25
    args = parse_args()
    vector_env = make_gym_vector_env(num_envs=args.n_envs, vector_mode=args.vector_mode)
    try:
        observations, _ = vector_env.reset(seed=args.seed)
        start_time = time.perf_counter()
        completed_episodes = 0
        for _ in range(args.steps):
            actions = np.stack([vector_env.single_action_space.sample() for _ in range(args.n_envs)])
            observations, rewards, terminations, truncations, infos = vector_env.step(actions)
            done_mask = np.logical_or(terminations, truncations)
            completed_episodes += int(np.count_nonzero(done_mask))
            if np.any(done_mask):
                observations, _ = vector_env.reset(options={"reset_mask": done_mask})
        elapsed = time.perf_counter() - start_time
    finally:
        vector_env.close()

    env_steps = args.steps * args.n_envs
    print(f"vector_mode={args.vector_mode}")
    print(f"n_envs={args.n_envs} steps={args.steps} env_steps={env_steps}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"env_steps_per_second={env_steps / max(elapsed, 1.0e-9):.2f}")
    print(f"completed_episodes={completed_episodes}")


if __name__ == "__main__":
    main()
