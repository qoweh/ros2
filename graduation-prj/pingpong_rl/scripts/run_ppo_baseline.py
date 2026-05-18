from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs import PingPongEEDeltaGymEnv
from pingpong_rl.training import PPOLoggingCallback
from pingpong_rl.utils import PPO_RUNS_ROOT, resolve_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PPO baseline with rollout-aligned logging.")
    parser.add_argument("--total-timesteps", type=int, default=2048, help="Total PPO training timesteps.")
    parser.add_argument("--max-episode-steps", type=int, default=300, help="Env time limit.")
    parser.add_argument("--ball-height", type=float, default=0.22, help="Spawn height above racket_center.")
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=0.5,
        help="Success threshold forwarded to the env. This script does not tune it automatically.",
    )
    parser.add_argument("--n-steps", type=int, default=256, help="PPO rollout length per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--device", type=str, default="auto", help="Torch device passed to PPO.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/etc/ppo_runs"),
        help="Directory for model, CSV logs, TensorBoard logs, and summary JSON.",
    )
    parser.add_argument("--run-name", type=str, default="ppo_baseline", help="Run name prefix for artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = PPO_RUNS_ROOT if args.output_dir == Path("docs/etc/ppo_runs") else resolve_output_path(args.output_dir)
    output_dir = output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_env() -> PingPongEEDeltaGymEnv:
        return PingPongEEDeltaGymEnv(
            max_episode_steps=args.max_episode_steps,
            ball_height=args.ball_height,
            success_velocity_threshold=args.success_velocity_threshold,
        )

    vec_env = DummyVecEnv([make_env])
    callback = PPOLoggingCallback(
        output_dir=output_dir,
        run_name=args.run_name,
        summary_config={
            "total_timesteps": int(args.total_timesteps),
            "max_episode_steps": int(args.max_episode_steps),
            "ball_height": float(args.ball_height),
            "success_velocity_threshold": float(args.success_velocity_threshold),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
        },
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=args.device,
    )
    model.learn(total_timesteps=args.total_timesteps, callback=callback, tb_log_name=args.run_name)
    model_path = output_dir / f"{args.run_name}_model"
    model.save(str(model_path))
    print(f"model_saved={model_path}.zip")
    print(f"episodes_csv={output_dir / f'{args.run_name}_episodes.csv'}")
    print(f"steps_csv={output_dir / f'{args.run_name}_steps.csv'}")
    print(f"contacts_csv={output_dir / f'{args.run_name}_contacts.csv'}")
    print(f"summary_json={output_dir / f'{args.run_name}_training_summary.json'}")
    print(f"tensorboard_dir={output_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()