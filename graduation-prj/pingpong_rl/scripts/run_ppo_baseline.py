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

from pingpong_rl.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_BATCH_SIZE,
    DEFAULT_PPO_GAMMA,
    DEFAULT_PPO_LEARNING_RATE,
    DEFAULT_PPO_N_STEPS,
    DEFAULT_PPO_RUN_NAME,
    DEFAULT_PPO_TOTAL_TIMESTEPS,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
)
from pingpong_rl.envs import PingPongEEDeltaGymEnv
from pingpong_rl.training import PPOLoggingCallback
from pingpong_rl.utils import PPO_RUNS_ROOT, resolve_input_path, resolve_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PPO baseline with rollout-aligned logging.")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_PPO_TOTAL_TIMESTEPS, help="Total PPO training timesteps.")
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS, help="Env time limit.")
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT, help="Spawn height above racket_center.")
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
        help="Success threshold forwarded to the env. This script does not tune it automatically.",
    )
    parser.add_argument(
        "--target-ball-height",
        type=float,
        default=None,
        help="Optional reward target for ball height above the racket after a useful bounce.",
    )
    parser.add_argument(
        "--height-tolerance",
        type=float,
        default=None,
        help="Optional tolerance band around --target-ball-height for the height reward.",
    )
    parser.add_argument(
        "--useful-contact-velocity-z",
        type=float,
        default=None,
        help="Optional minimum useful upward contact velocity for lift reward shaping.",
    )
    parser.add_argument(
        "--target-contact-velocity-z",
        type=float,
        default=None,
        help="Optional target upward contact velocity for lift reward shaping.",
    )
    parser.add_argument(
        "--lift-reward-weight",
        type=float,
        default=None,
        help="Optional weight for the contact-time upward lift reward.",
    )
    parser.add_argument(
        "--lift-overshoot-penalty-weight",
        type=float,
        default=None,
        help="Optional penalty weight for overpowered upward contacts.",
    )
    parser.add_argument(
        "--min-active-racket-velocity-z",
        type=float,
        default=None,
        help="Optional minimum upward racket velocity required for active-hit reward/success.",
    )
    parser.add_argument(
        "--target-active-racket-velocity-z",
        type=float,
        default=None,
        help="Optional target upward racket velocity for full active-hit reward.",
    )
    parser.add_argument(
        "--min-active-racket-acceleration-z",
        type=float,
        default=None,
        help="Optional minimum upward racket acceleration required for active-hit reward/success.",
    )
    parser.add_argument(
        "--target-active-racket-acceleration-z",
        type=float,
        default=None,
        help="Optional target upward racket acceleration for full active-hit reward.",
    )
    parser.add_argument(
        "--active-hit-reward-weight",
        type=float,
        default=None,
        help="Optional weight for active upward racket motion at contact.",
    )
    parser.add_argument(
        "--passive-contact-penalty",
        type=float,
        default=None,
        help="Optional penalty applied when the ball contacts a non-active racket.",
    )
    parser.add_argument(
        "--reset-xy-range",
        type=float,
        default=0.04,
        help="Uniform reset XY offset range around racket_center. Use 0 to disable.",
    )
    parser.add_argument(
        "--reset-velocity-xy-range",
        type=float,
        default=0.02,
        help="Uniform reset XY velocity range. Use 0 to disable.",
    )
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-0.05, 0.02),
        help="Uniform reset vertical velocity range.",
    )
    parser.add_argument("--n-steps", type=int, default=DEFAULT_PPO_N_STEPS, help="PPO rollout length per update.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PPO_BATCH_SIZE, help="PPO minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PPO_LEARNING_RATE, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_PPO_GAMMA, help="Discount factor.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device passed to PPO.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/etc/ppo_runs"),
        help="Directory for model, CSV logs, TensorBoard logs, and summary JSON.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_PPO_RUN_NAME,
        help="Run name prefix for artifacts. Use ppo_smoke only for very short wiring checks.",
    )
    parser.add_argument(
        "--init-model-path",
        type=Path,
        default=None,
        help="Optional starting checkpoint. If omitted, the script resumes the current run-name model when present.",
    )
    parser.add_argument(
        "--reset-model",
        action="store_true",
        help="Ignore any existing checkpoint for this run-name and start PPO from scratch.",
    )
    return parser.parse_args()


def _build_new_model(args: argparse.Namespace, vec_env: DummyVecEnv, output_dir: Path) -> PPO:
    return PPO(
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


def _resolve_starting_checkpoint(args: argparse.Namespace, output_dir: Path) -> tuple[Path | None, bool]:
    target_model_path = output_dir / f"{args.run_name}_model.zip"
    if args.reset_model:
        return None, False
    if args.init_model_path is not None:
        return resolve_input_path(args.init_model_path), False
    if target_model_path.is_file():
        return target_model_path, True
    return None, False


def main() -> None:
    args = parse_args()
    output_root = PPO_RUNS_ROOT if args.output_dir == Path("docs/etc/ppo_runs") else resolve_output_path(args.output_dir)
    output_dir = output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.run_name}_model"
    starting_checkpoint, resuming_same_run = _resolve_starting_checkpoint(args, output_dir)

    def make_env() -> PingPongEEDeltaGymEnv:
        env_kwargs: dict[str, object] = {
            "max_episode_steps": args.max_episode_steps,
            "ball_height": args.ball_height,
            "success_velocity_threshold": args.success_velocity_threshold,
        }
        if args.target_ball_height is not None:
            env_kwargs["target_ball_height"] = args.target_ball_height
        if args.height_tolerance is not None:
            env_kwargs["height_tolerance"] = args.height_tolerance
        if args.useful_contact_velocity_z is not None:
            env_kwargs["useful_contact_velocity_z"] = args.useful_contact_velocity_z
        if args.target_contact_velocity_z is not None:
            env_kwargs["target_contact_velocity_z"] = args.target_contact_velocity_z
        if args.lift_reward_weight is not None:
            env_kwargs["lift_reward_weight"] = args.lift_reward_weight
        if args.lift_overshoot_penalty_weight is not None:
            env_kwargs["lift_overshoot_penalty_weight"] = args.lift_overshoot_penalty_weight
        if args.min_active_racket_velocity_z is not None:
            env_kwargs["min_active_racket_velocity_z"] = args.min_active_racket_velocity_z
        if args.target_active_racket_velocity_z is not None:
            env_kwargs["target_active_racket_velocity_z"] = args.target_active_racket_velocity_z
        if args.min_active_racket_acceleration_z is not None:
            env_kwargs["min_active_racket_acceleration_z"] = args.min_active_racket_acceleration_z
        if args.target_active_racket_acceleration_z is not None:
            env_kwargs["target_active_racket_acceleration_z"] = args.target_active_racket_acceleration_z
        if args.active_hit_reward_weight is not None:
            env_kwargs["active_hit_reward_weight"] = args.active_hit_reward_weight
        if args.passive_contact_penalty is not None:
            env_kwargs["passive_contact_penalty"] = args.passive_contact_penalty
        env_kwargs["reset_xy_range"] = args.reset_xy_range
        env_kwargs["reset_velocity_xy_range"] = args.reset_velocity_xy_range
        env_kwargs["reset_velocity_z_range"] = tuple(args.reset_velocity_z_range)
        return PingPongEEDeltaGymEnv(
            **env_kwargs,
        )

    vec_env = DummyVecEnv([make_env])
    if starting_checkpoint is None:
        model = _build_new_model(args, vec_env, output_dir)
        training_mode = "new"
    else:
        model = PPO.load(str(starting_checkpoint), env=vec_env, device=args.device)
        model.tensorboard_log = str(output_dir / "tensorboard")
        model.verbose = 1
        training_mode = "resume" if resuming_same_run else "init_from_checkpoint"

    callback = PPOLoggingCallback(
        output_dir=output_dir,
        run_name=args.run_name,
        summary_config={
            "total_timesteps": int(args.total_timesteps),
            "max_episode_steps": int(args.max_episode_steps),
            "ball_height": float(args.ball_height),
            "success_velocity_threshold": float(args.success_velocity_threshold),
            "target_ball_height": None if args.target_ball_height is None else float(args.target_ball_height),
            "height_tolerance": None if args.height_tolerance is None else float(args.height_tolerance),
            "useful_contact_velocity_z": None
            if args.useful_contact_velocity_z is None
            else float(args.useful_contact_velocity_z),
            "target_contact_velocity_z": None
            if args.target_contact_velocity_z is None
            else float(args.target_contact_velocity_z),
            "lift_reward_weight": None if args.lift_reward_weight is None else float(args.lift_reward_weight),
            "lift_overshoot_penalty_weight": None
            if args.lift_overshoot_penalty_weight is None
            else float(args.lift_overshoot_penalty_weight),
            "min_active_racket_velocity_z": None
            if args.min_active_racket_velocity_z is None
            else float(args.min_active_racket_velocity_z),
            "target_active_racket_velocity_z": None
            if args.target_active_racket_velocity_z is None
            else float(args.target_active_racket_velocity_z),
            "min_active_racket_acceleration_z": None
            if args.min_active_racket_acceleration_z is None
            else float(args.min_active_racket_acceleration_z),
            "target_active_racket_acceleration_z": None
            if args.target_active_racket_acceleration_z is None
            else float(args.target_active_racket_acceleration_z),
            "active_hit_reward_weight": None
            if args.active_hit_reward_weight is None
            else float(args.active_hit_reward_weight),
            "passive_contact_penalty": None
            if args.passive_contact_penalty is None
            else float(args.passive_contact_penalty),
            "reset_xy_range": float(args.reset_xy_range),
            "reset_velocity_xy_range": float(args.reset_velocity_xy_range),
            "reset_velocity_z_range": [float(value) for value in args.reset_velocity_z_range],
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "gamma": float(args.gamma),
            "training_mode": training_mode,
            "starting_checkpoint": None if starting_checkpoint is None else str(starting_checkpoint),
        },
    )
    print(f"training_mode={training_mode}")
    if starting_checkpoint is not None:
        print(f"starting_checkpoint={starting_checkpoint}")
    print(f"effective_device={model.device}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=args.run_name,
        reset_num_timesteps=not resuming_same_run,
    )
    model.save(str(model_path))
    print(f"model_saved={model_path}.zip")
    print(f"episodes_csv={output_dir / f'{args.run_name}_episodes.csv'}")
    print(f"steps_csv={output_dir / f'{args.run_name}_steps.csv'}")
    print(f"contacts_csv={output_dir / f'{args.run_name}_contacts.csv'}")
    print(f"summary_json={output_dir / f'{args.run_name}_training_summary.json'}")
    print(f"tensorboard_dir={output_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
