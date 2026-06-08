from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_BATCH_SIZE,
    DEFAULT_PPO_CURRICULUM,
    DEFAULT_PPO_GAMMA,
    DEFAULT_PPO_LEARNING_RATE,
    DEFAULT_PPO_N_STEPS,
    DEFAULT_PPO_RUN_NAME,
    DEFAULT_PPO_TOTAL_TIMESTEPS,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
)
from pingpong_rl.envs import PingPongEEDeltaGymEnv
from pingpong_rl.training import CurriculumCallback, PPOLoggingCallback, curriculum_names
from pingpong_rl.utils import PPO_RUNS_ROOT, resolve_input_path, resolve_output_path


def _schedule_value(value: object) -> float | None:
    if value is None:
        return None
    if callable(value):
        return float(value(1.0))
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PPO baseline with rollout-aligned logging.")
    parser.add_argument(
        "--action-mode",
        type=str,
        default="position_tilt",
        choices=("position", "position_tilt"),
        help="Use position-only actions or add limited pitch/roll residual actions.",
    )
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
        help="Optional minimum keep-up target height above the racket used by the dynamic height reward band.",
    )
    parser.add_argument(
        "--height-tolerance",
        type=float,
        default=None,
        help="Optional tolerance band around the dynamic keep-up target height.",
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
        "--target-rebound-vertical-ratio",
        type=float,
        default=None,
        help="Optional target fraction of post-contact ball speed that should remain vertical.",
    )
    parser.add_argument(
        "--rebound-direction-reward-weight",
        type=float,
        default=None,
        help="Optional weight for contact-time rebound direction shaping toward vertical keep-up.",
    )
    parser.add_argument(
        "--tracking-assist-weight",
        type=float,
        default=None,
        help="Optional blend weight for strike-zone keep-up target assistance under the descending ball.",
    )
    parser.add_argument(
        "--tracking-assist-preview-time",
        type=float,
        default=None,
        help="Optional preview horizon used by the strike-zone tracking assist.",
    )
    parser.add_argument(
        "--reset-xy-range",
        type=float,
        default=None,
        help="Optional uniform reset XY offset range around racket_center. If omitted, use the env/curriculum defaults.",
    )
    parser.add_argument(
        "--reset-ball-height-range",
        type=float,
        default=0.0,
        help="Uniform reset height offset range around --ball-height. Use 0 to disable.",
    )
    parser.add_argument(
        "--reset-velocity-xy-range",
        type=float,
        default=0.01,
        help="Uniform reset XY velocity range. Use 0 to disable.",
    )
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-0.02, 0.01),
        help="Uniform reset vertical velocity range.",
    )
    parser.add_argument("--n-steps", type=int, default=DEFAULT_PPO_N_STEPS, help="PPO rollout length per update.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PPO_BATCH_SIZE, help="PPO minibatch size.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PPO_LEARNING_RATE, help="PPO learning rate.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_PPO_GAMMA, help="Discount factor.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device passed to PPO.")
    parser.add_argument(
        "--curriculum",
        type=str,
        default=DEFAULT_PPO_CURRICULUM,
        choices=curriculum_names(),
        help="Training curriculum preset. Use 'none' to keep a fixed env throughout training.",
    )
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
            "action_mode": args.action_mode,
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
        if args.target_rebound_vertical_ratio is not None:
            env_kwargs["target_rebound_vertical_ratio"] = args.target_rebound_vertical_ratio
        if args.rebound_direction_reward_weight is not None:
            env_kwargs["rebound_direction_reward_weight"] = args.rebound_direction_reward_weight
        if args.tracking_assist_weight is not None:
            env_kwargs["tracking_assist_weight"] = args.tracking_assist_weight
        if args.tracking_assist_preview_time is not None:
            env_kwargs["tracking_assist_preview_time"] = args.tracking_assist_preview_time
        env_kwargs["reset_ball_height_range"] = args.reset_ball_height_range
        if args.reset_xy_range is not None:
            env_kwargs["reset_xy_range"] = args.reset_xy_range
        env_kwargs["reset_velocity_xy_range"] = args.reset_velocity_xy_range
        env_kwargs["reset_velocity_z_range"] = tuple(args.reset_velocity_z_range)
        return PingPongEEDeltaGymEnv(
            **env_kwargs,
        )

    vec_env = DummyVecEnv([make_env])
    effective_env_config = vec_env.env_method("training_config")[0]
    if starting_checkpoint is None:
        model = _build_new_model(args, vec_env, output_dir)
        training_mode = "new"
    else:
        model = PPO.load(str(starting_checkpoint), env=vec_env, device=args.device)
        model.tensorboard_log = str(output_dir / "tensorboard")
        model.verbose = 1
        training_mode = "resume" if resuming_same_run else "init_from_checkpoint"

    reward_config = effective_env_config["reward_shaping"]
    reset_config = effective_env_config["reset_randomization"]
    core_config = effective_env_config["core"]
    controller_config = effective_env_config["controller"]
    effective_ppo_config = {
        "policy": "MlpPolicy",
        "device": str(model.device),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "learning_rate": float(args.learning_rate),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "n_epochs": int(model.n_epochs),
        "normalize_advantage": bool(model.normalize_advantage),
        "clip_range": _schedule_value(model.clip_range),
        "clip_range_vf": _schedule_value(model.clip_range_vf),
        "policy_kwargs": {} if model.policy_kwargs is None else dict(model.policy_kwargs),
    }

    callback_parts = []
    if args.curriculum != "none":
        callback_parts.append(
            CurriculumCallback(
                curriculum_name=args.curriculum,
                total_timesteps=args.total_timesteps,
                verbose=1,
            )
        )

    callback_parts.append(
        PPOLoggingCallback(
        output_dir=output_dir,
        run_name=args.run_name,
        summary_config={
            "action_mode": str(core_config["action_mode"]),
            "total_timesteps": int(args.total_timesteps),
            "max_episode_steps": int(core_config["max_episode_steps"]),
            "ball_height": float(core_config["ball_height"]),
            "curriculum": args.curriculum,
            "success_velocity_threshold": float(core_config["success_velocity_threshold"]),
            "target_ball_height": float(reward_config["target_ball_height"]),
            "target_ball_height_reference": str(reward_config["target_ball_height_reference"]),
            "height_tolerance": float(reward_config["height_tolerance"]),
            "useful_contact_velocity_z": float(reward_config["useful_contact_velocity_z"]),
            "target_contact_velocity_z": float(reward_config["target_contact_velocity_z"]),
            "lift_reward_weight": float(reward_config["lift_reward_weight"]),
            "lift_overshoot_penalty_weight": float(reward_config["lift_overshoot_penalty_weight"]),
            "min_active_racket_velocity_z": float(reward_config["min_active_racket_velocity_z"]),
            "target_active_racket_velocity_z": float(reward_config["target_active_racket_velocity_z"]),
            "min_active_racket_acceleration_z": float(reward_config["min_active_racket_acceleration_z"]),
            "target_active_racket_acceleration_z": float(reward_config["target_active_racket_acceleration_z"]),
            "active_hit_reward_weight": float(reward_config["active_hit_reward_weight"]),
            "passive_contact_penalty": float(reward_config["passive_contact_penalty"]),
            "tracking_alignment_reward_weight": float(reward_config["tracking_alignment_reward_weight"]),
            "contact_centering_reward_weight": float(reward_config["contact_centering_reward_weight"]),
            "contact_centering_radius": float(reward_config["contact_centering_radius"]),
            "target_rebound_vertical_ratio": float(reward_config["target_rebound_vertical_ratio"]),
            "rebound_direction_reward_weight": float(reward_config["rebound_direction_reward_weight"]),
            "tracking_assist_weight": float(controller_config["tracking_assist_weight"]),
            "tracking_assist_preview_time": float(controller_config["tracking_assist_preview_time"]),
            "target_tilt_limit": list(controller_config["target_tilt_limit"]),
            "tilt_action_limit": float(core_config["tilt_action_limit"]),
            "reset_xy_range": float(reset_config["reset_xy_range"]),
            "reset_velocity_xy_range": float(reset_config["reset_velocity_xy_range"]),
            "reset_velocity_z_range": list(reset_config["reset_velocity_z_range"]),
            "n_steps": int(effective_ppo_config["n_steps"]),
            "batch_size": int(effective_ppo_config["batch_size"]),
            "learning_rate": float(effective_ppo_config["learning_rate"]),
            "gamma": float(effective_ppo_config["gamma"]),
            "effective_env": effective_env_config,
            "effective_ppo": effective_ppo_config,
            "training_mode": training_mode,
            "starting_checkpoint": None if starting_checkpoint is None else str(starting_checkpoint),
        },
        )
    )
    callback = callback_parts[0] if len(callback_parts) == 1 else CallbackList(callback_parts)
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
