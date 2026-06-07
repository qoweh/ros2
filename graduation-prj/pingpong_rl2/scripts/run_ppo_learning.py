from __future__ import annotations

import json
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import torch as th

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.defaults import (
    SMOKE_PPO_BATCH_SIZE,
    SMOKE_PPO_N_STEPS,
    SMOKE_PPO_TOTAL_TIMESTEPS,
)
from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.training import make_sb3_async_vector_env
from pingpong_rl2.training.bootstrap import bootstrap_actor_from_dataset, collect_heuristic_bootstrap_dataset
from pingpong_rl2.training.cli_config import (
    apply_config_overrides,
    apply_training_config,
    explicit_cli_destinations,
    load_training_config,
    normalize_config_key,
    parse_args,
    parse_config_scalar,
    values_equal,
)
from pingpong_rl2.training.curriculum import (
    ResetDistributionCurriculumCallback,
    build_reset_xy_curriculum_callback,
)
from pingpong_rl2.training.env_config import (
    apply_env_preset,
    env_kwargs_from_args,
    resolve_tilt_profile,
    tilt_limit_ratio,
)
from pingpong_rl2.training.evaluation import evaluate_model
from pingpong_rl2.training.policy_init import (
    initialize_scaled_policy_log_std,
    learn_model,
    scaled_action_log_std,
)
from pingpong_rl2.training.presets import _ENV_PRESETS, _PRESET_MANAGED_ARG_DEFAULTS, _TILT_PROFILES
from pingpong_rl2.training.run_paths import (
    build_run_dir,
    build_session_monitor_path,
    default_model_path,
    resolve_starting_model,
)
from pingpong_rl2.utils import resolve_input_path, resolve_requested_run_name


def main() -> None:
    args = parse_args()
    resolved_preset = apply_env_preset(args)
    apply_config_overrides(args, args.config_overrides)
    if args.smoke:
        args.total_timesteps = SMOKE_PPO_TOTAL_TIMESTEPS
        args.n_steps = SMOKE_PPO_N_STEPS
        args.batch_size = SMOKE_PPO_BATCH_SIZE
        args.n_envs = min(args.n_envs, 2)
        if args.eval_episodes > 2:
            args.eval_episodes = 2
        if args.bootstrap_heuristic_episodes > 12:
            args.bootstrap_heuristic_episodes = 12
        if args.bootstrap_epochs > 2:
            args.bootstrap_epochs = 2
        if args.bootstrap_followup_epochs > 1:
            args.bootstrap_followup_epochs = 1
    resolved_run_name = resolve_requested_run_name(
        args.run_name,
        args.run_version,
        action_mode=args.action_mode,
        smoke=args.smoke,
    )
    resolved_tilt_profile = resolve_tilt_profile(args)
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError(f"batch-size must be <= n_steps * n_envs ({rollout_size}), got {args.batch_size}.")

    run_dir = build_run_dir(resolved_run_name, args.output_dir)
    training_mode, starting_model_path = resolve_starting_model(args, run_dir, resolved_run_name)
    env_kwargs = env_kwargs_from_args(args)
    config_env = PingPongKeepUpGymEnv(**env_kwargs)
    try:
        resolved_env_config = config_env.training_config()
    finally:
        config_env.close()
    vec_env = make_sb3_async_vector_env(num_envs=args.n_envs, env_kwargs=env_kwargs, seed=args.seed)
    monitor_path = build_session_monitor_path(run_dir)
    monitored_env = VecMonitor(venv=vec_env, filename=str(monitor_path))
    reset_xy_curriculum_callback = build_reset_xy_curriculum_callback(args)

    scaled_log_std_summary: dict[str, object] | None = None
    if starting_model_path is None:
        policy_kwargs = None if args.log_std_init is None else {"log_std_init": float(args.log_std_init)}
        model = PPO(
            "MlpPolicy",
            monitored_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_epochs=args.n_epochs,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            seed=args.seed,
            device=args.device,
            policy_kwargs=policy_kwargs,
        )
        if args.scale_log_std_by_action_limit:
            scaled_log_std_summary = initialize_scaled_policy_log_std(
                model=model,
                ratio=args.action_std_limit_ratio,
                min_std=args.action_std_min,
                max_std=args.action_std_max,
            )
        if args.zero_init_action_mean:
            th.nn.init.zeros_(model.policy.action_net.weight)
            th.nn.init.zeros_(model.policy.action_net.bias)
    else:
        model = PPO.load(
            str(starting_model_path),
            env=monitored_env,
            device=args.device,
        )
    bootstrap_summary: dict[str, object] | None = None
    try:
        if starting_model_path is None and args.bootstrap_heuristic_episodes > 0 and args.bootstrap_epochs > 0:
            bootstrap_dataset = collect_heuristic_bootstrap_dataset(
                env_kwargs=env_kwargs,
                episodes=args.bootstrap_heuristic_episodes,
                seed=args.seed + 40_000,
                min_useful_bounces=args.bootstrap_min_useful_bounces,
                max_samples=args.bootstrap_max_samples,
                sample_mode=args.bootstrap_sample_mode,
            )
            bootstrap_train_summary = bootstrap_actor_from_dataset(
                model=model,
                observations=bootstrap_dataset["observations"],
                actions=bootstrap_dataset["actions"],
                epochs=args.bootstrap_epochs,
                batch_size=args.bootstrap_batch_size,
                learning_rate=args.bootstrap_learning_rate,
                seed=args.seed + 50_000,
            )
            bootstrap_summary = {
                "base": {
                    "requested_episodes": bootstrap_dataset["requested_episodes"],
                    "accepted_episodes": bootstrap_dataset["accepted_episodes"],
                    "qualifying_episodes": bootstrap_dataset["qualifying_episodes"],
                    "accepted_samples": bootstrap_dataset["accepted_samples"],
                    "min_useful_bounces": args.bootstrap_min_useful_bounces,
                    "sample_mode": bootstrap_dataset["sample_mode"],
                    "mean_episode_useful_bounces": bootstrap_dataset["mean_episode_useful_bounces"],
                    **bootstrap_train_summary,
                }
            }
            if args.bootstrap_followup_epochs > 0:
                followup_dataset = collect_heuristic_bootstrap_dataset(
                    env_kwargs=env_kwargs,
                    episodes=args.bootstrap_heuristic_episodes,
                    seed=args.seed + 45_000,
                    min_useful_bounces=(
                        args.bootstrap_min_useful_bounces
                        if args.bootstrap_followup_min_useful_bounces is None
                        else args.bootstrap_followup_min_useful_bounces
                    ),
                    max_samples=args.bootstrap_max_samples,
                    sample_mode=args.bootstrap_followup_sample_mode,
                )
                followup_train_summary = bootstrap_actor_from_dataset(
                    model=model,
                    observations=followup_dataset["observations"],
                    actions=followup_dataset["actions"],
                    epochs=args.bootstrap_followup_epochs,
                    batch_size=args.bootstrap_batch_size,
                    learning_rate=(
                        args.bootstrap_learning_rate
                        if args.bootstrap_followup_learning_rate is None
                        else args.bootstrap_followup_learning_rate
                    ),
                    seed=args.seed + 55_000,
                )
                bootstrap_summary["followup"] = {
                    "requested_episodes": followup_dataset["requested_episodes"],
                    "accepted_episodes": followup_dataset["accepted_episodes"],
                    "qualifying_episodes": followup_dataset["qualifying_episodes"],
                    "accepted_samples": followup_dataset["accepted_samples"],
                    "min_useful_bounces": (
                        args.bootstrap_min_useful_bounces
                        if args.bootstrap_followup_min_useful_bounces is None
                        else args.bootstrap_followup_min_useful_bounces
                    ),
                    "sample_mode": followup_dataset["sample_mode"],
                    "mean_episode_useful_bounces": followup_dataset["mean_episode_useful_bounces"],
                    **followup_train_summary,
                }
            else:
                bootstrap_summary["followup"] = None
        completed_timesteps = learn_model(
            model=model,
            total_timesteps=args.total_timesteps,
            initial_reset_num_timesteps=starting_model_path is None,
            callback=reset_xy_curriculum_callback,
        )
        model_path = run_dir / f"{resolved_run_name}_model"
        model.save(str(model_path))
        evaluation = evaluate_model(
            model,
            env_kwargs=env_kwargs,
            episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            evaluation_step_limit=args.evaluation_step_limit,
        )
    finally:
        monitored_env.close()

    summary = {
        "run_name": resolved_run_name,
        "training_mode": training_mode,
        "starting_model_path": None if starting_model_path is None else str(starting_model_path.resolve()),
        "model_path": str((run_dir / f"{resolved_run_name}_model.zip").resolve()),
        "monitor_path": str(monitor_path.resolve()),
        "completed_timesteps": completed_timesteps,
        "config": {
            "config_file": None if args.config_file is None else str(resolve_input_path(args.config_file)),
            "config_overrides": list(args.config_overrides),
            "preset": resolved_preset,
            "run_version": args.run_version,
            "resolved_tilt_profile": resolved_tilt_profile,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "n_epochs": args.n_epochs,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "seed": args.seed,
            "device": args.device,
            "log_std_init": args.log_std_init,
            "scale_log_std_by_action_limit": args.scale_log_std_by_action_limit,
            "action_std_limit_ratio": args.action_std_limit_ratio,
            "action_std_min": args.action_std_min,
            "action_std_max": args.action_std_max,
            "zero_init_action_mean": args.zero_init_action_mean,
            "bootstrap_heuristic_episodes": args.bootstrap_heuristic_episodes,
            "bootstrap_min_useful_bounces": args.bootstrap_min_useful_bounces,
            "bootstrap_max_samples": args.bootstrap_max_samples,
            "bootstrap_epochs": args.bootstrap_epochs,
            "bootstrap_batch_size": args.bootstrap_batch_size,
            "bootstrap_learning_rate": args.bootstrap_learning_rate,
            "bootstrap_sample_mode": args.bootstrap_sample_mode,
            "bootstrap_followup_epochs": args.bootstrap_followup_epochs,
            "bootstrap_followup_sample_mode": args.bootstrap_followup_sample_mode,
            "bootstrap_followup_min_useful_bounces": args.bootstrap_followup_min_useful_bounces,
            "bootstrap_followup_learning_rate": args.bootstrap_followup_learning_rate,
            "eval_episodes": args.eval_episodes,
            "evaluation_step_limit": args.evaluation_step_limit,
            "reset_xy_curriculum_enabled": args.reset_xy_curriculum_enabled,
            "reset_xy_curriculum_start": args.reset_xy_curriculum_start,
            "reset_xy_curriculum_end": args.reset_xy_curriculum_end,
            "reset_xy_curriculum_fraction": args.reset_xy_curriculum_fraction,
            "reset_xy_curriculum_update_interval": args.reset_xy_curriculum_update_interval,
            "reset_velocity_xy_curriculum_start": args.reset_velocity_xy_curriculum_start,
            "reset_velocity_xy_curriculum_end": args.reset_velocity_xy_curriculum_end,
            "reset_velocity_z_curriculum_start": args.reset_velocity_z_curriculum_start,
            "reset_velocity_z_curriculum_end": args.reset_velocity_z_curriculum_end,
            "reset_ball_angular_velocity_curriculum_start": args.reset_ball_angular_velocity_curriculum_start,
            "reset_ball_angular_velocity_curriculum_end": args.reset_ball_angular_velocity_curriculum_end,
            **env_kwargs,
        },
        "scaled_log_std_initialization": scaled_log_std_summary,
        "env_config": resolved_env_config,
        "bootstrap": bootstrap_summary,
        "evaluation": evaluation,
    }
    summary_path = run_dir / f"{resolved_run_name}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"resolved_run_name={resolved_run_name}")
    print(f"resolved_preset={resolved_preset}")
    print(f"resolved_tilt_profile={resolved_tilt_profile}")
    print(f"training_mode={training_mode}")
    print(f"completed_timesteps={completed_timesteps}")
    if starting_model_path is not None:
        print(f"starting_model_path={starting_model_path}")
        if args.resume_from is None and not args.reset_model:
            print("resume_note=existing_model_in_run_dir")
    resolved_tilt_limit_ratio = tilt_limit_ratio(args)
    if resolved_tilt_limit_ratio is not None:
        print(f"tilt_limit_ratio={resolved_tilt_limit_ratio:.3f}")
        if resolved_tilt_limit_ratio > 0.33:
            print("tilt_limit_warning=tilt_action_limit is large relative to target_tilt_limit and may encourage chatter")
    print(f"run_dir={run_dir}")
    print(f"model_path={run_dir / f'{resolved_run_name}_model.zip'}")
    print(f"monitor_path={monitor_path}")
    print(f"summary_path={summary_path}")
    if bootstrap_summary is not None:
        print(
            "bootstrap "
            f"base_accepted_episodes={bootstrap_summary['base']['accepted_episodes']} "
            f"base_accepted_samples={bootstrap_summary['base']['accepted_samples']} "
            f"base_mean_loss={bootstrap_summary['base']['mean_loss']}"
        )
        if bootstrap_summary["followup"] is not None:
            print(
                "bootstrap_followup "
                f"accepted_episodes={bootstrap_summary['followup']['accepted_episodes']} "
                f"accepted_samples={bootstrap_summary['followup']['accepted_samples']} "
                f"mean_loss={bootstrap_summary['followup']['mean_loss']}"
            )
    print(
        "evaluation "
        f"mean_return={evaluation['mean_return']:.3f} "
        f"mean_useful_bounces={evaluation['mean_useful_bounces']:.3f} "
        f"max_useful_bounces={evaluation['max_useful_bounces']} "
        f"mean_stable_cycles={evaluation['mean_stable_cycles']:.3f} "
        f"max_stable_cycles={evaluation['max_stable_cycles']} "
        f"two_or_more_rate={evaluation['two_or_more_useful_bounce_rate']:.3f} "
        f"three_or_more_rate={evaluation['three_or_more_useful_bounce_rate']:.3f} "
        f"ten_or_more_rate={evaluation['ten_or_more_useful_bounce_rate']:.3f} "
        f"thirty_or_more_rate={evaluation['thirty_or_more_useful_bounce_rate']:.3f} "
        f"stable_two_or_more_rate={evaluation['two_or_more_stable_cycle_rate']:.3f} "
        f"stable_three_or_more_rate={evaluation['three_or_more_stable_cycle_rate']:.3f} "
        f"stable_thirty_or_more_rate={evaluation['thirty_or_more_stable_cycle_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
