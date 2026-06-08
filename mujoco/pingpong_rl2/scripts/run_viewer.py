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
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Optional MuJoCo scene XML override. Saved model env_config scene_path is used when this is omitted.",
    )
    parser.add_argument("--ball-height", type=float, default=None)
    parser.add_argument(
        "--target-ball-height",
        type=float,
        default=None,
        help="Override only the desired post-contact apex height. --ball-height still controls reset height.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
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
        "--keepup-target-xy-offset",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
    )
    parser.add_argument("--post-contact-return-z-offset", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-target-gain", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-target-max", type=float, default=None)
    parser.add_argument("--contact-frame-planner-enabled", action="store_true")
    parser.add_argument(
        "--disable-contact-frame-planner-hold-during-descent",
        action="store_false",
        dest="contact_frame_planner_hold_during_descent",
        default=True,
    )
    parser.add_argument("--contact-frame-planner-min-intercept-time", type=float, default=None)
    parser.add_argument("--contact-frame-planner-max-intercept-time", type=float, default=None)
    parser.add_argument("--contact-frame-planner-target-apex-z-offset", type=float, default=None)
    parser.add_argument("--controller-velocity-gain", type=float, default=None)
    parser.add_argument("--controller-velocity-feedback-gain", type=float, default=None)
    parser.add_argument("--controller-max-velocity-step", type=float, default=None)
    parser.add_argument("--contact-frame-trajectory-tilt-gain", type=float, default=None)
    parser.add_argument(
        "--contact-frame-trajectory-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-trajectory-tilt-deadband", type=float, default=None)
    parser.add_argument("--require-reachable-next-intercept-for-success", action="store_true")
    parser.add_argument("--min-easy-next-ball-score-for-success", type=float, default=None)
    parser.add_argument("--terminate-on-nonuseful-contact", action="store_true")
    parser.add_argument("--next-intercept-xy-error-penalty-weight", type=float, default=None)
    parser.add_argument("--post-contact-lateral-velocity-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-xy-error-penalty-weight", type=float, default=None)
    parser.add_argument("--nonuseful-contact-penalty-weight", type=float, default=None)
    parser.add_argument("--hold-final-seconds", type=float, default=1.5)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    # policy/heuristic/zero_action 중 action source를 고르고, 모델이 있으면 저장된 env 설정을 복원한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py:144
    # LINK: pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py:186
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
        scene_path=args.scene_path,
        ball_height=args.ball_height,
        target_ball_height=args.target_ball_height,
        max_episode_steps=args.max_episode_steps,
        reset_ball_height_range=args.reset_ball_height_range,
        reset_ball_height_bounds=args.reset_ball_height_bounds,
        reset_xy_range=args.reset_xy_range,
        reset_xy_sampling=args.reset_xy_sampling,
        reset_velocity_xy_range=args.reset_velocity_xy_range,
        reset_velocity_z_range=args.reset_velocity_z_range,
        reset_ball_angular_velocity_range=args.reset_ball_angular_velocity_range,
    )
    if args.keepup_target_xy_offset is not None:
        env_kwargs["keepup_target_xy_offset"] = tuple(args.keepup_target_xy_offset)
    if args.post_contact_return_z_offset is not None:
        env_kwargs["post_contact_return_z_offset"] = args.post_contact_return_z_offset
    if args.contact_frame_velocity_target_gain is not None:
        env_kwargs["contact_frame_velocity_target_gain"] = args.contact_frame_velocity_target_gain
    if args.contact_frame_velocity_target_max is not None:
        env_kwargs["contact_frame_velocity_target_max"] = args.contact_frame_velocity_target_max
    if args.contact_frame_planner_enabled:
        env_kwargs["contact_frame_planner_enabled"] = True
    if not args.contact_frame_planner_hold_during_descent:
        env_kwargs["contact_frame_planner_hold_during_descent"] = False
    if args.contact_frame_planner_min_intercept_time is not None:
        env_kwargs["contact_frame_planner_min_intercept_time"] = args.contact_frame_planner_min_intercept_time
    if args.contact_frame_planner_max_intercept_time is not None:
        env_kwargs["contact_frame_planner_max_intercept_time"] = args.contact_frame_planner_max_intercept_time
    if args.contact_frame_planner_target_apex_z_offset is not None:
        env_kwargs["contact_frame_planner_target_apex_z_offset"] = args.contact_frame_planner_target_apex_z_offset
    if args.controller_velocity_gain is not None:
        env_kwargs["controller_velocity_gain"] = args.controller_velocity_gain
    if args.controller_velocity_feedback_gain is not None:
        env_kwargs["controller_velocity_feedback_gain"] = args.controller_velocity_feedback_gain
    if args.controller_max_velocity_step is not None:
        env_kwargs["controller_max_velocity_step"] = args.controller_max_velocity_step
    if args.contact_frame_trajectory_tilt_gain is not None:
        env_kwargs["contact_frame_trajectory_tilt_gain"] = args.contact_frame_trajectory_tilt_gain
    if args.contact_frame_trajectory_tilt_limit is not None:
        env_kwargs["contact_frame_trajectory_tilt_limit"] = tuple(args.contact_frame_trajectory_tilt_limit)
    if args.contact_frame_trajectory_tilt_deadband is not None:
        env_kwargs["contact_frame_trajectory_tilt_deadband"] = args.contact_frame_trajectory_tilt_deadband
    if args.require_reachable_next_intercept_for_success:
        env_kwargs["require_reachable_next_intercept_for_success"] = True
    if args.min_easy_next_ball_score_for_success is not None:
        env_kwargs["min_easy_next_ball_score_for_success"] = args.min_easy_next_ball_score_for_success
    if args.terminate_on_nonuseful_contact:
        env_kwargs["terminate_on_nonuseful_contact"] = True
    if args.next_intercept_xy_error_penalty_weight is not None:
        env_kwargs["next_intercept_xy_error_penalty_weight"] = args.next_intercept_xy_error_penalty_weight
    if args.post_contact_lateral_velocity_penalty_weight is not None:
        env_kwargs["post_contact_lateral_velocity_penalty_weight"] = (
            args.post_contact_lateral_velocity_penalty_weight
        )
    if args.contact_xy_error_penalty_weight is not None:
        env_kwargs["contact_xy_error_penalty_weight"] = args.contact_xy_error_penalty_weight
    if args.nonuseful_contact_penalty_weight is not None:
        env_kwargs["nonuseful_contact_penalty_weight"] = args.nonuseful_contact_penalty_weight
    if args.mode == "heuristic" and configured_model_path is None:
        # 모델 없이 heuristic만 볼 때는 strike-control 관측/행동 구성을 명시적으로 켠다.
        # LINK: pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py:49
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

    # MuJoCo passive viewer loop: action을 고르고 env.step() 후 episode 종료 시 콘솔 요약을 찍는다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:57
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
