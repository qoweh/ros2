from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pingpong_rl2.defaults import (
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_PPO_BATCH_SIZE,
    DEFAULT_PPO_GAMMA,
    DEFAULT_PPO_LEARNING_RATE,
    DEFAULT_PPO_N_STEPS,
    DEFAULT_PPO_TOTAL_TIMESTEPS,
    DEFAULT_RESET_BALL_HEIGHT_RANGE,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
)
from pingpong_rl2.training.presets import _ENV_PRESETS, _PRESET_MANAGED_ARG_DEFAULTS
from pingpong_rl2.utils import resolve_input_path

_CONFIG_PATH_DESTS = {"output_dir", "resume_from", "scene_path"}


def normalize_config_key(raw_key: str) -> str:
    # JSON config와 --set에서 CLI 스타일 키를 argparse destination 이름으로 통일한다.
    return raw_key.lstrip("-").replace("-", "_")


def parse_config_scalar(raw_value: str) -> object:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def values_equal(left: object, right: object) -> bool:
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(values_equal(a, b) for a, b in zip(left, right))
    return left == right


def explicit_cli_destinations(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set[str]:
    # config-file 값보다 사용자가 명시한 CLI 값을 우선하기 위해 argv에 등장한 option dest를 모은다.
    option_destinations: dict[str, str] = {}
    for action in parser._actions:
        for option_string in action.option_strings:
            option_destinations[option_string] = action.dest

    destinations: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option_string = token.split("=", 1)[0]
        destination = option_destinations.get(option_string)
        if destination is not None:
            destinations.add(destination)
    return destinations


def load_training_config(path: Path) -> dict[str, object]:
    # 저장된 training_summary 전체 또는 args 객체만 담은 JSON을 모두 config 입력으로 허용한다.
    # LINK: mujoco/pingpong_rl2/scripts/run_ppo_learning.py:258
    resolved_path = resolve_input_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Training config file not found: {resolved_path}")
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a JSON object, got {type(data).__name__}.")
    args_data = data.get("args", data)
    if not isinstance(args_data, dict):
        raise ValueError("Training config 'args' field must be a JSON object when provided.")
    return dict(args_data)


def apply_training_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    explicit_destinations: set[str],
) -> None:
    # config-file 값은 CLI로 직접 준 옵션을 덮지 않고, preset 관리 키도 유효 키로 인정한다.
    if args.config_file is None:
        return

    config_values = load_training_config(Path(args.config_file))
    valid_destinations = {action.dest for action in parser._actions} | set(_PRESET_MANAGED_ARG_DEFAULTS)
    for raw_key, value in config_values.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"Training config keys must be strings, got {raw_key!r}.")
        destination = normalize_config_key(raw_key)
        if destination == "config_file":
            continue
        if destination not in valid_destinations:
            raise ValueError(f"Unknown training config key: {raw_key!r}.")
        if destination in explicit_destinations:
            continue
        if destination in _CONFIG_PATH_DESTS and value is not None:
            value = Path(str(value))
        setattr(args, destination, value)


def apply_config_overrides(args: argparse.Namespace, overrides: Sequence[str]) -> None:
    # --set KEY=VALUE는 config/preset 적용 뒤 마지막으로 들어오는 실험용 단일 override다.
    valid_destinations = set(vars(args)) | set(_PRESET_MANAGED_ARG_DEFAULTS)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set expects KEY=VALUE, got {override!r}.")
        raw_key, raw_value = override.split("=", 1)
        destination = normalize_config_key(raw_key)
        if destination in {"config_file", "config_overrides", "preset"}:
            raise ValueError(f"--set cannot override {raw_key!r}; pass it as a normal CLI/config value.")
        if destination not in valid_destinations:
            raise ValueError(f"Unknown --set key: {raw_key!r}.")
        value = parse_config_scalar(raw_value)
        if destination in _CONFIG_PATH_DESTS and value is not None:
            value = Path(str(value))
        setattr(args, destination, value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # 학습 스크립트가 쓰는 CLI 전체를 정의하고, 마지막에 config-file 값을 병합한다.
    # LINK: mujoco/pingpong_rl2/scripts/run_ppo_learning.py:64
    parser = argparse.ArgumentParser(description="Train the minimal pingpong_rl2 PPO baseline.")
    # config/preset/run 경로 옵션은 실제 학습 설정이 확정되기 전 가장 먼저 해석된다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/env_config.py:10
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON config file using argparse destination names. "
            "CLI values override config-file values, and presets are applied after the config is loaded."
        ),
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override any config/preset-managed key. VALUE is parsed as JSON when possible, "
            "so lists can be passed as --set reset_velocity_z_range='[-0.01,0.01]'."
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=tuple(_ENV_PRESETS.keys()),
        help="Optional experiment preset that applies a fixed env configuration before any manual overrides.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--run-version",
        type=str,
        default=None,
        help="Optional suffix appended as <run-name>_<run-version> so A/B runs stay in separate directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional model zip to continue from. Default behavior resumes <run-name>_model.zip when it exists.",
    )
    parser.add_argument(
        "--reset-model",
        action="store_true",
        help="Start a fresh model even when the target run directory already has a saved model.",
    )
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_PPO_TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_PPO_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PPO_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PPO_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_PPO_GAMMA)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Optional MuJoCo scene XML. Presets may use this for geometry A/B variants.",
    )
    # heuristic bootstrap은 PPO 업데이트 전에 hand-coded policy를 모방해 actor를 warm start한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/bootstrap.py:12
    parser.add_argument(
        "--bootstrap-heuristic-episodes",
        type=int,
        default=0,
        help="Optional number of heuristic rollout episodes to collect before PPO learning for actor warm-start.",
    )
    parser.add_argument(
        "--bootstrap-min-useful-bounces",
        type=int,
        default=1,
        help="Minimum useful bounce count required for a heuristic episode to be kept in the bootstrap dataset.",
    )
    parser.add_argument(
        "--bootstrap-max-samples",
        type=int,
        default=0,
        help="Optional hard cap on accepted bootstrap samples. Set 0 for no cap.",
    )
    parser.add_argument(
        "--bootstrap-epochs",
        type=int,
        default=0,
        help="Number of supervised actor pretraining epochs on the accepted heuristic dataset.",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=256,
        help="Batch size used for heuristic actor pretraining.",
    )
    parser.add_argument(
        "--bootstrap-learning-rate",
        type=float,
        default=1.0e-3,
        help="Learning rate used for heuristic actor pretraining.",
    )
    parser.add_argument(
        "--bootstrap-sample-mode",
        type=str,
        default="episode",
        choices=("episode", "post_success", "post_success_reachable"),
        help=(
            "Which heuristic samples to keep: full qualifying episodes, only post-success steps, "
            "or only post-success steps whose next ball is still reachable."
        ),
    )
    parser.add_argument(
        "--bootstrap-followup-epochs",
        type=int,
        default=0,
        help="Optional extra actor pretraining epochs on a follow-up-focused heuristic dataset after the base bootstrap pass.",
    )
    parser.add_argument(
        "--bootstrap-followup-sample-mode",
        type=str,
        default="post_success_reachable",
        choices=("post_success", "post_success_reachable"),
        help="Sample filter for the optional follow-up bootstrap pass.",
    )
    parser.add_argument(
        "--bootstrap-followup-min-useful-bounces",
        type=int,
        default=None,
        help="Optional useful-bounce threshold override for the follow-up bootstrap pass. Defaults to --bootstrap-min-useful-bounces.",
    )
    parser.add_argument(
        "--bootstrap-followup-learning-rate",
        type=float,
        default=None,
        help="Optional learning rate override for the follow-up bootstrap pass. Defaults to --bootstrap-learning-rate.",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default="position",
        choices=(
            "position",
            "position_strike",
            "position_tilt",
            "position_strike_tilt",
            "position_strike_tilt_lift",
            "position_contact_frame",
            "position_contact_frame_velocity_residual",
            "position_contact_frame_velocity_tilt_residual",
            "position_contact_frame_velocity_tilt_lateral_residual",
            "position_contact_frame_velocity_tilt_lateral_apex_residual",
            "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
        ),
    )
    # tilt profile은 tilt action mode의 limit/regularization 묶음을 편의상 한 번에 고른다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/env_config.py:33
    parser.add_argument(
        "--tilt-profile",
        type=str,
        default="auto",
        choices=("auto", "custom", "early", "mid", "late", "final"),
        help="Convenience preset for position_tilt limits and regularization. 'auto' resolves to 'early'.",
    )
    # reset 분포와 curriculum 옵션은 vector env 전체에 같은 난이도 스케줄을 적용한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/curriculum.py:136
    parser.add_argument("--ball-height", type=float, default=DEFAULT_BALL_HEIGHT)
    parser.add_argument("--reset-ball-height-range", type=float, default=DEFAULT_RESET_BALL_HEIGHT_RANGE)
    parser.add_argument(
        "--reset-ball-height-bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Optional absolute reset height range above the racket. Overrides --reset-ball-height-range when set.",
    )
    parser.add_argument(
        "--target-ball-height",
        type=float,
        default=None,
        help="Desired post-contact apex height above the racket. Defaults to --ball-height for backward compatibility.",
    )
    parser.add_argument(
        "--keepup-target-xy-offset",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Optional XY offset from the controller anchor for the repeat keep-up target.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--reset-xy-range", type=float, default=DEFAULT_RESET_XY_RANGE)
    parser.add_argument(
        "--reset-xy-sampling",
        type=str,
        choices=("square", "disk"),
        default="square",
        help="Shape used for random XY reset offsets. disk gives uniform 0-360 degree starts inside the radius.",
    )
    parser.add_argument(
        "--reset-xy-curriculum-enabled",
        action="store_true",
        help="Linearly widen reset_xy_range during PPO training while keeping final evaluation at --reset-xy-range.",
    )
    parser.add_argument("--reset-xy-curriculum-start", type=float, default=None)
    parser.add_argument("--reset-xy-curriculum-end", type=float, default=None)
    parser.add_argument(
        "--reset-xy-curriculum-fraction",
        type=float,
        default=1.0,
        help="Fraction of this run's timesteps used to reach reset_xy_curriculum_end.",
    )
    parser.add_argument(
        "--reset-xy-curriculum-update-interval",
        type=int,
        default=10_000,
        help="Training timesteps between reset_xy_range curriculum updates.",
    )
    parser.add_argument("--reset-velocity-xy-range", type=float, default=DEFAULT_RESET_VELOCITY_XY_RANGE)
    parser.add_argument("--reset-velocity-xy-curriculum-start", type=float, default=None)
    parser.add_argument("--reset-velocity-xy-curriculum-end", type=float, default=None)
    parser.add_argument(
        "--reset-velocity-z-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_RESET_VELOCITY_Z_RANGE,
    )
    parser.add_argument(
        "--reset-velocity-z-curriculum-start",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument(
        "--reset-velocity-z-curriculum-end",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument(
        "--reset-ball-angular-velocity-range",
        type=float,
        default=0.0,
        help="Uniform per-axis initial ball spin range in rad/s. 0 keeps the old no-spin reset.",
    )
    parser.add_argument("--reset-ball-angular-velocity-curriculum-start", type=float, default=None)
    parser.add_argument("--reset-ball-angular-velocity-curriculum-end", type=float, default=None)
    parser.add_argument(
        "--success-velocity-threshold",
        type=float,
        default=DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    )
    parser.add_argument("--lateral-action-limit", type=float, default=None)
    parser.add_argument("--vertical-action-limit", type=float, default=None)
    parser.add_argument("--tilt-action-limit", type=float, default=None)
    parser.add_argument("--followup-lift-action-limit", type=float, default=None)
    parser.add_argument(
        "--target-offset-low",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-0.12, -0.12, -0.04),
        help="Lower XYZ target clamp relative to the controller anchor.",
    )
    parser.add_argument(
        "--target-offset-high",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.12, 0.12, 0.12),
        help="Upper XYZ target clamp relative to the controller anchor.",
    )
    # tilt/strike 보정 옵션은 position 계열 action을 racket target과 face tilt로 확장한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/env_config.py:33
    parser.add_argument("--tracking-during-contact-scale", type=float, default=None)
    parser.add_argument("--useful-contact-outgoing-x-penalty-weight", type=float, default=None)
    parser.add_argument("--desired-outgoing-ball-velocity-x", type=float, default=None)
    parser.add_argument("--useful-contact-return-target-xy-reward-weight", type=float, default=None)
    parser.add_argument(
        "--return-target-xy-source",
        type=str,
        choices=("controller_anchor", "racket_home", "racket_position", "target_position"),
        default=None,
    )
    parser.add_argument("--return-target-xy-tolerance", type=float, default=None)
    parser.add_argument("--tilt-angle-penalty-weight", type=float, default=None)
    parser.add_argument("--tilt-action-delta-penalty-weight", type=float, default=None)
    parser.add_argument(
        "--target-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument(
        "--target-pitch-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Optional target pitch clamp applied after tilt integration. Use this for inward-only rebound A/B runs.",
    )
    parser.add_argument(
        "--initial-target-tilt",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
        help="Optional initial target tilt applied at env reset. Useful for breaking the zero-tilt symmetry in tilt A/B runs.",
    )
    parser.add_argument(
        "--strike-tilt-assist-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
        help="Optional pre-contact tilt assist limit for position_strike. The assist ramps toward center-seeking tilt and returns to neutral after contact.",
    )
    parser.add_argument(
        "--strike-tilt-assist-deadband",
        type=float,
        default=None,
        help="Deadband in meters below which position_strike tilt assist stays neutral.",
    )
    parser.add_argument(
        "--strike-tilt-ramp-pitch",
        type=float,
        default=None,
        help="Optional fixed pitch target for position_strike that ramps in only during pre-contact strike preparation and returns to neutral after contact.",
    )
    parser.add_argument(
        "--strike-tilt-ramp-xy-tolerance",
        type=float,
        default=None,
        help="Maximum XY alignment error allowed before the position_strike pitch ramp stays neutral.",
    )
    parser.add_argument(
        "--followup-strike-target-tilt",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
        help="Optional persistent target tilt used after the first useful bounce so follow-up strikes keep a nonzero inward face.",
    )
    parser.add_argument(
        "--followup-strike-contact-offset-ratio",
        type=float,
        default=None,
        help="Optional fraction of anchor correction used to bias follow-up descending strike contact points toward the strike zone center.",
    )
    parser.add_argument(
        "--followup-strike-contact-offset-max",
        type=float,
        default=None,
        help="Maximum meters of follow-up descending strike contact bias toward the strike zone center.",
    )
    parser.add_argument(
        "--followup-strike-lift-boost",
        type=float,
        default=None,
        help="Optional extra follow-up lift boost applied only after the first useful bounce.",
    )
    # contact-frame 옵션은 policy action을 접촉 좌표계 residual과 속도 residual로 해석하게 만든다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py:2144
    parser.add_argument("--contact-frame-base-strike-z-boost", type=float, default=None)
    parser.add_argument("--contact-frame-base-strike-z-offset", type=float, default=None)
    parser.add_argument("--contact-frame-base-strike-time-horizon", type=float, default=None)
    parser.add_argument(
        "--contact-frame-base-tilt-residual",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-apex-lift-gain", type=float, default=None)
    parser.add_argument("--contact-frame-apex-lift-max", type=float, default=None)
    parser.add_argument("--contact-frame-apex-lift-reference-velocity-z", type=float, default=None)
    parser.add_argument("--contact-frame-apex-lift-restitution", type=float, default=None)
    parser.add_argument("--contact-frame-low-apex-recovery-lift-gain", type=float, default=None)
    parser.add_argument("--contact-frame-low-apex-recovery-lift-max", type=float, default=None)
    parser.add_argument("--contact-frame-low-apex-recovery-velocity-gain", type=float, default=None)
    parser.add_argument("--contact-frame-low-apex-recovery-velocity-max", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-lead-gain", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-lead-max", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-target-gain", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-target-max", type=float, default=None)
    parser.add_argument("--contact-frame-velocity-scale-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-outgoing-xy-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-racket-vz-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-racket-xy-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-tilt-scale-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-target-apex-z-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-strike-plane-z-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-tracking-xy-action-limit", type=float, default=None)
    parser.add_argument("--contact-frame-intercept-velocity-gain", type=float, default=None)
    parser.add_argument("--contact-frame-intercept-velocity-max", type=float, default=None)
    parser.add_argument("--contact-frame-intercept-velocity-time-floor", type=float, default=None)
    parser.add_argument(
        "--contact-frame-planner-enabled",
        action="store_true",
        help="Use the self-rally contact-frame planner: fixed next target near the anchor, primitive base strike, RL residual only.",
    )
    parser.add_argument(
        "--disable-contact-frame-planner-hold-during-descent",
        action="store_false",
        dest="contact_frame_planner_hold_during_descent",
        default=True,
        help="Recompute planner target XY/apex every step instead of holding one target for the current descent.",
    )
    parser.add_argument("--contact-frame-planner-min-intercept-time", type=float, default=None)
    parser.add_argument("--contact-frame-planner-max-intercept-time", type=float, default=None)
    parser.add_argument("--contact-frame-planner-target-apex-z-offset", type=float, default=None)
    parser.add_argument("--contact-frame-planner-contact-offset-ratio", type=float, default=None)
    parser.add_argument("--contact-frame-planner-contact-offset-max", type=float, default=None)
    parser.add_argument("--contact-frame-strike-hold-time", type=float, default=None)
    parser.add_argument("--contact-frame-strike-hold-min-readiness", type=float, default=None)
    parser.add_argument("--contact-frame-followthrough-gain", type=float, default=None)
    parser.add_argument("--contact-frame-followthrough-time", type=float, default=None)
    parser.add_argument("--contact-frame-followthrough-max", type=float, default=None)
    parser.add_argument("--contact-frame-lateral-brake-gain", type=float, default=None)
    parser.add_argument("--contact-frame-lateral-brake-max", type=float, default=None)
    parser.add_argument("--contact-frame-lateral-brake-radius", type=float, default=None)
    parser.add_argument("--contact-frame-trajectory-tilt-gain", type=float, default=None)
    parser.add_argument(
        "--contact-frame-trajectory-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-trajectory-tilt-deadband", type=float, default=None)
    parser.add_argument("--contact-frame-tilt-ramp-time", type=float, default=None)
    # controller 옵션은 목표 end-effector pose를 MuJoCo joint target으로 바꾸는 내부 제어기를 조율한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/controllers/ee_pose_controller.py:1
    parser.add_argument("--controller-orientation-gain", type=float, default=None)
    parser.add_argument("--controller-max-orientation-step", type=float, default=None)
    parser.add_argument("--controller-velocity-gain", type=float, default=None)
    parser.add_argument("--controller-velocity-feedback-gain", type=float, default=None)
    parser.add_argument("--controller-max-velocity-step", type=float, default=None)
    parser.add_argument("--controller-nullspace-posture-gain", type=float, default=None)
    parser.add_argument("--controller-nullspace-posture-max-step", type=float, default=None)
    parser.add_argument(
        "--controller-nullspace-posture-target",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        default=None,
    )
    parser.add_argument("--controller-body-clearance-gain", type=float, default=None)
    parser.add_argument("--controller-body-clearance-margin", type=float, default=None)
    parser.add_argument("--controller-body-clearance-vertical-margin", type=float, default=None)
    parser.add_argument("--controller-body-clearance-max-step", type=float, default=None)
    parser.add_argument(
        "--controller-body-clearance-body-names",
        type=str,
        nargs="+",
        default=None,
        help="Robot body names that the nullspace clearance controller should keep away from the ball.",
    )
    parser.add_argument(
        "--tracking-strike-plane-offset",
        type=float,
        default=None,
        help="Vertical strike-plane offset above the controller anchor used for contact/intercept timing.",
    )
    parser.add_argument(
        "--contact-frame-centering-tilt-limit",
        type=float,
        nargs=2,
        metavar=("PITCH", "ROLL"),
        default=None,
    )
    parser.add_argument("--contact-frame-centering-tilt-radius", type=float, default=None)
    parser.add_argument("--contact-frame-centering-tilt-deadband", type=float, default=None)
    parser.add_argument("--contact-frame-action-penalty-weight", type=float, default=None)
    # 접촉 품질/반동 reward 옵션은 분석 CSV의 contact 지표와 같은 이름 계열을 공유한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/analysis/rebound_metrics.py:54
    parser.add_argument("--next-intercept-xy-error-penalty-weight", type=float, default=None)
    parser.add_argument("--post-contact-lateral-velocity-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-xy-error-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-racket-lateral-velocity-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-racket-lateral-velocity-tolerance", type=float, default=None)
    parser.add_argument("--contact-racket-outward-velocity-penalty-weight", type=float, default=None)
    parser.add_argument("--contact-racket-outward-velocity-tolerance", type=float, default=None)
    parser.add_argument("--max-contact-racket-lateral-speed-for-success", type=float, default=None)
    parser.add_argument("--nonuseful-contact-penalty-weight", type=float, default=None)
    parser.add_argument(
        "--contact-apex-under-target-penalty-weight",
        type=float,
        default=None,
        help="Penalty scale for upward contacts whose projected apex stays below the target keep-up height.",
    )
    parser.add_argument(
        "--contact-apex-progress-reward-weight",
        type=float,
        default=None,
        help="Dense reward scale for upward contacts whose projected apex moves toward target_ball_height.",
    )
    parser.add_argument(
        "--contact-apex-recovery-progress-reward-weight",
        type=float,
        default=None,
        help="Dense reward scale for upward contacts that improve the projected apex after a previous low contact.",
    )
    parser.add_argument(
        "--gate-contact-apex-progress-by-easy-next-ball",
        action="store_true",
        help="Scale dense apex progress by easy_next_ball_score so height shaping only helps recoverable keep-up balls.",
    )
    parser.add_argument(
        "--contact-apex-progress-min-easy-next-ball-score",
        type=float,
        default=None,
        help="Optional easy_next_ball_score floor below which apex progress shaping is zeroed.",
    )
    parser.add_argument(
        "--contact-apex-potential-reward-weight",
        type=float,
        default=None,
        help="Potential-style shaping scale for moving projected contact apex closer to target height.",
    )
    parser.add_argument(
        "--contact-apex-potential-gamma",
        type=float,
        default=None,
        help="Discount factor used by --contact-apex-potential-reward-weight.",
    )
    parser.add_argument(
        "--contact-apex-potential-cap",
        type=float,
        default=None,
        help="Maximum normalized shortfall used by --contact-apex-potential-reward-weight.",
    )
    parser.add_argument(
        "--contact-lateral-stability-reward-weight",
        type=float,
        default=None,
        help="Reward scale for upward contacts with low lateral outgoing speed and centered projected apex XY.",
    )
    parser.add_argument(
        "--contact-lateral-stability-speed-tolerance",
        type=float,
        default=None,
        help="Outgoing lateral-speed tolerance used by --contact-lateral-stability-reward-weight.",
    )
    parser.add_argument(
        "--contact-lateral-stability-xy-tolerance",
        type=float,
        default=None,
        help="Projected apex XY tolerance used by --contact-lateral-stability-reward-weight.",
    )
    parser.add_argument(
        "--contact-lateral-stability-min-apex-ratio",
        type=float,
        default=None,
        help="Minimum projected apex/target apex ratio before lateral stability reward can be paid.",
    )
    parser.add_argument(
        "--stable-contact-reward-weight",
        type=float,
        default=None,
        help="Reward scale for contacts that combine target apex height with an easy next descending intercept.",
    )
    parser.add_argument(
        "--stable-contact-min-apex-ratio",
        type=float,
        default=None,
        help="Minimum projected apex/target apex ratio before stable-contact reward can be paid.",
    )
    parser.add_argument(
        "--stable-cycle-reward-weight",
        type=float,
        default=None,
        help="Additional reward scale for consecutive useful contacts that keep target apex and an easy next intercept.",
    )
    parser.add_argument(
        "--stable-cycle-reward-cap",
        type=int,
        default=4,
        help="Maximum consecutive stable-cycle count used to scale --stable-cycle-reward-weight.",
    )
    parser.add_argument(
        "--stable-cycle-min-easy-next-ball-score",
        type=float,
        default=None,
        help="Minimum easy_next_ball_score required before a useful contact counts as a stable cycle.",
    )
    parser.add_argument("--post-contact-return-assist-weight", type=float, default=None)
    parser.add_argument("--post-contact-return-max-intercept-time", type=float, default=None)
    parser.add_argument(
        "--post-contact-return-z-offset",
        type=float,
        default=None,
        help="Vertical offset applied to the racket target while the ball is rising after contact.",
    )
    parser.add_argument(
        "--disable-post-contact-return-predict-during-rise",
        action="store_false",
        dest="post_contact_return_predict_during_rise",
        default=True,
        help="Return to the anchor while the ball rises instead of chasing a predicted future intercept.",
    )
    parser.add_argument("--next-intercept-reachable-bonus-weight", type=float, default=None)
    parser.add_argument("--easy-next-ball-reward-weight", type=float, default=None)
    parser.add_argument(
        "--next-intercept-success-radius",
        type=float,
        default=None,
        help="XY radius used to decide whether the next descending intercept is easy enough to count as useful.",
    )
    parser.add_argument(
        "--easy-next-ball-xy-radius",
        type=float,
        default=None,
        help="XY radius used inside the dense easy-next-ball score.",
    )
    parser.add_argument(
        "--require-reachable-next-intercept-for-success",
        action="store_true",
        help="Only count a contact as useful when the next descending intercept remains inside the strike zone.",
    )
    parser.add_argument(
        "--require-apex-height-window-for-success",
        action="store_true",
        help="Only count a contact as useful when the projected post-contact apex stays within height_tolerance of target_ball_height.",
    )
    parser.add_argument(
        "--min-easy-next-ball-score-for-success",
        type=float,
        default=None,
        help="Optional lower bound on easy_next_ball_score for useful-contact success.",
    )
    parser.add_argument(
        "--gate-nonuseful-easy-next-ball-by-apex",
        action="store_true",
        help="Scale non-success easy-next-ball shaping by projected apex height match.",
    )
    parser.add_argument(
        "--terminate-on-nonuseful-contact",
        action="store_true",
        help="End the episode immediately when a racket contact does not satisfy useful-contact success.",
    )
    parser.add_argument(
        "--terminate-on-low-apex-contact",
        action="store_true",
        help="End the episode when an upward racket contact projects below the low-apex threshold.",
    )
    parser.add_argument(
        "--low-apex-contact-height-threshold",
        type=float,
        default=None,
        help="Minimum projected post-contact apex height above the racket before a low-apex contact terminates.",
    )
    parser.add_argument(
        "--low-apex-contact-grace-count",
        type=int,
        default=0,
        help="Number of consecutive low-apex upward contacts allowed before low-apex termination.",
    )
    parser.add_argument(
        "--trajectory-match-reward-weight",
        type=float,
        default=None,
        help="Optional contact-event reward bonus for matching the desired outgoing ball velocity.",
    )
    parser.add_argument(
        "--trajectory-error-penalty-weight",
        type=float,
        default=None,
        help="Optional contact-event penalty for outgoing ball velocity error.",
    )
    parser.add_argument(
        "--reward-contact-quality-on-any-upward-contact",
        action="store_true",
        help="Apply apex/easy-next-ball shaping to upward contacts even before they satisfy strict success.",
    )
    parser.add_argument("--next-intercept-max-time", type=float, default=None)
    # observation 확장 옵션은 policy 입력 벡터에 phase, contact, next-intercept 정보를 추가한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/envs/observation_layout.py:1
    parser.add_argument(
        "--include-velocity-domain-observation",
        action="store_true",
        help="Add relative velocity and racket face normal to the observation for velocity-domain rebound experiments.",
    )
    parser.add_argument(
        "--include-task-phase-observation",
        action="store_true",
        help="Add prepare/strike/return/recovery phase one-hot observation for repeated keep-up experiments.",
    )
    parser.add_argument(
        "--include-contact-context-observation",
        action="store_true",
        help="Add time-since-contact and clipped bounce-count observation signals.",
    )
    parser.add_argument(
        "--include-next-intercept-observation",
        action="store_true",
        help="Add next descending intercept and recovery-readiness observation signals.",
    )
    parser.add_argument(
        "--include-desired-outgoing-velocity-observation",
        action="store_true",
        help="Add the desired outgoing ball velocity target to the observation for trajectory-matching experiments.",
    )
    parser.add_argument(
        "--desired-outgoing-xy-mode",
        type=str,
        choices=("next_intercept", "apex"),
        default="next_intercept",
        help="Whether desired outgoing XY velocity aims at the next descending intercept or at the apex.",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    # evaluation과 policy 초기화 옵션은 학습 종료 후 검증 및 residual action 탐색 폭을 제어한다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/policy_init.py:63
    parser.add_argument(
        "--evaluation-step-limit",
        type=int,
        default=None,
        help=(
            "Safety cap used only by deterministic evaluations when the env itself has unlimited episode length. "
            "Defaults to 3600 steps for unlimited envs."
        ),
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=None,
        help="Optional initial log std for PPO Gaussian actions. Useful for small residual action spaces.",
    )
    parser.add_argument(
        "--scale-log-std-by-action-limit",
        action="store_true",
        help="Initialize PPO log std per action dimension from the environment action limits.",
    )
    parser.add_argument(
        "--action-std-limit-ratio",
        type=float,
        default=None,
        help="Std/action-limit ratio used with --scale-log-std-by-action-limit.",
    )
    parser.add_argument(
        "--action-std-min",
        type=float,
        default=None,
        help="Minimum per-dimension action std used with --scale-log-std-by-action-limit.",
    )
    parser.add_argument(
        "--action-std-max",
        type=float,
        default=None,
        help="Maximum per-dimension action std used with --scale-log-std-by-action-limit.",
    )
    parser.add_argument(
        "--zero-init-action-mean",
        action="store_true",
        help="Initialize the PPO action mean head to zero for residual action spaces.",
    )
    parser.add_argument("--smoke", action="store_true")
    # CLI 명시값을 먼저 기억한 뒤, config-file의 나머지 값만 args에 병합한다.
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    apply_training_config(args, parser, explicit_cli_destinations(parser, raw_argv))
    return args
