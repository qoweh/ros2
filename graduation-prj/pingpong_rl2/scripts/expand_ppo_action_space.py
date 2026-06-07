from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from stable_baselines3 import PPO
import torch as th

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
SCRIPTS_ROOT = ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pingpong_rl2.envs import PingPongKeepUpGymEnv
from pingpong_rl2.utils import resolve_input_path, resolve_output_path

import run_ppo_learning


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a larger-action PPO model by copying a smaller residual policy and "
            "zero-initializing newly added action dimensions."
        )
    )
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--target-config-file",
        type=Path,
        default=None,
        help="Training config whose preset/env defines the target observation/action spaces.",
    )
    parser.add_argument(
        "--target-preset",
        type=str,
        default=None,
        choices=tuple(run_ppo_learning._ENV_PRESETS.keys()),
        help="Target preset used when --target-config-file is not provided.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--new-action-std-ratio",
        type=float,
        default=0.10,
        help="Std/action-limit ratio for action dimensions that do not exist in the source policy.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output-model stem>_summary.json.",
    )
    return parser.parse_args(argv)


def target_env_kwargs(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    # target preset/config는 run_ppo_learning의 파서를 그대로 써서 향후 학습 공간과 맞춘다.
    # LINK: pingpong_rl2/scripts/run_ppo_learning.py:60
    # LINK: pingpong_rl2/src/pingpong_rl2/training/env_config.py:88
    learning_argv: list[str] = []
    if args.target_config_file is not None:
        learning_argv.extend(["--config-file", str(args.target_config_file)])
    elif args.target_preset is not None:
        learning_argv.extend(["--preset", args.target_preset])
    else:
        raise ValueError("Pass either --target-config-file or --target-preset.")

    learning_argv.extend(["--total-timesteps", "0", "--reset-model"])
    learning_args = run_ppo_learning.parse_args(learning_argv)
    resolved_preset = run_ppo_learning.apply_env_preset(learning_args)
    run_ppo_learning.apply_config_overrides(learning_args, learning_args.config_overrides)
    run_ppo_learning.resolve_tilt_profile(learning_args)
    return run_ppo_learning.env_kwargs_from_args(learning_args), resolved_preset


def copy_policy_prefix(
    *,
    source_model: PPO,
    target_model: PPO,
    new_action_std_ratio: float,
) -> dict[str, object]:
    # observation space는 그대로 두고 action head/log_std의 앞부분만 이전 policy에서 복사한다.
    if new_action_std_ratio <= 0.0:
        raise ValueError(f"new-action-std-ratio must be positive, got {new_action_std_ratio}.")
    if source_model.observation_space.shape != target_model.observation_space.shape:
        raise ValueError(
            "Source and target observation spaces must match for direct transfer: "
            f"{source_model.observation_space} vs {target_model.observation_space}."
        )

    source_action_dim = int(np.prod(source_model.action_space.shape))
    target_action_dim = int(np.prod(target_model.action_space.shape))
    if source_action_dim >= target_action_dim:
        raise ValueError(
            "This helper only expands action spaces; got "
            f"source={source_action_dim}, target={target_action_dim}."
        )

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    copied_state: dict[str, th.Tensor] = {}
    for key, target_tensor in target_state.items():
        if key not in source_state:
            copied_state[key] = target_tensor
            continue

        source_tensor = source_state[key]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            copied_state[key] = source_tensor.detach().clone()
            continue

        if key == "log_std":
            expanded = target_tensor.detach().clone()
            expanded[:source_action_dim] = source_tensor.detach().clone()
            action_high = np.asarray(target_model.action_space.high, dtype=float)
            new_std = np.maximum(action_high[source_action_dim:] * new_action_std_ratio, 1.0e-4)
            expanded[source_action_dim:] = th.as_tensor(
                np.log(new_std),
                dtype=expanded.dtype,
                device=expanded.device,
            )
            copied_state[key] = expanded
            continue

        if key == "action_net.weight":
            expanded = target_tensor.detach().clone()
            expanded.zero_()
            expanded[:source_action_dim, :] = source_tensor.detach().clone()
            copied_state[key] = expanded
            continue

        if key == "action_net.bias":
            expanded = target_tensor.detach().clone()
            expanded.zero_()
            expanded[:source_action_dim] = source_tensor.detach().clone()
            copied_state[key] = expanded
            continue

        raise ValueError(
            f"Cannot copy parameter {key!r}: source shape {tuple(source_tensor.shape)} "
            f"does not match target shape {tuple(target_tensor.shape)}."
        )

    target_model.policy.load_state_dict(copied_state)
    return {
        "source_action_dim": source_action_dim,
        "target_action_dim": target_action_dim,
        "copied_action_dims": source_action_dim,
        "new_action_dims": target_action_dim - source_action_dim,
        "new_action_std_ratio": float(new_action_std_ratio),
        "target_action_high": np.asarray(target_model.action_space.high, dtype=float).tolist(),
        "target_log_std": target_model.policy.log_std.detach().cpu().numpy().tolist(),
    }


def main() -> None:
    # source PPO를 로드하고 target env/action space에 맞는 새 PPO 껍데기를 만든다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:15
    args = parse_args()
    source_path = resolve_input_path(args.source_model)
    output_path = resolve_output_path(args.output_model)
    if output_path.suffix != ".zip":
        output_path = output_path.with_suffix(".zip")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs, resolved_preset = target_env_kwargs(args)
    env = PingPongKeepUpGymEnv(**env_kwargs)
    try:
        source_model = PPO.load(str(source_path), device=args.device)
        target_model = PPO(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=512,
            learning_rate=1.0e-5,
            n_epochs=1,
            clip_range=0.05,
            verbose=0,
            seed=7,
            device=args.device,
        )
        transfer_summary = copy_policy_prefix(
            source_model=source_model,
            target_model=target_model,
            new_action_std_ratio=args.new_action_std_ratio,
        )
        target_model.save(str(output_path))
        summary = {
            "source_model": str(source_path.resolve()),
            "output_model": str(output_path.resolve()),
            "resolved_preset": resolved_preset,
            "env_kwargs": env_kwargs,
            "env_config": env.training_config(),
            "target_env_config": env.training_config(),
            "transfer": transfer_summary,
        }
    finally:
        env.close()

    summary_path = (
        resolve_output_path(args.summary_path)
        if args.summary_path is not None
        else output_path.with_name(f"{output_path.stem.removesuffix('_model')}_training_summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"source_model={source_path}")
    print(f"output_model={output_path}")
    print(f"summary_path={summary_path}")
    print(
        "transfer "
        f"source_action_dim={transfer_summary['source_action_dim']} "
        f"target_action_dim={transfer_summary['target_action_dim']}"
    )


if __name__ == "__main__":
    main()
