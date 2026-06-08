from __future__ import annotations

from typing import Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch as th

def learn_model(
    *,
    model: PPO,
    total_timesteps: int,
    initial_reset_num_timesteps: bool,
    callback: BaseCallback | None = None,
) -> int:
    # smoke/eval-only 실행도 같은 경로를 타도록 total_timesteps=0이면 learn을 건너뛴다.
    # LINK: mujoco/pingpong_rl2/scripts/run_ppo_learning.py:238
    if total_timesteps < 0:
        raise ValueError(f"total_timesteps must be non-negative, got {total_timesteps}.")
    if total_timesteps > 0:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=False,
            reset_num_timesteps=initial_reset_num_timesteps,
            callback=callback,
        )
    return total_timesteps


def scaled_action_log_std(
    *,
    action_high: Sequence[float],
    ratio: float,
    min_std: float | None,
    max_std: float | None,
) -> np.ndarray:
    # action space limit의 일정 비율을 Gaussian std로 삼아 residual action 탐색 폭을 맞춘다.
    # LINK: mujoco/pingpong_rl2/src/pingpong_rl2/training/env_config.py:73
    if ratio <= 0.0:
        raise ValueError(f"action-std-limit-ratio must be positive, got {ratio}.")
    if min_std is not None and min_std <= 0.0:
        raise ValueError(f"action-std-min must be positive when provided, got {min_std}.")
    if max_std is not None and max_std <= 0.0:
        raise ValueError(f"action-std-max must be positive when provided, got {max_std}.")
    if min_std is not None and max_std is not None and min_std > max_std:
        raise ValueError(f"action-std-min must be <= action-std-max, got {min_std} > {max_std}.")

    high = np.asarray(action_high, dtype=float)
    if high.ndim != 1 or high.size == 0:
        raise ValueError(f"action_high must be a non-empty 1D vector, got shape {high.shape}.")
    if not np.all(np.isfinite(high)) or np.any(high <= 0.0):
        raise ValueError(f"action_high must contain positive finite limits, got {high}.")

    std = high * ratio
    if min_std is not None:
        std = np.maximum(std, float(min_std))
    if max_std is not None:
        std = np.minimum(std, float(max_std))
    return np.log(std)


def initialize_scaled_policy_log_std(
    *,
    model: PPO,
    ratio: float | None,
    min_std: float | None,
    max_std: float | None,
) -> dict[str, object]:
    # SB3 policy의 log_std 파라미터를 action dimension별 limit 기반 값으로 직접 초기화한다.
    # LINK: mujoco/pingpong_rl2/scripts/run_ppo_learning.py:141
    resolved_ratio = 0.35 if ratio is None else float(ratio)
    log_std = scaled_action_log_std(
        action_high=model.action_space.high,
        ratio=resolved_ratio,
        min_std=min_std,
        max_std=max_std,
    )
    if not hasattr(model.policy, "log_std"):
        raise ValueError("Selected PPO policy does not expose log_std for scaled initialization.")
    if tuple(model.policy.log_std.shape) != tuple(log_std.shape):
        raise ValueError(
            "Policy log_std shape does not match action space: "
            f"{tuple(model.policy.log_std.shape)} vs {tuple(log_std.shape)}."
        )
    with th.no_grad():
        model.policy.log_std.copy_(th.as_tensor(log_std, dtype=model.policy.log_std.dtype, device=model.device))
    return {
        "ratio": resolved_ratio,
        "min_std": min_std,
        "max_std": max_std,
        "log_std": log_std.tolist(),
        "std": np.exp(log_std).tolist(),
    }
