from __future__ import annotations

from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from pingpong_rl2.envs import PingPongKeepUpGymEnv

_UNLIMITED_EVAL_STEP_LIMIT = 3_600

def evaluate_model(
    model: PPO,
    env_kwargs: dict[str, object],
    episodes: int,
    seed: int,
    evaluation_step_limit: int | None = None,
) -> dict[str, object]:
    # 학습 직후 deterministic policy를 별도 env에서 굴려 summary JSON에 들어갈 성공률 지표를 만든다.
    # LINK: mujoco/pingpong_rl2/scripts/run_ppo_learning.py:238
    env = PingPongKeepUpGymEnv(**env_kwargs)
    if evaluation_step_limit is None:
        # env episode가 무제한이면 평가가 멈추도록 별도 safety cap을 둔다.
        effective_step_limit = _UNLIMITED_EVAL_STEP_LIMIT if env.base_env.max_episode_steps is None else None
    else:
        parsed_step_limit = int(evaluation_step_limit)
        effective_step_limit = None if parsed_step_limit <= 0 else parsed_step_limit
    returns: list[float] = []
    useful_bounces: list[int] = []
    stable_cycles: list[int] = []
    failure_counts: Counter[str] = Counter()
    for episode_index in range(episodes):
        # deterministic action만 사용해 학습 noise 없이 episode return과 useful bounce count를 측정한다.
        observation, _ = env.reset(seed=seed + episode_index)
        episode_return = 0.0
        info: dict[str, object] = {}
        step_count = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            step_count += 1
            if not terminated and not truncated and effective_step_limit is not None and step_count >= effective_step_limit:
                truncated = True
                info = dict(info)
                info["truncated"] = True
                info["evaluation_step_limit"] = effective_step_limit
            if terminated or truncated:
                break
        returns.append(episode_return)
        useful_bounces.append(int(info.get("successful_bounce_count", 0)))
        stable_cycles.append(int(info.get("stable_cycle_count", info.get("successful_bounce_count", 0))))
        failure_reason = info.get("failure_reason")
        if failure_reason is None:
            failure_reason = "time_limit" if bool(info.get("truncated", False)) else "none"
        failure_counts[str(failure_reason)] += 1
    env.close()
    returns_array = np.asarray(returns, dtype=float)
    bounce_array = np.asarray(useful_bounces, dtype=float)
    stable_cycle_array = np.asarray(stable_cycles, dtype=float)
    # threshold별 episode count는 후속 분석/발표 자료에서 성공률 컷을 바로 비교하기 위한 값이다.
    one_or_more_useful = int(np.count_nonzero(bounce_array >= 1.0)) if bounce_array.size else 0
    two_or_more_useful = int(np.count_nonzero(bounce_array >= 2.0)) if bounce_array.size else 0
    three_or_more_useful = int(np.count_nonzero(bounce_array >= 3.0)) if bounce_array.size else 0
    ten_or_more_useful = int(np.count_nonzero(bounce_array >= 10.0)) if bounce_array.size else 0
    twenty_or_more_useful = int(np.count_nonzero(bounce_array >= 20.0)) if bounce_array.size else 0
    thirty_or_more_useful = int(np.count_nonzero(bounce_array >= 30.0)) if bounce_array.size else 0
    one_or_more_stable_cycles = (
        int(np.count_nonzero(stable_cycle_array >= 1.0)) if stable_cycle_array.size else 0
    )
    two_or_more_stable_cycles = (
        int(np.count_nonzero(stable_cycle_array >= 2.0)) if stable_cycle_array.size else 0
    )
    three_or_more_stable_cycles = (
        int(np.count_nonzero(stable_cycle_array >= 3.0)) if stable_cycle_array.size else 0
    )
    ten_or_more_stable_cycles = (
        int(np.count_nonzero(stable_cycle_array >= 10.0)) if stable_cycle_array.size else 0
    )
    twenty_or_more_stable_cycles = (
        int(np.count_nonzero(stable_cycle_array >= 20.0)) if stable_cycle_array.size else 0
    )
    thirty_or_more_stable_cycles = (
        int(np.count_nonzero(stable_cycle_array >= 30.0)) if stable_cycle_array.size else 0
    )
    ball_out_of_bounds_count = int(failure_counts.get("ball_out_of_bounds", 0))
    floor_contact_count = int(failure_counts.get("floor_contact", 0))
    robot_body_contact_count = int(failure_counts.get("robot_body_contact", 0))
    ball_speed_limit_count = int(failure_counts.get("ball_speed_limit", 0))
    return {
        "episodes": episodes,
        "mean_return": float(returns_array.mean()) if returns_array.size else 0.0,
        "mean_useful_bounces": float(bounce_array.mean()) if bounce_array.size else 0.0,
        "max_useful_bounces": int(bounce_array.max()) if bounce_array.size else 0,
        "mean_stable_cycles": float(stable_cycle_array.mean()) if stable_cycle_array.size else 0.0,
        "max_stable_cycles": int(stable_cycle_array.max()) if stable_cycle_array.size else 0,
        "episodes_with_one_or_more_useful_bounces": one_or_more_useful,
        "one_or_more_useful_bounce_rate": (one_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_two_or_more_useful_bounces": two_or_more_useful,
        "two_or_more_useful_bounce_rate": (two_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_three_or_more_useful_bounces": three_or_more_useful,
        "three_or_more_useful_bounce_rate": (three_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_ten_or_more_useful_bounces": ten_or_more_useful,
        "ten_or_more_useful_bounce_rate": (ten_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_twenty_or_more_useful_bounces": twenty_or_more_useful,
        "twenty_or_more_useful_bounce_rate": (twenty_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_thirty_or_more_useful_bounces": thirty_or_more_useful,
        "thirty_or_more_useful_bounce_rate": (thirty_or_more_useful / episodes) if episodes > 0 else 0.0,
        "episodes_with_one_or_more_stable_cycles": one_or_more_stable_cycles,
        "one_or_more_stable_cycle_rate": (one_or_more_stable_cycles / episodes) if episodes > 0 else 0.0,
        "episodes_with_two_or_more_stable_cycles": two_or_more_stable_cycles,
        "two_or_more_stable_cycle_rate": (two_or_more_stable_cycles / episodes) if episodes > 0 else 0.0,
        "episodes_with_three_or_more_stable_cycles": three_or_more_stable_cycles,
        "three_or_more_stable_cycle_rate": (three_or_more_stable_cycles / episodes) if episodes > 0 else 0.0,
        "episodes_with_ten_or_more_stable_cycles": ten_or_more_stable_cycles,
        "ten_or_more_stable_cycle_rate": (ten_or_more_stable_cycles / episodes) if episodes > 0 else 0.0,
        "episodes_with_twenty_or_more_stable_cycles": twenty_or_more_stable_cycles,
        "twenty_or_more_stable_cycle_rate": (
            (twenty_or_more_stable_cycles / episodes) if episodes > 0 else 0.0
        ),
        "episodes_with_thirty_or_more_stable_cycles": thirty_or_more_stable_cycles,
        "thirty_or_more_stable_cycle_rate": (
            (thirty_or_more_stable_cycles / episodes) if episodes > 0 else 0.0
        ),
        "ball_out_of_bounds_rate": (ball_out_of_bounds_count / episodes) if episodes > 0 else 0.0,
        "floor_contact_rate": (floor_contact_count / episodes) if episodes > 0 else 0.0,
        "robot_body_contact_rate": (robot_body_contact_count / episodes) if episodes > 0 else 0.0,
        "ball_speed_limit_rate": (ball_speed_limit_count / episodes) if episodes > 0 else 0.0,
        "failure_counts": dict(failure_counts),
    }

