from __future__ import annotations

from math import ceil

import numpy as np
from stable_baselines3 import PPO
import torch as th

from pingpong_rl2.controllers import HeuristicKeepUpPolicy
from pingpong_rl2.envs import PingPongKeepUpGymEnv

def collect_heuristic_bootstrap_dataset(
    *,
    env_kwargs: dict[str, object],
    episodes: int,
    seed: int,
    min_useful_bounces: int,
    max_samples: int,
    sample_mode: str,
) -> dict[str, object]:
    # hand-coded policy rollout에서 충분히 성공한 episode만 모아 PPO actor 사전학습 샘플을 만든다.
    # LINK: pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py:49
    if episodes <= 0:
        return {
            "requested_episodes": episodes,
            "accepted_episodes": 0,
            "accepted_samples": 0,
            "mean_episode_useful_bounces": 0.0,
            "sample_mode": sample_mode,
            "observations": np.empty((0, 0), dtype=np.float32),
            "actions": np.empty((0, 0), dtype=np.float32),
        }
    if env_kwargs.get("action_mode") not in {
        "position_strike",
        "position_strike_tilt",
        "position_strike_tilt_lift",
        "position_contact_frame",
        "position_contact_frame_velocity_residual",
        "position_contact_frame_velocity_tilt_residual",
        "position_contact_frame_velocity_tilt_lateral_residual",
        "position_contact_frame_velocity_tilt_lateral_apex_residual",
        "position_contact_frame_velocity_tilt_lateral_apex_tracking_residual",
    }:
        raise ValueError(
            "Heuristic bootstrap currently requires action_mode='position_strike', 'position_strike_tilt', "
            "'position_strike_tilt_lift', 'position_contact_frame', or "
            "'position_contact_frame_velocity_residual', or "
            "'position_contact_frame_velocity_tilt_residual', or "
            "'position_contact_frame_velocity_tilt_lateral_residual', or "
            "'position_contact_frame_velocity_tilt_lateral_apex_residual', or "
            "'position_contact_frame_velocity_tilt_lateral_apex_tracking_residual'."
        )
    if sample_mode not in {"episode", "post_success", "post_success_reachable"}:
        raise ValueError(f"Unsupported bootstrap sample mode: {sample_mode}")

    env = PingPongKeepUpGymEnv(**env_kwargs)
    policy = HeuristicKeepUpPolicy()
    # episode별 sample을 임시로 쌓은 뒤 성공 기준과 sample_mode를 통과한 것만 전체 dataset에 넣는다.
    accepted_observations: list[np.ndarray] = []
    accepted_actions: list[np.ndarray] = []
    accepted_episode_useful_bounces: list[int] = []
    accepted_episode_count = 0
    qualifying_episode_count = 0
    try:
        for episode_index in range(episodes):
            observation, _ = env.reset(seed=seed + episode_index)
            policy.reset()
            episode_samples: list[dict[str, object]] = []
            info: dict[str, object] = {}
            while True:
                action = policy.predict(env.base_env).astype(np.float32, copy=False)
                next_observation, _, terminated, truncated, info = env.step(action)
                episode_samples.append(
                    {
                        "observation": np.asarray(observation, dtype=np.float32).copy(),
                        "action": np.asarray(action, dtype=np.float32).copy(),
                        "successful_bounce_count": int(info.get("successful_bounce_count", 0)),
                        "next_intercept_reachable": bool(info.get("next_intercept_reachable", False)),
                    }
                )
                observation = next_observation
                if terminated or truncated:
                    break

            useful_bounces = int(info.get("successful_bounce_count", 0))
            if useful_bounces < min_useful_bounces:
                continue
            qualifying_episode_count += 1

            # follow-up bootstrap은 성공 이후 상태만 남겨 반복 keep-up 구간의 actor bias를 키운다.
            if sample_mode == "episode":
                selected_samples = episode_samples
            elif sample_mode == "post_success":
                selected_samples = [
                    sample for sample in episode_samples if int(sample["successful_bounce_count"]) > 0
                ]
            else:
                selected_samples = [
                    sample
                    for sample in episode_samples
                    if int(sample["successful_bounce_count"]) > 0 and bool(sample["next_intercept_reachable"])
                ]
            if not selected_samples:
                continue

            accepted_episode_count += 1
            accepted_episode_useful_bounces.append(useful_bounces)
            accepted_observations.extend(
                np.asarray(sample["observation"], dtype=np.float32) for sample in selected_samples
            )
            accepted_actions.extend(
                np.asarray(sample["action"], dtype=np.float32) for sample in selected_samples
            )
            if max_samples > 0 and len(accepted_observations) >= max_samples:
                accepted_observations = accepted_observations[:max_samples]
                accepted_actions = accepted_actions[:max_samples]
                break
    finally:
        env.close()

    if not accepted_observations:
        # 수집 실패도 호출부가 같은 summary schema를 쓸 수 있도록 빈 배열과 count를 반환한다.
        observation_shape = (0, env.base_env.observation_size)
        action_shape = (0, env.action_space.shape[0])
        return {
            "requested_episodes": episodes,
            "accepted_episodes": accepted_episode_count,
            "qualifying_episodes": qualifying_episode_count,
            "accepted_samples": 0,
            "mean_episode_useful_bounces": 0.0,
            "sample_mode": sample_mode,
            "observations": np.empty(observation_shape, dtype=np.float32),
            "actions": np.empty(action_shape, dtype=np.float32),
        }

    observations_array = np.asarray(accepted_observations, dtype=np.float32)
    actions_array = np.asarray(accepted_actions, dtype=np.float32)
    return {
        "requested_episodes": episodes,
        "accepted_episodes": accepted_episode_count,
        "qualifying_episodes": qualifying_episode_count,
        "accepted_samples": int(observations_array.shape[0]),
        "mean_episode_useful_bounces": float(np.mean(accepted_episode_useful_bounces)),
        "sample_mode": sample_mode,
        "observations": observations_array,
        "actions": actions_array,
    }


def bootstrap_actor_from_dataset(
    *,
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    # PPO rollout 전에 policy mean action을 heuristic action에 맞추는 supervised warm start다.
    # LINK: pingpong_rl2/scripts/run_ppo_learning.py:162
    if epochs <= 0 or observations.size == 0 or actions.size == 0:
        return {
            "epochs": epochs,
            "samples": int(observations.shape[0]) if observations.ndim == 2 else 0,
            "mean_loss": None,
            "last_loss": None,
        }

    optimizer = th.optim.Adam(model.policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    loss_history: list[float] = []
    model.policy.train()
    # SB3 distribution에서 deterministic action을 뽑아 MSE로 actor parameters만 직접 업데이트한다.
    for _ in range(epochs):
        permutation = rng.permutation(observations.shape[0])
        batch_count = max(1, ceil(observations.shape[0] / batch_size))
        for batch_index in range(batch_count):
            batch_slice = permutation[batch_index * batch_size:(batch_index + 1) * batch_size]
            if batch_slice.size == 0:
                continue
            observation_tensor = th.as_tensor(observations[batch_slice], device=model.device)
            action_tensor = th.as_tensor(actions[batch_slice], device=model.device)
            distribution = model.policy.get_distribution(observation_tensor)
            predicted_action_tensor = distribution.get_actions(deterministic=True)
            loss = th.nn.functional.mse_loss(predicted_action_tensor, action_tensor)
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()
            loss_history.append(float(loss.detach().cpu().item()))

    return {
        "epochs": epochs,
        "samples": int(observations.shape[0]),
        "mean_loss": float(np.mean(loss_history)) if loss_history else None,
        "last_loss": loss_history[-1] if loss_history else None,
    }

