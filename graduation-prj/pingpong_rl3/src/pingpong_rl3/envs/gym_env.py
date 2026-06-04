from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pingpong_rl3.envs.two_ball_keepup_env import TwoBallKeepUpEnv


def _vector_safe_info(info: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in info.items() if value is not None}


class TwoBallKeepUpGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs: object) -> None:
        super().__init__()
        self.base_env = TwoBallKeepUpEnv(**env_kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_env.observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self.base_env.action_low.astype(np.float32),
            high=self.base_env.action_high.astype(np.float32),
            shape=(self.base_env.action_size,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        observation, info = self.base_env.reset(seed=seed, options=options)
        return observation.astype(np.float32, copy=False), _vector_safe_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.base_env.step(action)
        return observation.astype(np.float32, copy=False), reward, terminated, truncated, _vector_safe_info(info)

    def training_config(self) -> dict[str, object]:
        return self.base_env.training_config()

    def close(self) -> None:
        self.base_env.close()
