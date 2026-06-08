from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pingpong_rl.envs.ee_delta_env import PingPongEEDeltaEnv


class PingPongEEDeltaGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, **env_kwargs: object) -> None:
        super().__init__()
        self.base_env = PingPongEEDeltaEnv(**env_kwargs)
        observation_shape = (self.base_env.observation_size,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,
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
        if seed is not None:
            self.base_env.seed(seed)
        options = {} if options is None else dict(options)
        ball_height = options.get("ball_height")
        ball_velocity = options.get("ball_velocity")
        ball_xy_offset = options.get("ball_xy_offset")
        observation, info = self.base_env.reset(
            ball_height=ball_height,
            ball_velocity=ball_velocity,
            ball_xy_offset=ball_xy_offset,
        )
        return observation.astype(np.float32, copy=False), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.base_env.step(action)
        return observation.astype(np.float32, copy=False), reward, terminated, truncated, info

    def apply_curriculum_stage(self, stage_name: str, stage_updates: dict[str, object]) -> None:
        self.base_env.apply_curriculum_stage(stage_name, stage_updates)

    def training_config(self) -> dict[str, object]:
        return self.base_env.training_config()

    def close(self) -> None:
        return None
