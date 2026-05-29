from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pingpong_rl2.envs.keepup_env import PingPongKeepUpEnv


@dataclass(slots=True)
class HeuristicKeepUpPolicy:
    return_blend: float = 0.72
    recovery_blend: float = 0.52
    strike_z_boost: float = 0.018
    strike_time_horizon: float = 0.14

    def reset(self) -> None:
        return None

    def predict(self, env: PingPongKeepUpEnv) -> np.ndarray:
        if env.action_mode != "position_strike":
            raise ValueError(
                "HeuristicKeepUpPolicy requires action_mode='position_strike' so it can steer the strike controller."
            )

        base_target = env._strike_action_target_position(np.zeros(3, dtype=float))
        anchor_position = env._controller_anchor_position()
        desired_target = base_target.copy()
        phase_name = env._phase_name()

        if phase_name in {"prepare", "strike"}:
            intercept_time = env._predicted_intercept_time()
            urgency = 1.0 - np.clip(intercept_time / max(self.strike_time_horizon, 1.0e-6), 0.0, 1.0)
            strike_readiness = max(env._pre_contact_height_readiness(), urgency)
            desired_target[2] = base_target[2] + self.strike_z_boost * strike_readiness
        else:
            next_intercept_metrics = env._next_intercept_metrics()
            if next_intercept_metrics["time"] > 0.0:
                next_intercept_xy = env.sim.racket_position[:2] + np.asarray(next_intercept_metrics["relative_xy"], dtype=float)
                blend = self.return_blend if phase_name == "return_shaping" else self.recovery_blend
                desired_target[:2] = (1.0 - blend) * base_target[:2] + blend * next_intercept_xy
            desired_target[2] = base_target[2]

        action = desired_target - base_target
        return np.clip(action, env.action_low, env.action_high)