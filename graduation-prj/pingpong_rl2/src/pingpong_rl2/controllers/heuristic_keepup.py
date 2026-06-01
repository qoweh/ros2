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
    fixed_position_residual: tuple[float, float, float] = (0.0, 0.0, 0.0)
    strike_position_residual: tuple[float, float, float] | None = None
    recovery_position_residual: tuple[float, float, float] | None = None
    fixed_tilt_residual_pitch: float = 0.0
    fixed_tilt_residual_roll: float = 0.0
    strike_tilt_residual_pitch: float | None = None
    strike_tilt_residual_roll: float | None = None
    recovery_tilt_residual_pitch: float | None = None
    recovery_tilt_residual_roll: float | None = None
    fixed_followup_lift_residual: float = 0.0
    strike_followup_lift_residual: float | None = None
    recovery_followup_lift_residual: float | None = None

    def reset(self) -> None:
        return None

    def _position_residual_for_phase(self, phase_name: str) -> np.ndarray:
        if phase_name in {"prepare", "strike"}:
            position_residual = self.strike_position_residual
        elif phase_name in {"return_shaping", "recovery"}:
            position_residual = self.recovery_position_residual
        else:
            position_residual = None

        if position_residual is None:
            position_residual = self.fixed_position_residual
        return np.asarray(position_residual, dtype=float)

    def _tilt_residual_for_phase(self, phase_name: str) -> np.ndarray:
        if phase_name in {"prepare", "strike"}:
            pitch = self.strike_tilt_residual_pitch
            roll = self.strike_tilt_residual_roll
        elif phase_name in {"return_shaping", "recovery"}:
            pitch = self.recovery_tilt_residual_pitch
            roll = self.recovery_tilt_residual_roll
        else:
            pitch = None
            roll = None

        if pitch is None:
            pitch = self.fixed_tilt_residual_pitch
        if roll is None:
            roll = self.fixed_tilt_residual_roll
        return np.array([pitch, roll], dtype=float)

    def _followup_lift_residual_for_phase(self, phase_name: str) -> float:
        if phase_name in {"prepare", "strike"}:
            followup_lift_residual = self.strike_followup_lift_residual
        elif phase_name in {"return_shaping", "recovery"}:
            followup_lift_residual = self.recovery_followup_lift_residual
        else:
            followup_lift_residual = None

        if followup_lift_residual is None:
            followup_lift_residual = self.fixed_followup_lift_residual
        return float(followup_lift_residual)

    def predict(self, env: PingPongKeepUpEnv) -> np.ndarray:
        if env.action_mode not in ("position_strike", "position_strike_tilt", "position_strike_tilt_lift"):
            raise ValueError(
                "HeuristicKeepUpPolicy requires a strike-contract action mode so it can steer the strike controller."
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

        desired_target = desired_target + self._position_residual_for_phase(phase_name)
        action = desired_target - base_target
        if env.action_mode == "position_strike_tilt":
            tilt_residual = self._tilt_residual_for_phase(phase_name)
            action = np.concatenate([action, tilt_residual])
        elif env.action_mode == "position_strike_tilt_lift":
            tilt_residual = self._tilt_residual_for_phase(phase_name)
            followup_lift_residual = np.array([self._followup_lift_residual_for_phase(phase_name)], dtype=float)
            action = np.concatenate([action, tilt_residual, followup_lift_residual])
        return np.clip(action, env.action_low, env.action_high)