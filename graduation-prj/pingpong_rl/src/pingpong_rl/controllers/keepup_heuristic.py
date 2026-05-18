from __future__ import annotations

from typing import Sequence

import numpy as np

from pingpong_rl.controllers.ee_pose_controller import RacketCartesianController
from pingpong_rl.envs.pingpong_env import PingPongSim


def _solve_intercept_time(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    target_z: float,
    gravity_z: float,
    fallback_time: float,
    max_intercept_time: float,
) -> float:
    quadratic_a = 0.5 * float(gravity_z)
    quadratic_b = float(ball_velocity[2])
    quadratic_c = float(ball_position[2] - target_z)

    candidate_times: list[float] = []
    if abs(quadratic_a) < 1.0e-9:
        if abs(quadratic_b) > 1.0e-9:
            candidate_times.append(-quadratic_c / quadratic_b)
    else:
        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        if discriminant >= 0.0:
            sqrt_discriminant = float(np.sqrt(discriminant))
            denominator = 2.0 * quadratic_a
            candidate_times.extend(
                [
                    (-quadratic_b - sqrt_discriminant) / denominator,
                    (-quadratic_b + sqrt_discriminant) / denominator,
                ]
            )

    valid_times = [time_value for time_value in candidate_times if 0.0 <= time_value <= max_intercept_time]
    if valid_times:
        return min(valid_times)
    return float(np.clip(fallback_time, 0.0, max_intercept_time))


def compute_keepup_target(
    anchor_position: Sequence[float],
    ball_position: Sequence[float],
    ball_velocity: Sequence[float],
    *,
    preview_time: float = 0.10,
    strike_plane_offset: float = 0.02,
    return_height_offset: float = -0.03,
    max_xy_offset: float = 0.18,
    min_z_offset: float = -0.03,
    max_z_offset: float = 0.10,
    gravity_z: float = -9.81,
    max_intercept_time: float = 0.35,
) -> np.ndarray:
    anchor = np.asarray(anchor_position, dtype=float)
    ball_pos = np.asarray(ball_position, dtype=float)
    ball_vel = np.asarray(ball_velocity, dtype=float)

    if anchor.shape != (3,):
        raise ValueError(f"anchor_position must have shape (3,), got {anchor.shape}.")
    if ball_pos.shape != (3,):
        raise ValueError(f"ball_position must have shape (3,), got {ball_pos.shape}.")
    if ball_vel.shape != (3,):
        raise ValueError(f"ball_velocity must have shape (3,), got {ball_vel.shape}.")

    target = anchor.copy()
    strike_plane_z = float(anchor[2]) + float(np.clip(strike_plane_offset, min_z_offset, max_z_offset))
    if float(ball_vel[2]) <= 0.0:
        intercept_time = _solve_intercept_time(
            ball_position=ball_pos,
            ball_velocity=ball_vel,
            target_z=strike_plane_z,
            gravity_z=gravity_z,
            fallback_time=preview_time,
            max_intercept_time=max_intercept_time,
        )
        predicted_xy = ball_pos[:2] + intercept_time * ball_vel[:2]
    else:
        predicted_xy = anchor[:2]
    xy_offset = np.clip(predicted_xy - anchor[:2], -float(max_xy_offset), float(max_xy_offset))
    target[:2] = anchor[:2] + xy_offset

    if float(ball_vel[2]) <= 0.0:
        desired_z = strike_plane_z
    else:
        desired_z = float(anchor[2]) + float(return_height_offset)
    target[2] = float(anchor[2]) + np.clip(
        desired_z - float(anchor[2]),
        float(min_z_offset),
        float(max_z_offset),
    )
    return target


class KeepUpHeuristicController:
    def __init__(
        self,
        sim: PingPongSim,
        *,
        preview_time: float = 0.10,
        strike_plane_offset: float = 0.02,
        return_height_offset: float = -0.03,
        max_xy_offset: float = 0.18,
        min_z_offset: float = -0.03,
        max_z_offset: float = 0.10,
        max_intercept_time: float = 0.35,
        descent_resume_velocity: float = -0.05,
        contact_hold_steps: int = 2,
        position_gain: float = 0.2,
        damping: float = 1.0e-3,
        max_position_step: float = 0.03,
    ) -> None:
        self._sim = sim
        self._preview_time = float(preview_time)
        self._strike_plane_offset = float(strike_plane_offset)
        self._return_height_offset = float(return_height_offset)
        self._max_xy_offset = float(max_xy_offset)
        self._min_z_offset = float(min_z_offset)
        self._max_z_offset = float(max_z_offset)
        self._max_intercept_time = float(max_intercept_time)
        self._descent_resume_velocity = float(descent_resume_velocity)
        self._contact_hold_steps = max(0, int(contact_hold_steps))
        self._cartesian_controller = RacketCartesianController(
            sim,
            damping=float(damping),
            position_gain=float(position_gain),
            max_position_step=max_position_step,
        )
        self._anchor_position = sim.racket_position.copy()
        self._holding_after_contact = False
        self._contact_hold_steps_remaining = 0

    @property
    def anchor_position(self) -> np.ndarray:
        return self._anchor_position.copy()

    def set_anchor_position(self, position: Sequence[float]) -> np.ndarray:
        position_array = np.asarray(position, dtype=float)
        if position_array.shape != (3,):
            raise ValueError(f"Anchor position must have shape (3,), got {position_array.shape}.")
        self._anchor_position = position_array.copy()
        return self.anchor_position

    def reset(self) -> np.ndarray:
        self._anchor_position = self._sim.racket_position.copy()
        self._holding_after_contact = False
        self._contact_hold_steps_remaining = 0
        self._cartesian_controller.reset()
        return self.anchor_position

    def notify_contact_event(self, contact_event: bool) -> None:
        if not contact_event:
            return
        self._holding_after_contact = True
        self._contact_hold_steps_remaining = self._contact_hold_steps

    def _should_resume_tracking(self) -> bool:
        return float(self._sim.ball_velocity[2]) <= self._descent_resume_velocity

    def target_position(self) -> np.ndarray:
        if self._holding_after_contact:
            if self._contact_hold_steps_remaining > 0:
                self._contact_hold_steps_remaining -= 1
            elif self._should_resume_tracking():
                self._holding_after_contact = False

        return compute_keepup_target(
            self._anchor_position,
            self._sim.ball_position,
            self._sim.ball_velocity if not self._holding_after_contact else np.array([0.0, 0.0, 1.0], dtype=float),
            preview_time=self._preview_time,
            strike_plane_offset=self._strike_plane_offset,
            return_height_offset=self._return_height_offset,
            max_xy_offset=self._max_xy_offset,
            min_z_offset=self._min_z_offset,
            max_z_offset=self._max_z_offset,
            gravity_z=float(self._sim.model.opt.gravity[2]),
            max_intercept_time=self._max_intercept_time,
        )

    def compute_joint_targets(self) -> np.ndarray:
        self._cartesian_controller.set_target_position(self.target_position())
        return self._cartesian_controller.compute_joint_targets()