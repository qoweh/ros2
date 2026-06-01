from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from pingpong_rl2.envs.pingpong_sim import PingPongSim


class RacketCartesianController:
    def __init__(
        self,
        sim: PingPongSim,
        damping: float = 1.0e-4,
        position_gain: float = 1.2,
        orientation_gain: float = 0.35,
        max_position_step: float = 0.05,
        max_orientation_step: float = 0.12,
        velocity_gain: float = 1.0,
        velocity_feedback_gain: float = 0.0,
        max_velocity_step: float = 0.02,
        target_offset_low: Sequence[float] | None = None,
        target_offset_high: Sequence[float] | None = None,
        target_tilt_limit: Sequence[float] | None = None,
    ) -> None:
        self._sim = sim
        self._damping = float(damping)
        self._position_gain = float(position_gain)
        self._orientation_gain = float(orientation_gain)
        self._max_position_step = float(max_position_step)
        self._max_orientation_step = float(max_orientation_step)
        self._velocity_gain = float(velocity_gain)
        self._velocity_feedback_gain = float(velocity_feedback_gain)
        self._max_velocity_step = float(max_velocity_step)
        self._target_offset_low = None if target_offset_low is None else np.asarray(target_offset_low, dtype=float)
        self._target_offset_high = None if target_offset_high is None else np.asarray(target_offset_high, dtype=float)
        self._target_tilt_limit = (
            np.asarray(target_tilt_limit, dtype=float)
            if target_tilt_limit is not None
            else np.array([0.35, 0.35], dtype=float)
        )
        if self._target_offset_low is not None and self._target_offset_low.shape != (3,):
            raise ValueError(f"target_offset_low must have shape (3,), got {self._target_offset_low.shape}.")
        if self._target_offset_high is not None and self._target_offset_high.shape != (3,):
            raise ValueError(f"target_offset_high must have shape (3,), got {self._target_offset_high.shape}.")
        if self._target_tilt_limit.shape != (2,):
            raise ValueError(f"target_tilt_limit must have shape (2,), got {self._target_tilt_limit.shape}.")
        if (self._target_offset_low is None) != (self._target_offset_high is None):
            raise ValueError("target_offset_low and target_offset_high must be provided together.")
        if self._target_offset_low is not None and np.any(self._target_offset_low > self._target_offset_high):
            raise ValueError(
                f"target_offset_low must be <= target_offset_high, got {self._target_offset_low} "
                f"and {self._target_offset_high}."
            )
        if np.any(self._target_tilt_limit < 0.0):
            raise ValueError(f"target_tilt_limit must be non-negative, got {self._target_tilt_limit}.")
        if self._velocity_gain < 0.0:
            raise ValueError(f"velocity_gain must be non-negative, got {self._velocity_gain}.")
        if self._velocity_feedback_gain < 0.0:
            raise ValueError(f"velocity_feedback_gain must be non-negative, got {self._velocity_feedback_gain}.")
        if self._max_velocity_step < 0.0:
            raise ValueError(f"max_velocity_step must be non-negative, got {self._max_velocity_step}.")

        self._joint_ids = [sim.model.joint(f"joint{index}").id for index in range(1, 8)]
        self._joint_qpos_indices = np.array([sim.model.jnt_qposadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._joint_dof_indices = np.array([sim.model.jnt_dofadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._joint_limits = sim.model.jnt_range[self._joint_ids].copy()
        self._position_jacobian = np.zeros((3, sim.model.nv), dtype=float)
        self._rotation_jacobian = np.zeros((3, sim.model.nv), dtype=float)
        self.targets = sim.home_joint_targets.copy()
        self._target_position = sim.racket_position.copy()
        self._anchor_position = sim.racket_position.copy()
        self._target_tilt = np.zeros(2, dtype=float)
        self._target_velocity = np.zeros(3, dtype=float)
        self._target_velocity_enabled = False

    @property
    def target_position(self) -> np.ndarray:
        return self._target_position.copy()

    @property
    def target_tilt(self) -> np.ndarray:
        return self._target_tilt.copy()

    @property
    def target_velocity(self) -> np.ndarray:
        return self._target_velocity.copy()

    @property
    def target_face_normal(self) -> np.ndarray:
        return self._target_face_normal_from_tilt(self._target_tilt)

    def reset(self) -> np.ndarray:
        self.targets[:] = self._sim.home_joint_targets
        self._anchor_position = self._sim.racket_position.copy()
        self._target_position = self._sim.racket_position.copy()
        self._target_tilt = np.zeros(2, dtype=float)
        self._target_velocity = np.zeros(3, dtype=float)
        self._target_velocity_enabled = False
        return self.targets.copy()

    def _clip_target_position(self, position: np.ndarray) -> np.ndarray:
        if self._target_offset_low is None or self._target_offset_high is None:
            return position
        target_offset = np.clip(position - self._anchor_position, self._target_offset_low, self._target_offset_high)
        return self._anchor_position + target_offset

    def set_target_position(self, position: Sequence[float]) -> np.ndarray:
        position_array = np.asarray(position, dtype=float)
        if position_array.shape != (3,):
            raise ValueError(f"Target position must have shape (3,), got {position_array.shape}.")
        self._target_position = self._clip_target_position(position_array)
        return self.target_position

    def add_target_offset(self, delta: Sequence[float]) -> np.ndarray:
        delta_array = np.asarray(delta, dtype=float)
        if delta_array.shape != (3,):
            raise ValueError(f"Target delta must have shape (3,), got {delta_array.shape}.")
        self._target_position = self._clip_target_position(self._target_position + delta_array)
        return self.target_position

    def set_target_tilt(self, tilt: Sequence[float]) -> np.ndarray:
        tilt_array = np.asarray(tilt, dtype=float)
        if tilt_array.shape != (2,):
            raise ValueError(f"Target tilt must have shape (2,), got {tilt_array.shape}.")
        self._target_tilt = np.clip(tilt_array, -self._target_tilt_limit, self._target_tilt_limit)
        return self.target_tilt

    def set_target_velocity(self, velocity: Sequence[float] | None) -> np.ndarray:
        if velocity is None:
            self._target_velocity = np.zeros(3, dtype=float)
            self._target_velocity_enabled = False
            return self.target_velocity
        velocity_array = np.asarray(velocity, dtype=float)
        if velocity_array.shape != (3,):
            raise ValueError(f"Target velocity must have shape (3,), got {velocity_array.shape}.")
        self._target_velocity = velocity_array
        self._target_velocity_enabled = True
        return self.target_velocity

    @staticmethod
    def _target_face_normal_from_tilt(tilt: np.ndarray) -> np.ndarray:
        pitch = float(tilt[0])
        roll = float(tilt[1])
        rotation_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll), -np.sin(roll)],
                [0.0, np.sin(roll), np.cos(roll)],
            ],
            dtype=float,
        )
        rotation_y = np.array(
            [
                [np.cos(pitch), 0.0, np.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch), 0.0, np.cos(pitch)],
            ],
            dtype=float,
        )
        normal = rotation_y @ rotation_x @ np.array([0.0, 0.0, -1.0], dtype=float)
        return normal / max(np.linalg.norm(normal), 1.0e-9)

    def compute_joint_targets(self) -> np.ndarray:
        current_position = self._sim.racket_position
        position_error = self._target_position - current_position
        error_norm = np.linalg.norm(position_error)
        if error_norm > self._max_position_step:
            position_error = position_error * (self._max_position_step / error_norm)
        velocity_step = np.zeros(3, dtype=float)
        if self._target_velocity_enabled:
            velocity_command = self._velocity_gain * self._target_velocity
            if self._velocity_feedback_gain > 0.0:
                velocity_command = velocity_command + self._velocity_feedback_gain * (
                    self._target_velocity - self._sim.racket_velocity
                )
            velocity_step = velocity_command * self._sim.control_dt
        velocity_step_norm = np.linalg.norm(velocity_step)
        if self._max_velocity_step > 0.0 and velocity_step_norm > self._max_velocity_step:
            velocity_step = velocity_step * (self._max_velocity_step / velocity_step_norm)

        current_face_normal = self._sim.racket_face_normal
        target_face_normal = self.target_face_normal
        orientation_error = np.cross(current_face_normal, target_face_normal)
        orientation_error_norm = np.linalg.norm(orientation_error)
        if orientation_error_norm > self._max_orientation_step:
            orientation_error = orientation_error * (self._max_orientation_step / orientation_error_norm)

        mujoco.mj_jacSite(
            self._sim.model,
            self._sim.data,
            self._position_jacobian,
            self._rotation_jacobian,
            self._sim.racket_site_id,
        )
        task_jacobian = np.vstack(
            [
                self._position_jacobian[:, self._joint_dof_indices],
                self._rotation_jacobian[:, self._joint_dof_indices],
            ]
        )
        task_error = np.concatenate(
            [
                self._position_gain * position_error + velocity_step,
                self._orientation_gain * orientation_error,
            ]
        )
        task_metric = task_jacobian @ task_jacobian.T + self._damping * np.eye(6)
        delta_q = task_jacobian.T @ np.linalg.solve(task_metric, task_error)

        current_joint_positions = self._sim.data.qpos[self._joint_qpos_indices]
        next_targets = current_joint_positions + delta_q
        clipped_targets = np.clip(next_targets, self._joint_limits[:, 0], self._joint_limits[:, 1])
        self.targets[:] = clipped_targets
        return self.targets.copy()
