from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from pingpong_rl3.controllers import RacketCartesianController
from pingpong_rl3.defaults import (
    DEFAULT_APEX_XY_TOLERANCE,
    DEFAULT_CONTROL_DT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_MAX_USEFUL_APEX_HEIGHT,
    DEFAULT_MIN_USEFUL_APEX_HEIGHT,
    DEFAULT_RESET_HEIGHT_JITTER,
    DEFAULT_RESET_SPIN_RANGE,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_TARGET_APEX_HEIGHT,
)
from pingpong_rl3.envs.two_ball_sim import TwoBallPingPongSim

GRAVITY = 9.81

ACTION_NAMES = (
    "contact_x_residual",
    "contact_y_residual",
    "contact_z_residual",
    "tilt_pitch",
    "tilt_roll",
    "racket_vx_residual",
    "racket_vy_residual",
    "racket_vz_residual",
    "outgoing_vx_residual",
    "outgoing_vy_residual",
    "target_apex_z_residual",
    "ball_0_priority_bias",
    "ball_1_priority_bias",
)


class TwoBallKeepUpEnv:
    def __init__(
        self,
        sim: TwoBallPingPongSim | None = None,
        scene_path: Path | str | None = None,
        control_dt: float = DEFAULT_CONTROL_DT,
        max_episode_steps: int | None = DEFAULT_MAX_EPISODE_STEPS,
        target_apex_height: float = DEFAULT_TARGET_APEX_HEIGHT,
        min_useful_apex_height: float = DEFAULT_MIN_USEFUL_APEX_HEIGHT,
        max_useful_apex_height: float = DEFAULT_MAX_USEFUL_APEX_HEIGHT,
        apex_xy_tolerance: float = DEFAULT_APEX_XY_TOLERANCE,
        reset_xy_range: float = DEFAULT_RESET_XY_RANGE,
        reset_height_jitter: float = DEFAULT_RESET_HEIGHT_JITTER,
        reset_base_heights: Sequence[float] = (0.40, 0.67),
        reset_velocity_xy_range: float = DEFAULT_RESET_VELOCITY_XY_RANGE,
        reset_velocity_z_range: Sequence[float] = DEFAULT_RESET_VELOCITY_Z_RANGE,
        reset_spin_range: float = DEFAULT_RESET_SPIN_RANGE,
        strike_plane_offset: float = 0.145,
        reachable_radius: float = 0.17,
        min_intercept_time: float = 0.045,
        max_intercept_time: float = 0.75,
        target_velocity_max: float = 2.2,
        intercept_velocity_gain: float = 0.65,
        restitution: float = 0.82,
        lateral_velocity_max: float = 1.15,
        tilt_gain: float = 1.15,
        tilt_limit: Sequence[float] = (0.28, 0.28),
        min_useful_outgoing_vz: float = 0.35,
        min_useful_racket_vz: float = 0.0,
        action_penalty_weight: float = 0.015,
        tracking_reward_weight: float = 0.08,
        contact_bonus: float = 2.0,
        useful_contact_bonus: float = 7.0,
        alternating_bonus: float = 1.0,
        contact_apex_under_min_penalty_weight: float = 0.0,
        contact_apex_progress_reward_weight: float = 0.0,
        contact_apex_recovery_progress_reward_weight: float = 0.0,
        contact_apex_potential_reward_weight: float = 0.0,
        contact_apex_potential_gamma: float = 0.99,
        contact_apex_potential_cap: float = 2.0,
        low_apex_recovery_lift_gain: float = 0.0,
        low_apex_recovery_lift_max: float = 0.0,
        low_apex_recovery_velocity_gain: float = 0.0,
        low_apex_recovery_velocity_max: float = 0.0,
        failure_penalty: float = -12.0,
        slot_xy_offsets: Sequence[Sequence[float]] = ((-0.035, -0.025), (0.035, 0.025)),
        x_bounds: Sequence[float] = (0.0, 1.35),
        y_bounds: Sequence[float] = (-0.65, 0.65),
        z_bounds: Sequence[float] = (-0.05, 2.1),
        max_ball_speed: float = 8.0,
        terminate_on_ball_ball_contact: bool = True,
        controller_position_gain: float = 1.25,
        controller_orientation_gain: float = 0.42,
        controller_max_position_step: float = 0.055,
        controller_max_orientation_step: float = 0.13,
        controller_velocity_gain: float = 1.0,
        controller_velocity_feedback_gain: float = 0.25,
        controller_max_velocity_step: float = 0.028,
        seed: int | None = None,
    ) -> None:
        self.sim = sim if sim is not None else TwoBallPingPongSim(scene_path=scene_path, control_dt=control_dt)
        if max_episode_steps is None or int(max_episode_steps) <= 0:
            self.max_episode_steps = None
        else:
            self.max_episode_steps = int(max_episode_steps)
        self.target_apex_height = float(target_apex_height)
        self.min_useful_apex_height = float(min_useful_apex_height)
        self.max_useful_apex_height = float(max_useful_apex_height)
        self.apex_xy_tolerance = float(apex_xy_tolerance)
        self.reset_xy_range = float(reset_xy_range)
        self.reset_height_jitter = float(reset_height_jitter)
        self.reset_base_heights = np.asarray(reset_base_heights, dtype=float)
        self.reset_velocity_xy_range = float(reset_velocity_xy_range)
        self.reset_velocity_z_range = tuple(float(value) for value in reset_velocity_z_range)
        self.reset_spin_range = float(reset_spin_range)
        self.strike_plane_offset = float(strike_plane_offset)
        self.reachable_radius = float(reachable_radius)
        self.min_intercept_time = float(min_intercept_time)
        self.max_intercept_time = float(max_intercept_time)
        self.target_velocity_max = float(target_velocity_max)
        self.intercept_velocity_gain = float(intercept_velocity_gain)
        self.restitution = float(restitution)
        self.lateral_velocity_max = float(lateral_velocity_max)
        self.tilt_gain = float(tilt_gain)
        self.tilt_limit = np.asarray(tilt_limit, dtype=float)
        self.min_useful_outgoing_vz = float(min_useful_outgoing_vz)
        self.min_useful_racket_vz = float(min_useful_racket_vz)
        self.action_penalty_weight = float(action_penalty_weight)
        self.tracking_reward_weight = float(tracking_reward_weight)
        self.contact_bonus = float(contact_bonus)
        self.useful_contact_bonus = float(useful_contact_bonus)
        self.alternating_bonus = float(alternating_bonus)
        self.contact_apex_under_min_penalty_weight = float(contact_apex_under_min_penalty_weight)
        self.contact_apex_progress_reward_weight = float(contact_apex_progress_reward_weight)
        self.contact_apex_recovery_progress_reward_weight = float(contact_apex_recovery_progress_reward_weight)
        self.contact_apex_potential_reward_weight = float(contact_apex_potential_reward_weight)
        self.contact_apex_potential_gamma = float(contact_apex_potential_gamma)
        self.contact_apex_potential_cap = float(contact_apex_potential_cap)
        self.low_apex_recovery_lift_gain = float(low_apex_recovery_lift_gain)
        self.low_apex_recovery_lift_max = float(low_apex_recovery_lift_max)
        self.low_apex_recovery_velocity_gain = float(low_apex_recovery_velocity_gain)
        self.low_apex_recovery_velocity_max = float(low_apex_recovery_velocity_max)
        self.failure_penalty = float(failure_penalty)
        self.slot_xy_offsets = np.asarray(slot_xy_offsets, dtype=float)
        self.x_bounds = (float(x_bounds[0]), float(x_bounds[1]))
        self.y_bounds = (float(y_bounds[0]), float(y_bounds[1]))
        self.z_bounds = (float(z_bounds[0]), float(z_bounds[1]))
        self.max_ball_speed = float(max_ball_speed)
        self.terminate_on_ball_ball_contact = bool(terminate_on_ball_ball_contact)
        self.controller_position_gain = float(controller_position_gain)
        self.controller_orientation_gain = float(controller_orientation_gain)
        self.controller_max_position_step = float(controller_max_position_step)
        self.controller_max_orientation_step = float(controller_max_orientation_step)
        self.controller_velocity_gain = float(controller_velocity_gain)
        self.controller_velocity_feedback_gain = float(controller_velocity_feedback_gain)
        self.controller_max_velocity_step = float(controller_max_velocity_step)
        self.rng = np.random.default_rng(seed)

        if self.slot_xy_offsets.shape != (self.sim.ball_count, 2):
            raise ValueError(f"slot_xy_offsets must have shape ({self.sim.ball_count}, 2).")
        if self.tilt_limit.shape != (2,):
            raise ValueError(f"tilt_limit must have shape (2,), got {self.tilt_limit.shape}.")
        if self.reset_base_heights.shape != (self.sim.ball_count,):
            raise ValueError(f"reset_base_heights must have shape ({self.sim.ball_count},).")
        if self.min_useful_outgoing_vz < 0.0:
            raise ValueError(f"min_useful_outgoing_vz must be non-negative, got {self.min_useful_outgoing_vz}.")
        if self.contact_apex_potential_cap <= 0.0:
            raise ValueError(f"contact_apex_potential_cap must be positive, got {self.contact_apex_potential_cap}.")
        if self.reset_velocity_z_range[0] > self.reset_velocity_z_range[1]:
            raise ValueError(f"reset_velocity_z_range must be sorted, got {self.reset_velocity_z_range}.")
        if not (0.0 < self.min_useful_apex_height <= self.target_apex_height <= self.max_useful_apex_height):
            raise ValueError(
                "Expected 0 < min_useful_apex_height <= target_apex_height <= max_useful_apex_height."
            )

        self.controller = RacketCartesianController(
            self.sim,
            position_gain=self.controller_position_gain,
            orientation_gain=self.controller_orientation_gain,
            max_position_step=self.controller_max_position_step,
            max_orientation_step=self.controller_max_orientation_step,
            velocity_gain=self.controller_velocity_gain,
            velocity_feedback_gain=self.controller_velocity_feedback_gain,
            max_velocity_step=self.controller_max_velocity_step,
            target_offset_low=(-0.20, -0.20, -0.08),
            target_offset_high=(0.20, 0.20, 0.24),
            target_tilt_limit=self.tilt_limit,
            nullspace_posture_gain=0.015,
            nullspace_posture_max_step=0.010,
            body_clearance_gain=0.30,
            body_clearance_margin=0.10,
            body_clearance_vertical_margin=0.26,
            body_clearance_max_step=0.010,
        )

        self.action_low = np.array(
            [-0.05, -0.05, -0.035, -0.16, -0.16, -0.45, -0.45, -0.20, -0.45, -0.45, -0.10, -0.12, -0.12],
            dtype=float,
        )
        self.action_high = np.array(
            [0.05, 0.05, 0.035, 0.16, 0.16, 0.45, 0.45, 0.90, 0.45, 0.45, 0.10, 0.12, 0.12],
            dtype=float,
        )
        self.action_size = len(ACTION_NAMES)
        self._anchor_position = self.sim.racket_position.copy()
        self._previous_action = np.zeros(self.action_size, dtype=float)
        self._previous_racket_contacts = [False for _ in range(self.sim.ball_count)]
        self._last_projected_apex_heights = np.full(self.sim.ball_count, np.nan, dtype=float)
        self._last_contact_apex_shortfalls = np.zeros(self.sim.ball_count, dtype=float)
        self._last_hit_ball_index: int | None = None
        self._last_target_ball_index = 0
        self.elapsed_steps = 0
        self.contact_count = 0
        self.useful_bounce_count = 0
        self.nonuseful_contact_count = 0
        self.observation_size = int(self._get_observation().shape[0])

    def seed(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if seed is not None:
            self.seed(seed)
        options = {} if options is None else dict(options)

        self.sim.reset()
        self.controller.reset()
        self._anchor_position = self.sim.racket_position.copy()

        ball_positions, ball_velocities, ball_spins = self._sample_reset_balls(options)
        self.sim.reset(
            ball_positions=ball_positions,
            ball_velocities=ball_velocities,
            ball_angular_velocities=ball_spins,
        )
        self.controller.reset()
        self._anchor_position = self.sim.racket_position.copy()
        self._previous_action[:] = 0.0
        self._previous_racket_contacts = [False for _ in range(self.sim.ball_count)]
        self._last_projected_apex_heights[:] = np.nan
        self._last_contact_apex_shortfalls[:] = 0.0
        self._last_hit_ball_index = None
        self._last_target_ball_index = 0
        self.elapsed_steps = 0
        self.contact_count = 0
        self.useful_bounce_count = 0
        self.nonuseful_contact_count = 0
        return self._get_observation(), self._info()

    def _sample_reset_balls(self, options: dict[str, object]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        xy_range = float(options.get("reset_xy_range", self.reset_xy_range))
        height_jitter = float(options.get("reset_height_jitter", self.reset_height_jitter))
        base_heights = np.asarray(options.get("reset_base_heights", self.reset_base_heights), dtype=float)
        velocity_xy_range = float(options.get("reset_velocity_xy_range", self.reset_velocity_xy_range))
        velocity_z_range = tuple(options.get("reset_velocity_z_range", self.reset_velocity_z_range))
        spin_range = float(options.get("reset_spin_range", self.reset_spin_range))
        if base_heights.shape != (self.sim.ball_count,):
            raise ValueError(f"reset_base_heights must have shape ({self.sim.ball_count},).")

        base_heights = base_heights.copy()
        if bool(self.rng.integers(0, 2)):
            base_heights = base_heights[::-1]

        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        spins: list[np.ndarray] = []
        for ball_index in range(self.sim.ball_count):
            slot_xy = self.slot_xy_offsets[ball_index]
            xy_offset = slot_xy + self._sample_disk_xy(xy_range)
            z_offset = base_heights[ball_index] + self.rng.uniform(-height_jitter, height_jitter)
            positions.append(self._anchor_position + np.array([xy_offset[0], xy_offset[1], z_offset], dtype=float))
            velocities.append(
                np.array(
                    [
                        self.rng.uniform(-velocity_xy_range, velocity_xy_range),
                        self.rng.uniform(-velocity_xy_range, velocity_xy_range),
                        self.rng.uniform(float(velocity_z_range[0]), float(velocity_z_range[1])),
                    ],
                    dtype=float,
                )
            )
            spins.append(self.rng.uniform(-spin_range, spin_range, size=3).astype(float))
        return positions, velocities, spins

    def _sample_disk_xy(self, radius: float) -> np.ndarray:
        if radius <= 0.0:
            return np.zeros(2, dtype=float)
        angle = self.rng.uniform(0.0, 2.0 * np.pi)
        distance = radius * np.sqrt(self.rng.uniform(0.0, 1.0))
        return np.array([distance * np.cos(angle), distance * np.sin(angle)], dtype=float)

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_array = np.clip(np.asarray(action, dtype=float), self.action_low, self.action_high)
        previous_contacts = [
            self.sim.has_contact(ball.geom_name, "racket_head")
            for ball in self.sim.balls
        ]
        target_ball_index, target_metrics = self._select_target_ball(action_array)
        self._last_target_ball_index = target_ball_index
        command = self._build_command(target_ball_index, target_metrics, action_array)

        self.controller.set_target_position(command["target_position"])
        self.controller.set_target_tilt(command["target_tilt"])
        self.controller.set_target_velocity(command["target_velocity"])
        self.controller.set_body_clearance_reference(
            command["contact_position"],
            active=bool(command["contact_active"]),
        )
        joint_targets = self.controller.compute_joint_targets()
        contact_trace = self.sim.step_with_contact_trace(joint_targets=joint_targets)

        reward, contact_infos = self._reward(action_array, contact_trace, command, previous_contacts)
        failure_reason = self.sim.failure_reason(
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
            z_bounds=self.z_bounds,
            max_ball_speed=self.max_ball_speed,
            terminate_on_ball_ball_contact=self.terminate_on_ball_ball_contact,
        )
        if failure_reason is not None:
            reward += self.failure_penalty

        self.elapsed_steps += 1
        terminated = failure_reason is not None
        truncated = self.max_episode_steps is not None and self.elapsed_steps >= self.max_episode_steps
        self._previous_action = action_array
        self._previous_racket_contacts = [
            self.sim.has_contact(ball.geom_name, "racket_head")
            for ball in self.sim.balls
        ]
        for contact_info in contact_infos:
            ball_index = int(contact_info["ball_index"])
            if bool(contact_info["useful"]):
                self._last_hit_ball_index = ball_index

        info = self._info(
            failure_reason=failure_reason,
            target_ball_index=target_ball_index,
            target_metrics=target_metrics,
            command=command,
            contact_infos=contact_infos,
        )
        return self._get_observation(), float(reward), terminated, truncated, info

    def _reward(
        self,
        action: np.ndarray,
        contact_trace: dict[str, object],
        command: dict[str, object],
        previous_contacts: Sequence[bool],
    ) -> tuple[float, list[dict[str, object]]]:
        reward = 0.01
        reward += self._tracking_term(command)
        action_scale = action / np.maximum(np.abs(self.action_high), 1.0e-6)
        reward -= self.action_penalty_weight * float(np.mean(np.square(action_scale)))

        contact_infos: list[dict[str, object]] = []
        for event in contact_trace["contact_events"]:
            ball_index = int(event["ball_index"])
            if bool(previous_contacts[ball_index]):
                continue
            useful, metrics = self._contact_usefulness(event)
            self.contact_count += 1
            reward += self.contact_bonus
            if float(metrics["outgoing_vz"]) > 0.0:
                reward += self._contact_apex_progress_term(metrics)
                reward += self._contact_apex_recovery_progress_term(ball_index, metrics)
                reward += self._contact_apex_potential_term(ball_index, metrics)
                reward += self._contact_apex_under_min_penalty_term(metrics)
            if useful:
                self.useful_bounce_count += 1
                height_score = float(metrics["height_score"])
                xy_score = float(metrics["xy_score"])
                reward += self.useful_contact_bonus + 2.0 * height_score + 1.5 * xy_score
                if self._last_hit_ball_index is not None and self._last_hit_ball_index != ball_index:
                    reward += self.alternating_bonus
            else:
                self.nonuseful_contact_count += 1
                reward -= 2.0
            contact_info = {
                "ball_index": ball_index,
                "contact_started": True,
                "useful": useful,
                **metrics,
            }
            contact_infos.append(contact_info)
            self._update_contact_apex_memory(ball_index, metrics)
        return reward, contact_infos

    def _tracking_term(self, command: dict[str, object]) -> float:
        if not bool(command["contact_active"]):
            return 0.0
        target_position = np.asarray(command["target_position"], dtype=float)
        racket_position = self.sim.racket_position
        xy_error = float(np.linalg.norm(target_position[:2] - racket_position[:2]))
        z_error = abs(float(target_position[2] - racket_position[2]))
        xy_score = max(1.0 - xy_error / max(self.reachable_radius, 1.0e-6), 0.0)
        z_score = max(1.0 - z_error / 0.18, 0.0)
        return float(self.tracking_reward_weight * xy_score * z_score)

    def _contact_usefulness(self, event: dict[str, object]) -> tuple[bool, dict[str, object]]:
        ball_index = int(event["ball_index"])
        velocity = np.asarray(event["contact_ball_velocity"], dtype=float)
        position = np.asarray(event["contact_ball_position"], dtype=float)
        racket_velocity = np.asarray(event["contact_racket_velocity"], dtype=float)
        apex_height, apex_xy = self._projected_apex(position, velocity)
        slot_xy = self._slot_xy(ball_index)
        xy_error = float(np.linalg.norm(apex_xy - slot_xy))
        height_error = abs(apex_height - self.target_apex_height)
        height_score = max(1.0 - height_error / max(self.max_useful_apex_height - self.min_useful_apex_height, 1.0e-6), 0.0)
        xy_score = max(1.0 - xy_error / max(self.apex_xy_tolerance, 1.0e-6), 0.0)
        useful = (
            velocity[2] > self.min_useful_outgoing_vz
            and racket_velocity[2] >= self.min_useful_racket_vz
            and self.min_useful_apex_height <= apex_height <= self.max_useful_apex_height
            and xy_error <= self.apex_xy_tolerance
        )
        return useful, {
            "projected_apex_height": float(apex_height),
            "projected_apex_xy_error": xy_error,
            "height_score": float(height_score),
            "xy_score": float(xy_score),
            "outgoing_vz": float(velocity[2]),
            "contact_racket_vz": float(racket_velocity[2]),
        }

    def _apex_height_tolerance(self) -> float:
        return max(self.target_apex_height - self.min_useful_apex_height, 1.0e-6)

    def _contact_apex_progress_term(self, metrics: dict[str, object]) -> float:
        if self.contact_apex_progress_reward_weight <= 0.0:
            return 0.0
        apex_height = float(metrics["projected_apex_height"])
        progress = np.clip(apex_height / max(self.target_apex_height, 1.0e-6), 0.0, 1.0)
        return float(self.contact_apex_progress_reward_weight * progress)

    def _contact_apex_recovery_progress_term(self, ball_index: int, metrics: dict[str, object]) -> float:
        if self.contact_apex_recovery_progress_reward_weight <= 0.0:
            return 0.0
        previous_apex = float(self._last_projected_apex_heights[ball_index])
        if not np.isfinite(previous_apex):
            return 0.0
        previous_shortfall = self.target_apex_height - previous_apex
        if previous_shortfall <= 0.0:
            return 0.0
        improvement = float(metrics["projected_apex_height"]) - previous_apex
        if improvement <= 0.0:
            return 0.0
        normalized_improvement = min(improvement / self._apex_height_tolerance(), 2.0)
        return float(self.contact_apex_recovery_progress_reward_weight * normalized_improvement)

    def _contact_apex_potential_score(self, apex_height: float) -> float:
        shortfall = max(self.target_apex_height - apex_height, 0.0)
        normalized_shortfall = min(shortfall / self._apex_height_tolerance(), self.contact_apex_potential_cap)
        return float(max(1.0 - normalized_shortfall / self.contact_apex_potential_cap, 0.0))

    def _contact_apex_potential_term(self, ball_index: int, metrics: dict[str, object]) -> float:
        if self.contact_apex_potential_reward_weight <= 0.0:
            return 0.0
        previous_apex = float(self._last_projected_apex_heights[ball_index])
        if not np.isfinite(previous_apex):
            return 0.0
        current_score = self._contact_apex_potential_score(float(metrics["projected_apex_height"]))
        previous_score = self._contact_apex_potential_score(previous_apex)
        return float(self.contact_apex_potential_reward_weight * (self.contact_apex_potential_gamma * current_score - previous_score))

    def _contact_apex_under_min_penalty_term(self, metrics: dict[str, object]) -> float:
        if self.contact_apex_under_min_penalty_weight <= 0.0:
            return 0.0
        shortfall = max(self.min_useful_apex_height - float(metrics["projected_apex_height"]), 0.0)
        normalized_shortfall = min(shortfall / self._apex_height_tolerance(), 4.0)
        return float(-self.contact_apex_under_min_penalty_weight * normalized_shortfall)

    def _update_contact_apex_memory(self, ball_index: int, metrics: dict[str, object]) -> None:
        apex_height = float(metrics["projected_apex_height"])
        self._last_projected_apex_heights[ball_index] = apex_height
        self._last_contact_apex_shortfalls[ball_index] = max(self.min_useful_apex_height - apex_height, 0.0)

    def _projected_apex(self, position: np.ndarray, velocity: np.ndarray) -> tuple[float, np.ndarray]:
        if velocity[2] <= 0.0:
            return float(position[2] - self._anchor_position[2]), position[:2].copy()
        time_to_apex = float(velocity[2] / GRAVITY)
        apex_z = float(position[2] + velocity[2] * time_to_apex - 0.5 * GRAVITY * time_to_apex * time_to_apex)
        apex_xy = position[:2] + velocity[:2] * time_to_apex
        return float(apex_z - self._anchor_position[2]), apex_xy

    def _slot_xy(self, ball_index: int) -> np.ndarray:
        return self._anchor_position[:2] + self.slot_xy_offsets[ball_index]

    def _select_target_ball(self, action: np.ndarray | None = None) -> tuple[int, dict[str, object]]:
        action_bias = np.zeros(self.sim.ball_count, dtype=float)
        if action is not None:
            action_bias[:] = action[11:13]
        metrics = [self._intercept_metrics(index) for index in range(self.sim.ball_count)]
        scores = []
        for index, item in enumerate(metrics):
            intercept_time = float(item["intercept_time"])
            score = intercept_time if np.isfinite(intercept_time) else 1.5 + 0.2 * index
            if not bool(item["reachable"]):
                score += 0.7
            score -= float(action_bias[index])
            scores.append(score)
        selected = int(np.argmin(scores))
        return selected, metrics[selected]

    def _intercept_metrics(self, ball_index: int, strike_plane_z: float | None = None) -> dict[str, object]:
        position = self.sim.ball_position(ball_index)
        velocity = self.sim.ball_velocity(ball_index)
        plane_z = self._anchor_position[2] + self.strike_plane_offset if strike_plane_z is None else float(strike_plane_z)
        roots = np.roots([0.5 * -GRAVITY, velocity[2], position[2] - plane_z])
        valid_times = []
        for root in roots:
            if abs(float(np.imag(root))) > 1.0e-8:
                continue
            time_value = float(np.real(root))
            if time_value < self.min_intercept_time or time_value > self.max_intercept_time:
                continue
            velocity_z_at_time = float(velocity[2] - GRAVITY * time_value)
            if velocity_z_at_time >= -0.02:
                continue
            valid_times.append(time_value)
        if not valid_times:
            return {
                "ball_index": ball_index,
                "intercept_time": float("inf"),
                "intercept_position": position.copy(),
                "reachable": False,
                "xy_error": float(np.linalg.norm(position[:2] - self._anchor_position[:2])),
                "descending": bool(velocity[2] < -0.02),
            }
        intercept_time = min(valid_times)
        intercept_position = position + velocity * intercept_time + np.array([0.0, 0.0, -0.5 * GRAVITY * intercept_time**2])
        intercept_position[2] = plane_z
        xy_error = float(np.linalg.norm(intercept_position[:2] - self._anchor_position[:2]))
        return {
            "ball_index": ball_index,
            "intercept_time": float(intercept_time),
            "intercept_position": intercept_position,
            "reachable": xy_error <= self.reachable_radius,
            "xy_error": xy_error,
            "descending": True,
        }

    def _build_command(
        self,
        ball_index: int,
        target_metrics: dict[str, object],
        action: np.ndarray,
    ) -> dict[str, object]:
        base_strike_plane_z = self._anchor_position[2] + self.strike_plane_offset + float(action[2])
        base_metrics = self._intercept_metrics(ball_index, strike_plane_z=base_strike_plane_z)
        if np.isfinite(float(base_metrics["intercept_time"])):
            base_intercept_time = max(float(base_metrics["intercept_time"]), self.min_intercept_time)
            base_readiness = 1.0 - np.clip(base_intercept_time / 0.32, 0.0, 1.0)
        else:
            base_readiness = 0.0
        recovery_lift = self._low_apex_recovery_lift(ball_index, float(base_readiness))
        recovery_velocity = self._low_apex_recovery_velocity(ball_index, float(base_readiness))
        strike_plane_z = base_strike_plane_z + recovery_lift
        metrics = self._intercept_metrics(ball_index, strike_plane_z=strike_plane_z)
        contact_active = bool(np.isfinite(float(metrics["intercept_time"])))
        if contact_active:
            contact_position = np.asarray(metrics["intercept_position"], dtype=float).copy()
            contact_position[:2] = self._clip_xy_to_reachable(contact_position[:2])
            contact_position[:2] += action[:2]
            contact_position[:2] = self._clip_xy_to_reachable(contact_position[:2])
            contact_position[2] = strike_plane_z
            target_position = contact_position.copy()
            intercept_time = max(float(metrics["intercept_time"]), self.min_intercept_time)
            readiness = 1.0 - np.clip(intercept_time / 0.32, 0.0, 1.0)
        else:
            ball_position = self.sim.ball_position(ball_index)
            contact_position = self._anchor_position.copy()
            contact_position[:2] = self._clip_xy_to_reachable(ball_position[:2])
            contact_position[2] = self._anchor_position[2] + 0.08
            target_position = contact_position.copy()
            intercept_time = self.max_intercept_time
            readiness = 0.0

        desired_outgoing_velocity = self._desired_outgoing_velocity(ball_index, contact_position, action)
        incoming_velocity = self.sim.ball_velocity(ball_index)
        required_racket_velocity = (desired_outgoing_velocity + self.restitution * incoming_velocity) / (1.0 + self.restitution)
        intercept_velocity = self.intercept_velocity_gain * (target_position - self.sim.racket_position) / max(intercept_time, 1.0e-3)
        intercept_velocity = self._clip_norm(intercept_velocity, self.target_velocity_max)
        target_velocity = intercept_velocity + readiness * required_racket_velocity + action[5:8]
        target_velocity[2] += recovery_velocity
        target_velocity[:2] = self._clip_norm(target_velocity[:2], self.lateral_velocity_max)
        target_velocity = self._clip_norm(target_velocity, self.target_velocity_max)

        correction_xy = self._slot_xy(ball_index) - contact_position[:2]
        base_tilt = np.array([correction_xy[0], -correction_xy[1]], dtype=float) * self.tilt_gain * readiness
        target_tilt = np.clip(base_tilt + action[3:5], -self.tilt_limit, self.tilt_limit)
        return {
            "ball_index": ball_index,
            "target_position": target_position,
            "target_tilt": target_tilt,
            "target_velocity": target_velocity,
            "desired_outgoing_velocity": desired_outgoing_velocity,
            "contact_position": contact_position,
            "contact_active": contact_active,
            "intercept_time": float(intercept_time),
            "readiness": float(readiness),
            "recovery_lift": float(recovery_lift),
            "recovery_velocity": float(recovery_velocity),
            "metrics": metrics,
        }

    def _normalized_last_contact_apex_shortfall(self, ball_index: int) -> float:
        shortfall = float(self._last_contact_apex_shortfalls[ball_index])
        if shortfall <= 0.0:
            return 0.0
        return float(np.clip(shortfall / self._apex_height_tolerance(), 0.0, 2.0))

    def _low_apex_recovery_lift(self, ball_index: int, readiness: float) -> float:
        if self.low_apex_recovery_lift_gain <= 0.0 or self.low_apex_recovery_lift_max <= 0.0:
            return 0.0
        if float(self.sim.ball_velocity(ball_index)[2]) >= -0.02:
            return 0.0
        normalized_shortfall = self._normalized_last_contact_apex_shortfall(ball_index)
        if normalized_shortfall <= 0.0:
            return 0.0
        lift = self.low_apex_recovery_lift_gain * normalized_shortfall * np.clip(readiness, 0.0, 1.0)
        return float(np.clip(lift, 0.0, self.low_apex_recovery_lift_max))

    def _low_apex_recovery_velocity(self, ball_index: int, readiness: float) -> float:
        if self.low_apex_recovery_velocity_gain <= 0.0 or self.low_apex_recovery_velocity_max <= 0.0:
            return 0.0
        if float(self.sim.ball_velocity(ball_index)[2]) >= -0.02:
            return 0.0
        normalized_shortfall = self._normalized_last_contact_apex_shortfall(ball_index)
        if normalized_shortfall <= 0.0:
            return 0.0
        velocity = self.low_apex_recovery_velocity_gain * normalized_shortfall * np.clip(readiness, 0.0, 1.0)
        return float(np.clip(velocity, 0.0, self.low_apex_recovery_velocity_max))

    def _desired_outgoing_velocity(self, ball_index: int, contact_position: np.ndarray, action: np.ndarray) -> np.ndarray:
        target_apex_z = self._anchor_position[2] + self.target_apex_height + float(action[10])
        target_apex_z = max(target_apex_z, float(contact_position[2] + self.min_useful_apex_height))
        desired_vz = np.sqrt(max(2.0 * GRAVITY * (target_apex_z - contact_position[2]), 0.05))
        time_to_apex = max(desired_vz / GRAVITY, 0.08)
        desired_xy = (self._slot_xy(ball_index) - contact_position[:2]) / time_to_apex + action[8:10]
        desired_xy = self._clip_norm(desired_xy, self.lateral_velocity_max)
        return np.array([desired_xy[0], desired_xy[1], desired_vz], dtype=float)

    def _clip_xy_to_reachable(self, xy: Sequence[float]) -> np.ndarray:
        xy_array = np.asarray(xy, dtype=float)
        delta = xy_array - self._anchor_position[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= self.reachable_radius:
            return xy_array.copy()
        return self._anchor_position[:2] + delta * (self.reachable_radius / max(distance, 1.0e-9))

    @staticmethod
    def _clip_norm(vector: Sequence[float], max_norm: float) -> np.ndarray:
        vector_array = np.asarray(vector, dtype=float).copy()
        norm = float(np.linalg.norm(vector_array))
        if max_norm > 0.0 and norm > max_norm:
            return vector_array * (max_norm / norm)
        return vector_array

    def _get_observation(self) -> np.ndarray:
        selected_ball, selected_metrics = self._select_target_ball(None)
        racket_position = self.sim.racket_position
        features: list[float] = []
        features.extend((racket_position - self._anchor_position).tolist())
        features.extend(self.sim.racket_velocity.tolist())
        features.extend(self.sim.racket_face_normal.tolist())
        features.extend((self.controller.target_position - self._anchor_position).tolist())
        features.extend(self.controller.target_tilt.tolist())
        for ball_index in range(self.sim.ball_count):
            position = self.sim.ball_position(ball_index)
            velocity = self.sim.ball_velocity(ball_index)
            metrics = selected_metrics if ball_index == selected_ball else self._intercept_metrics(ball_index)
            intercept_time = float(metrics["intercept_time"])
            time_feature = 1.0 if not np.isfinite(intercept_time) else np.clip(intercept_time / self.max_intercept_time, 0.0, 1.0)
            features.extend((position - racket_position).tolist())
            features.extend(velocity.tolist())
            features.extend((position - self._anchor_position).tolist())
            features.append(float(time_feature))
            features.extend((position[:2] - self._slot_xy(ball_index)).tolist())
            features.append(1.0 if velocity[2] < -0.02 else 0.0)
        for ball_index in range(self.sim.ball_count):
            features.append(1.0 if ball_index == selected_ball else 0.0)
        return np.asarray(features, dtype=np.float32)

    def _info(
        self,
        *,
        failure_reason: str | None = None,
        target_ball_index: int | None = None,
        target_metrics: dict[str, object] | None = None,
        command: dict[str, object] | None = None,
        contact_infos: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        info: dict[str, object] = {
            "elapsed_steps": self.elapsed_steps,
            "contact_count": self.contact_count,
            "useful_bounces": self.useful_bounce_count,
            "successful_bounce_count": self.useful_bounce_count,
            "nonuseful_contact_count": self.nonuseful_contact_count,
            "failure_reason": failure_reason,
            "target_ball_index": self._last_target_ball_index if target_ball_index is None else target_ball_index,
        }
        if target_metrics is not None:
            info["target_intercept_time"] = float(target_metrics["intercept_time"])
            info["target_reachable"] = bool(target_metrics["reachable"])
            info["target_xy_error"] = float(target_metrics["xy_error"])
        if command is not None:
            info["target_readiness"] = float(command["readiness"])
            info["recovery_lift"] = float(command["recovery_lift"])
            info["recovery_velocity"] = float(command["recovery_velocity"])
        if contact_infos:
            info["last_contact_useful"] = bool(contact_infos[-1]["useful"])
            info["last_projected_apex_height"] = float(contact_infos[-1]["projected_apex_height"])
            info["last_projected_apex_xy_error"] = float(contact_infos[-1]["projected_apex_xy_error"])
            info["last_contact_racket_vz"] = float(contact_infos[-1]["contact_racket_vz"])
        info["last_contact_apex_shortfalls"] = self._last_contact_apex_shortfalls.tolist()
        return info

    def training_config(self) -> dict[str, object]:
        return {
            "scene_path": str(self.sim.scene_path),
            "max_episode_steps": self.max_episode_steps,
            "target_apex_height": self.target_apex_height,
            "min_useful_apex_height": self.min_useful_apex_height,
            "max_useful_apex_height": self.max_useful_apex_height,
            "apex_xy_tolerance": self.apex_xy_tolerance,
            "reset_xy_range": self.reset_xy_range,
            "reset_height_jitter": self.reset_height_jitter,
            "reset_base_heights": self.reset_base_heights.tolist(),
            "reset_velocity_xy_range": self.reset_velocity_xy_range,
            "reset_velocity_z_range": self.reset_velocity_z_range,
            "reset_spin_range": self.reset_spin_range,
            "strike_plane_offset": self.strike_plane_offset,
            "reachable_radius": self.reachable_radius,
            "min_intercept_time": self.min_intercept_time,
            "max_intercept_time": self.max_intercept_time,
            "target_velocity_max": self.target_velocity_max,
            "intercept_velocity_gain": self.intercept_velocity_gain,
            "restitution": self.restitution,
            "lateral_velocity_max": self.lateral_velocity_max,
            "tilt_gain": self.tilt_gain,
            "tilt_limit": self.tilt_limit.tolist(),
            "min_useful_outgoing_vz": self.min_useful_outgoing_vz,
            "min_useful_racket_vz": self.min_useful_racket_vz,
            "action_penalty_weight": self.action_penalty_weight,
            "tracking_reward_weight": self.tracking_reward_weight,
            "contact_bonus": self.contact_bonus,
            "useful_contact_bonus": self.useful_contact_bonus,
            "alternating_bonus": self.alternating_bonus,
            "contact_apex_under_min_penalty_weight": self.contact_apex_under_min_penalty_weight,
            "contact_apex_progress_reward_weight": self.contact_apex_progress_reward_weight,
            "contact_apex_recovery_progress_reward_weight": self.contact_apex_recovery_progress_reward_weight,
            "contact_apex_potential_reward_weight": self.contact_apex_potential_reward_weight,
            "contact_apex_potential_gamma": self.contact_apex_potential_gamma,
            "contact_apex_potential_cap": self.contact_apex_potential_cap,
            "low_apex_recovery_lift_gain": self.low_apex_recovery_lift_gain,
            "low_apex_recovery_lift_max": self.low_apex_recovery_lift_max,
            "low_apex_recovery_velocity_gain": self.low_apex_recovery_velocity_gain,
            "low_apex_recovery_velocity_max": self.low_apex_recovery_velocity_max,
            "failure_penalty": self.failure_penalty,
            "slot_xy_offsets": self.slot_xy_offsets.tolist(),
            "x_bounds": list(self.x_bounds),
            "y_bounds": list(self.y_bounds),
            "z_bounds": list(self.z_bounds),
            "max_ball_speed": self.max_ball_speed,
            "terminate_on_ball_ball_contact": self.terminate_on_ball_ball_contact,
            "controller_position_gain": self.controller_position_gain,
            "controller_orientation_gain": self.controller_orientation_gain,
            "controller_max_position_step": self.controller_max_position_step,
            "controller_max_orientation_step": self.controller_max_orientation_step,
            "controller_velocity_gain": self.controller_velocity_gain,
            "controller_velocity_feedback_gain": self.controller_velocity_feedback_gain,
            "controller_max_velocity_step": self.controller_max_velocity_step,
            "action_names": ACTION_NAMES,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "observation_size": self.observation_size,
        }

    def close(self) -> None:
        return None
