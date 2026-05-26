from __future__ import annotations

from typing import Sequence

import numpy as np

from pingpong_rl2.controllers import RacketCartesianController
from pingpong_rl2.defaults import (
    DEFAULT_ACTION_LIMIT,
    DEFAULT_APEX_MATCH_REWARD_WEIGHT,
    DEFAULT_BALL_HEIGHT,
    DEFAULT_BODY_CONTACT_PENALTY,
    DEFAULT_CONTACT_BONUS,
    DEFAULT_FAILURE_PENALTY,
    DEFAULT_FLOOR_PENALTY,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_RESET_BALL_HEIGHT_RANGE,
    DEFAULT_RESET_VELOCITY_XY_RANGE,
    DEFAULT_RESET_VELOCITY_Z_RANGE,
    DEFAULT_RESET_XY_RANGE,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
    DEFAULT_TRACKING_REWARD_WEIGHT,
)
from pingpong_rl2.envs.pingpong_sim import PingPongSim

_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("joint_positions", 7),
    ("joint_velocities", 7),
    ("racket_position", 3),
    ("target_position", 3),
    ("ball_position", 3),
    ("ball_velocity", 3),
    ("ball_relative_position", 3),
)

_OBSERVATION_SLICES: dict[str, slice] = {}
_observation_offset = 0
for component_name, component_size in _OBSERVATION_COMPONENTS:
    _OBSERVATION_SLICES[component_name] = slice(_observation_offset, _observation_offset + component_size)
    _observation_offset += component_size
_OBSERVATION_SIZE = _observation_offset


class PingPongKeepUpEnv:
    def __init__(
        self,
        sim: PingPongSim | None = None,
        action_limit: float = DEFAULT_ACTION_LIMIT,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        success_velocity_threshold: float = DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
        ball_height: float = DEFAULT_BALL_HEIGHT,
        target_ball_height: float = DEFAULT_BALL_HEIGHT,
        height_tolerance: float = 0.10,
        tracking_reward_weight: float = DEFAULT_TRACKING_REWARD_WEIGHT,
        contact_bonus: float = DEFAULT_CONTACT_BONUS,
        apex_match_reward_weight: float = DEFAULT_APEX_MATCH_REWARD_WEIGHT,
        descending_ball_velocity_threshold: float = -0.05,
        strike_zone_xy_radius: float = 0.10,
        strike_zone_height_tolerance: float = 0.16,
        contact_centering_radius: float = 0.04,
        min_upward_racket_velocity_z: float = 0.05,
        floor_penalty: float = DEFAULT_FLOOR_PENALTY,
        robot_body_contact_penalty: float = DEFAULT_BODY_CONTACT_PENALTY,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        reset_ball_height_range: float = DEFAULT_RESET_BALL_HEIGHT_RANGE,
        reset_xy_range: float = DEFAULT_RESET_XY_RANGE,
        reset_velocity_xy_range: float = DEFAULT_RESET_VELOCITY_XY_RANGE,
        reset_velocity_z_range: tuple[float, float] = DEFAULT_RESET_VELOCITY_Z_RANGE,
        target_offset_low: Sequence[float] = (-0.12, -0.12, -0.04),
        target_offset_high: Sequence[float] = (0.12, 0.12, 0.12),
        controller_position_gain: float = 1.6,
        controller_orientation_gain: float = 0.45,
        controller_max_position_step: float = 0.06,
        controller_max_orientation_step: float = 0.12,
    ) -> None:
        self.sim = PingPongSim() if sim is None else sim
        self.action_limit = float(action_limit)
        self.max_episode_steps = int(max_episode_steps)
        self.success_velocity_threshold = float(success_velocity_threshold)
        self.ball_height = float(ball_height)
        self.target_ball_height = float(target_ball_height)
        self.height_tolerance = float(height_tolerance)
        self.tracking_reward_weight = float(tracking_reward_weight)
        self.contact_bonus = float(contact_bonus)
        self.apex_match_reward_weight = float(apex_match_reward_weight)
        self.descending_ball_velocity_threshold = float(descending_ball_velocity_threshold)
        self.strike_zone_xy_radius = float(strike_zone_xy_radius)
        self.strike_zone_height_tolerance = float(strike_zone_height_tolerance)
        self.contact_centering_radius = float(contact_centering_radius)
        self.min_upward_racket_velocity_z = float(min_upward_racket_velocity_z)
        self.floor_penalty = float(floor_penalty)
        self.robot_body_contact_penalty = float(robot_body_contact_penalty)
        self.failure_penalty = float(failure_penalty)
        self.reset_ball_height_range = float(reset_ball_height_range)
        self.reset_xy_range = float(reset_xy_range)
        self.reset_velocity_xy_range = float(reset_velocity_xy_range)
        self.reset_velocity_z_range = (float(reset_velocity_z_range[0]), float(reset_velocity_z_range[1]))
        self.target_offset_low = np.asarray(target_offset_low, dtype=float)
        self.target_offset_high = np.asarray(target_offset_high, dtype=float)
        self.controller_position_gain = float(controller_position_gain)
        self.controller_orientation_gain = float(controller_orientation_gain)
        self.controller_max_position_step = float(controller_max_position_step)
        self.controller_max_orientation_step = float(controller_max_orientation_step)
        if self.action_limit <= 0.0:
            raise ValueError(f"action_limit must be positive, got {self.action_limit}.")
        if self.max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}.")
        if self.height_tolerance <= 0.0:
            raise ValueError(f"height_tolerance must be positive, got {self.height_tolerance}.")
        if self.reset_velocity_z_range[0] > self.reset_velocity_z_range[1]:
            raise ValueError(
                "reset_velocity_z_range must be ordered as (low, high), got "
                f"{self.reset_velocity_z_range}."
            )
        self.controller = RacketCartesianController(
            self.sim,
            position_gain=self.controller_position_gain,
            orientation_gain=self.controller_orientation_gain,
            max_position_step=self.controller_max_position_step,
            max_orientation_step=self.controller_max_orientation_step,
            target_offset_low=self.target_offset_low,
            target_offset_high=self.target_offset_high,
        )
        self.action_low = -np.full(3, self.action_limit, dtype=float)
        self.action_high = np.full(3, self.action_limit, dtype=float)
        self.action_size = 3
        self.observation_size = _OBSERVATION_SIZE
        self._rng = np.random.default_rng()
        self._spawn_ball_height_above_racket = self.ball_height
        self.step_count = 0
        self.contact_count = 0
        self.successful_bounce_count = 0
        self._contact_active_previous_step = False
        self._previous_action = np.zeros(self.action_size, dtype=float)

    @property
    def observation_slices(self) -> dict[str, slice]:
        return dict(_OBSERVATION_SLICES)

    def seed(self, seed: int | None = None) -> int | None:
        self._rng = np.random.default_rng(seed)
        return seed

    def observation(self) -> np.ndarray:
        ball_relative_position = self.sim.ball_position - self.sim.racket_position
        return np.concatenate(
            [
                self.sim.joint_positions,
                self.sim.joint_velocities,
                self.sim.racket_position,
                self.controller.target_position,
                self.sim.ball_position,
                self.sim.ball_velocity,
                ball_relative_position,
            ]
        )

    def reset(
        self,
        ball_height: float | None = None,
        ball_velocity: Sequence[float] | None = None,
        ball_xy_offset: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        spawn_height = self.ball_height if ball_height is None else float(ball_height)
        spawn_height += self._sample_reset_ball_height_offset() if ball_height is None else 0.0
        spawn_velocity = self._sample_reset_velocity() if ball_velocity is None else np.asarray(ball_velocity, dtype=float)
        spawn_xy_offset = self._sample_reset_xy_offset() if ball_xy_offset is None else np.asarray(ball_xy_offset, dtype=float)
        self.sim.reset(ball_height=spawn_height, ball_velocity=spawn_velocity, ball_xy_offset=spawn_xy_offset)
        self.controller.reset()
        self.step_count = 0
        self.contact_count = 0
        self.successful_bounce_count = 0
        self._contact_active_previous_step = False
        self._previous_action[:] = 0.0
        self._spawn_ball_height_above_racket = float(spawn_height)
        info: dict[str, object] = {
            "contact_count": self.contact_count,
            "successful_bounce_count": self.successful_bounce_count,
            "step_count": self.step_count,
            "target_position": self.controller.target_position,
            "ball_height_above_racket": self._ball_height_above_racket(),
            "spawn_ball_height_above_racket": self._spawn_ball_height_above_racket,
        }
        return self.observation(), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (3,):
            raise ValueError(f"EE delta action must have shape (3,), got {action_array.shape}.")
        applied_action = np.clip(action_array, self.action_low, self.action_high)
        self.controller.add_target_offset(applied_action)
        safe_target_position = self._guarded_target_position(self.controller.target_position)
        self.controller.set_target_position(safe_target_position)
        joint_targets = self.controller.compute_joint_targets()
        contact_trace = self.sim.step_with_contact_trace(joint_targets=joint_targets, n_substeps=self.sim.n_substeps)
        self.step_count += 1

        failure_reason = self._failure_reason()
        contact_active = bool(contact_trace["contact_observed"] or self.sim.has_contact("ball_geom", "racket_head"))
        contact_event = contact_active and not self._contact_active_previous_step
        if contact_event:
            self.contact_count += 1
        success_reason = self._success_reason(failure_reason, contact_trace, contact_event)
        if success_reason is not None:
            self.successful_bounce_count += 1

        reward_terms = self._reward_terms(failure_reason, success_reason, contact_event, contact_trace)
        reward = float(sum(reward_terms.values()))
        terminated = failure_reason is not None
        truncated = (not terminated) and self.step_count >= self.max_episode_steps
        episode_success_reason = None
        if truncated and self.successful_bounce_count > 0:
            episode_success_reason = "keepup_time_limit"
        info: dict[str, object] = {
            "failure_reason": failure_reason,
            "success_reason": success_reason,
            "episode_success_reason": episode_success_reason,
            "reward_terms": reward_terms,
            "contact_event_during_step": contact_event,
            "contact_observed_during_step": bool(contact_trace["contact_observed"]),
            "contact_count": self.contact_count,
            "successful_bounce_count": self.successful_bounce_count,
            "step_count": self.step_count,
            "target_position": self.controller.target_position,
            "ball_height_above_racket": self._ball_height_above_racket(),
            "target_ball_height_above_racket": self._target_ball_height_above_racket(),
            "xy_alignment_error": self._xy_alignment_error(),
            "projected_contact_apex_height_above_racket": (
                self._projected_contact_apex_height_above_racket(contact_trace) if contact_event else None
            ),
            "contact_ball_height_above_racket": contact_trace.get("contact_ball_height_above_racket"),
            "contact_xy_alignment_error": contact_trace.get("contact_xy_alignment_error"),
            "racket_velocity_z": float(self.sim.racket_velocity[2]),
            "contact_ball_velocity_z": contact_trace.get("contact_ball_velocity_z"),
            "contact_racket_velocity_z": contact_trace.get("contact_racket_velocity_z"),
            "terminated": terminated,
            "truncated": truncated,
        }
        self._contact_active_previous_step = contact_active
        self._previous_action = applied_action.copy()
        return self.observation(), reward, terminated, truncated, info

    def training_config(self) -> dict[str, object]:
        return {
            "action_limit": self.action_limit,
            "max_episode_steps": self.max_episode_steps,
            "success_velocity_threshold": self.success_velocity_threshold,
            "ball_height": self.ball_height,
            "target_ball_height": self.target_ball_height,
            "height_tolerance": self.height_tolerance,
            "tracking_reward_weight": self.tracking_reward_weight,
            "contact_bonus": self.contact_bonus,
            "apex_match_reward_weight": self.apex_match_reward_weight,
            "descending_ball_velocity_threshold": self.descending_ball_velocity_threshold,
            "strike_zone_xy_radius": self.strike_zone_xy_radius,
            "strike_zone_height_tolerance": self.strike_zone_height_tolerance,
            "contact_centering_radius": self.contact_centering_radius,
            "min_upward_racket_velocity_z": self.min_upward_racket_velocity_z,
            "reset_ball_height_range": self.reset_ball_height_range,
            "reset_xy_range": self.reset_xy_range,
            "reset_velocity_xy_range": self.reset_velocity_xy_range,
            "reset_velocity_z_range": list(self.reset_velocity_z_range),
            "target_offset_low": self.target_offset_low.tolist(),
            "target_offset_high": self.target_offset_high.tolist(),
        }

    def close(self) -> None:
        return None

    def _contact_float(self, contact_trace: dict[str, object] | None, key: str, default: float) -> float:
        if contact_trace is None:
            return float(default)
        value = contact_trace.get(key)
        if value is None:
            return float(default)
        return float(value)

    def _ball_height_above_racket(self) -> float:
        return float(self.sim.ball_position[2] - self.sim.racket_position[2])

    def _xy_alignment_error(self) -> float:
        return float(np.linalg.norm(self.sim.ball_position[:2] - self.sim.racket_position[:2]))

    def _gravity_z(self) -> float:
        return float(self.sim.model.opt.gravity[2])

    def _target_ball_height_above_racket(self) -> float:
        return max(self.target_ball_height, self._spawn_ball_height_above_racket)

    def _failure_z_bounds(self) -> tuple[float, float]:
        dynamic_upper_bound = (
            self.sim.racket_position[2]
            + self._target_ball_height_above_racket()
            + self.height_tolerance
            + max(self.height_tolerance, 0.20)
        )
        return (-0.05, max(2.0, float(dynamic_upper_bound)))

    def _failure_reason(self) -> str | None:
        return self.sim.failure_reason(z_bounds=self._failure_z_bounds())

    def _projected_apex_height_above_racket(self, ball_height_above_racket: float, ball_velocity_z: float) -> float:
        current_height = max(float(ball_height_above_racket), 0.0)
        if ball_velocity_z <= 0.0:
            return current_height
        gravity_magnitude = max(abs(self._gravity_z()), 1.0e-6)
        return current_height + float(ball_velocity_z * ball_velocity_z / (2.0 * gravity_magnitude))

    def _projected_contact_apex_height_above_racket(self, contact_trace: dict[str, object] | None) -> float:
        contact_ball_velocity_z = self._contact_float(
            contact_trace,
            "contact_ball_velocity_z",
            default=float(self.sim.ball_velocity[2]),
        )
        contact_ball_height = self._contact_float(
            contact_trace,
            "contact_ball_height_above_racket",
            default=self._ball_height_above_racket(),
        )
        return self._projected_apex_height_above_racket(contact_ball_height, contact_ball_velocity_z)

    def _apex_match_term(self, contact_trace: dict[str, object] | None) -> float:
        projected_apex = self._projected_contact_apex_height_above_racket(contact_trace)
        height_error = abs(projected_apex - self._target_ball_height_above_racket())
        height_match = max(1.0 - height_error / self.height_tolerance, 0.0)
        return float(self.apex_match_reward_weight * height_match)

    def _preparation_target_height_above_racket(self) -> float:
        return float(np.clip(min(self._target_ball_height_above_racket(), 0.18), 0.12, 0.18))

    def _tracking_term(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        ball_height_above_racket = self._ball_height_above_racket()
        vertical_error = abs(ball_height_above_racket - self._preparation_target_height_above_racket())
        if vertical_error > self.strike_zone_height_tolerance:
            return 0.0
        vertical_score = max(1.0 - vertical_error / self.strike_zone_height_tolerance, 0.0)
        xy_score = max(1.0 - self._xy_alignment_error() / self.strike_zone_xy_radius, 0.0)
        return float(self.tracking_reward_weight * xy_score * vertical_score)

    def _success_reason(
        self,
        failure_reason: str | None,
        contact_trace: dict[str, object],
        contact_event: bool,
    ) -> str | None:
        if failure_reason is not None or not contact_event:
            return None
        contact_ball_velocity_z = self._contact_float(contact_trace, "contact_ball_velocity_z", float(self.sim.ball_velocity[2]))
        contact_racket_velocity_z = self._contact_float(
            contact_trace,
            "contact_racket_velocity_z",
            float(self.sim.racket_velocity[2]),
        )
        if contact_ball_velocity_z <= self.success_velocity_threshold:
            return None
        if contact_racket_velocity_z <= self.min_upward_racket_velocity_z:
            return None
        if self._projected_contact_apex_height_above_racket(contact_trace) + 1.0e-6 < self._target_ball_height_above_racket():
            return None
        return "useful_keepup_bounce"

    def _reward_terms(
        self,
        failure_reason: str | None,
        success_reason: str | None,
        contact_event: bool,
        contact_trace: dict[str, object],
    ) -> dict[str, float]:
        reward_terms = {
            "tracking_term": self._tracking_term(),
            "contact_bonus": 0.0,
            "apex_match_term": 0.0,
            "failure_penalty": 0.0,
        }
        if contact_event and success_reason is not None:
            reward_terms["contact_bonus"] = self.contact_bonus
            reward_terms["apex_match_term"] = self._apex_match_term(contact_trace)
        if failure_reason == "floor_contact":
            reward_terms["failure_penalty"] = self.floor_penalty
        elif failure_reason == "robot_body_contact":
            reward_terms["failure_penalty"] = self.robot_body_contact_penalty
        elif failure_reason is not None:
            reward_terms["failure_penalty"] = self.failure_penalty
        return reward_terms

    def _sample_reset_xy_offset(self) -> np.ndarray:
        if self.reset_xy_range <= 0.0:
            return np.zeros(2, dtype=float)
        return self._rng.uniform(-self.reset_xy_range, self.reset_xy_range, size=2)

    def _sample_reset_ball_height_offset(self) -> float:
        if self.reset_ball_height_range <= 0.0:
            return 0.0
        return float(self._rng.uniform(-self.reset_ball_height_range, self.reset_ball_height_range))

    def _sample_reset_velocity(self) -> np.ndarray:
        velocity = np.zeros(3, dtype=float)
        if self.reset_velocity_xy_range > 0.0:
            velocity[:2] = self._rng.uniform(-self.reset_velocity_xy_range, self.reset_velocity_xy_range, size=2)
        velocity[2] = self._rng.uniform(self.reset_velocity_z_range[0], self.reset_velocity_z_range[1])
        return velocity

    def _controller_anchor_position(self) -> np.ndarray:
        anchor_position = getattr(self.controller, "_anchor_position", None)
        if anchor_position is None:
            return np.asarray(self.sim.racket_position, dtype=float)
        return np.asarray(anchor_position, dtype=float)

    def _pre_contact_readiness(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0
        preparation_height = self._preparation_target_height_above_racket()
        activation_height = float(np.clip(preparation_height + 0.08, 0.16, 0.22))
        ball_height = self._ball_height_above_racket()
        if ball_height >= activation_height:
            return 0.0
        height_score = 1.0 - np.clip(
            (ball_height - preparation_height) / max(activation_height - preparation_height, 1.0e-6),
            0.0,
            1.0,
        )
        xy_score = max(1.0 - self._xy_alignment_error() / self.strike_zone_xy_radius, 0.0)
        return float(np.clip(min(height_score, xy_score), 0.0, 1.0))

    def _pre_contact_upward_ready(self) -> bool:
        return self._pre_contact_readiness() >= 0.95

    def _pre_contact_xy_limit(self) -> float:
        full_xy_limit = min(self.strike_zone_xy_radius, float(np.min(self.target_offset_high[:2])))
        base_xy_limit = min(max(self.contact_centering_radius, 0.4 * full_xy_limit), full_xy_limit)
        readiness = self._pre_contact_readiness()
        return float(base_xy_limit + readiness * (full_xy_limit - base_xy_limit))

    def _body_safe_target_position(self, target_position: Sequence[float]) -> np.ndarray:
        safe_target = np.asarray(target_position, dtype=float).copy()
        keepout_specs = (("link5", 0.12), ("link6", 0.10), ("link7", 0.09), ("hand", 0.08))
        anchor_xy = np.asarray(self.sim.racket_position[:2], dtype=float)
        for body_name, keepout_radius in keepout_specs:
            try:
                body_id = self.sim.model.body(body_name).id
            except Exception:
                continue
            body_position = np.asarray(self.sim.data.xpos[body_id], dtype=float)
            delta_xy = safe_target[:2] - body_position[:2]
            distance_xy = float(np.linalg.norm(delta_xy))
            if distance_xy >= keepout_radius:
                continue
            if distance_xy <= 1.0e-9:
                fallback_direction = safe_target[:2] - anchor_xy
                fallback_norm = float(np.linalg.norm(fallback_direction))
                delta_direction = np.array([1.0, 0.0], dtype=float) if fallback_norm <= 1.0e-9 else fallback_direction / fallback_norm
            else:
                delta_direction = delta_xy / distance_xy
            safe_target[:2] = body_position[:2] + keepout_radius * delta_direction
        return safe_target

    def _guarded_target_position(self, target_position: Sequence[float]) -> np.ndarray:
        safe_target = self._body_safe_target_position(target_position)
        anchor_position = self._controller_anchor_position()
        pre_contact_xy_limit = self._pre_contact_xy_limit()
        safe_target[:2] = anchor_position[:2] + np.clip(
            safe_target[:2] - anchor_position[:2],
            -pre_contact_xy_limit,
            pre_contact_xy_limit,
        )
        if not self._pre_contact_upward_ready():
            safe_target[2] = min(float(safe_target[2]), float(anchor_position[2] + 0.02))
        return safe_target
