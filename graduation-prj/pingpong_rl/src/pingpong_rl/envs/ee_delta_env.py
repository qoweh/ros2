from __future__ import annotations

from typing import Sequence

import numpy as np

from pingpong_rl.controllers import RacketCartesianController
from pingpong_rl.defaults import (
    DEFAULT_ACTION_LIMIT,
    DEFAULT_BALL_HEIGHT,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
)
from pingpong_rl.envs.pingpong_env import PingPongSim


_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("joint_positions", 7),
    ("joint_velocities", 7),
    ("racket_position", 3),
    ("racket_velocity", 3),
    ("target_position", 3),
    ("ball_position", 3),
    ("ball_velocity", 3),
)

_OBSERVATION_SLICES: dict[str, slice] = {}
_observation_offset = 0
for component_name, component_size in _OBSERVATION_COMPONENTS:
    _OBSERVATION_SLICES[component_name] = slice(_observation_offset, _observation_offset + component_size)
    _observation_offset += component_size
_OBSERVATION_SIZE = _observation_offset


class PingPongEEDeltaEnv:
    def __init__(
        self,
        sim: PingPongSim | None = None,
        action_limit: float = DEFAULT_ACTION_LIMIT,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        success_velocity_threshold: float = DEFAULT_SUCCESS_VELOCITY_THRESHOLD,
        ball_height: float = DEFAULT_BALL_HEIGHT,
        contact_bonus: float = 1.0,
        stale_contact_penalty: float = -0.25,
        success_bonus: float = 12.0,
        rally_progress_bonus: float = 6.0,
        target_ball_height: float = 0.30,
        height_tolerance: float = 0.10,
        height_reward_weight: float = 0.25,
        height_overshoot_penalty_weight: float = 1.5,
        useful_contact_velocity_z: float = 0.70,
        target_contact_velocity_z: float = 1.40,
        lift_reward_weight: float = 8.0,
        lift_overshoot_penalty_weight: float = 4.0,
        min_active_racket_velocity_z: float = 0.10,
        target_active_racket_velocity_z: float = 0.35,
        min_active_racket_acceleration_z: float = 1.0,
        target_active_racket_acceleration_z: float = 6.0,
        active_hit_reward_weight: float = 6.0,
        passive_contact_penalty: float = -2.5,
        preparation_reward_weight: float = 2.5,
        downward_motion_penalty_weight: float = 1.5,
        descending_ball_velocity_threshold: float = -0.05,
        strike_zone_xy_radius: float = 0.10,
        strike_zone_height_tolerance: float = 0.16,
        xy_alignment_weight: float = 0.75,
        floor_penalty: float = -8.0,
        failure_penalty: float = -5.0,
        reset_xy_range: float = 0.015,
        reset_velocity_xy_range: float = 0.01,
        reset_velocity_z_range: tuple[float, float] = (-0.02, 0.01),
        target_offset_low: Sequence[float] = (-0.24, -0.24, -0.08),
        target_offset_high: Sequence[float] = (0.24, 0.24, 0.18),
    ) -> None:
        self.sim = PingPongSim() if sim is None else sim
        self.action_limit = float(action_limit)
        self.max_episode_steps = int(max_episode_steps)
        self.success_velocity_threshold = float(success_velocity_threshold)
        self.ball_height = float(ball_height)
        self.contact_bonus = float(contact_bonus)
        self.stale_contact_penalty = float(stale_contact_penalty)
        self.success_bonus = float(success_bonus)
        self.rally_progress_bonus = float(rally_progress_bonus)
        self.target_ball_height = float(target_ball_height)
        self.height_tolerance = float(height_tolerance)
        self.height_reward_weight = float(height_reward_weight)
        self.height_overshoot_penalty_weight = float(height_overshoot_penalty_weight)
        self.useful_contact_velocity_z = float(useful_contact_velocity_z)
        self.target_contact_velocity_z = float(target_contact_velocity_z)
        self.lift_reward_weight = float(lift_reward_weight)
        self.lift_overshoot_penalty_weight = float(lift_overshoot_penalty_weight)
        self.min_active_racket_velocity_z = float(min_active_racket_velocity_z)
        self.target_active_racket_velocity_z = float(target_active_racket_velocity_z)
        self.min_active_racket_acceleration_z = float(min_active_racket_acceleration_z)
        self.target_active_racket_acceleration_z = float(target_active_racket_acceleration_z)
        self.active_hit_reward_weight = float(active_hit_reward_weight)
        self.passive_contact_penalty = float(passive_contact_penalty)
        self.preparation_reward_weight = float(preparation_reward_weight)
        self.downward_motion_penalty_weight = float(downward_motion_penalty_weight)
        self.descending_ball_velocity_threshold = float(descending_ball_velocity_threshold)
        self.strike_zone_xy_radius = float(strike_zone_xy_radius)
        self.strike_zone_height_tolerance = float(strike_zone_height_tolerance)
        self.xy_alignment_weight = float(xy_alignment_weight)
        self.floor_penalty = float(floor_penalty)
        self.failure_penalty = float(failure_penalty)
        self.reset_xy_range = float(reset_xy_range)
        self.reset_velocity_xy_range = float(reset_velocity_xy_range)
        self.reset_velocity_z_range = (float(reset_velocity_z_range[0]), float(reset_velocity_z_range[1]))
        self.target_offset_low = np.asarray(target_offset_low, dtype=float)
        self.target_offset_high = np.asarray(target_offset_high, dtype=float)
        if self.max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}.")
        if self.success_velocity_threshold < 0.0:
            raise ValueError(
                f"success_velocity_threshold must be non-negative, got {self.success_velocity_threshold}."
            )
        if self.height_tolerance <= 0.0:
            raise ValueError(f"height_tolerance must be positive, got {self.height_tolerance}.")
        if self.target_contact_velocity_z < self.useful_contact_velocity_z:
            raise ValueError(
                "target_contact_velocity_z must be greater than or equal to useful_contact_velocity_z. "
                f"Got target_contact_velocity_z={self.target_contact_velocity_z}, "
                f"useful_contact_velocity_z={self.useful_contact_velocity_z}."
            )
        if self.target_active_racket_velocity_z < self.min_active_racket_velocity_z:
            raise ValueError(
                "target_active_racket_velocity_z must be greater than or equal to "
                "min_active_racket_velocity_z."
            )
        if self.target_active_racket_acceleration_z < self.min_active_racket_acceleration_z:
            raise ValueError(
                "target_active_racket_acceleration_z must be greater than or equal to "
                "min_active_racket_acceleration_z."
            )
        if self.strike_zone_xy_radius <= 0.0:
            raise ValueError(f"strike_zone_xy_radius must be positive, got {self.strike_zone_xy_radius}.")
        if self.strike_zone_height_tolerance <= 0.0:
            raise ValueError(
                f"strike_zone_height_tolerance must be positive, got {self.strike_zone_height_tolerance}."
            )
        if self.reset_xy_range < 0.0:
            raise ValueError(f"reset_xy_range must be non-negative, got {self.reset_xy_range}.")
        if self.reset_velocity_xy_range < 0.0:
            raise ValueError(
                f"reset_velocity_xy_range must be non-negative, got {self.reset_velocity_xy_range}."
            )
        if self.reset_velocity_z_range[0] > self.reset_velocity_z_range[1]:
            raise ValueError(f"reset_velocity_z_range min must be <= max, got {self.reset_velocity_z_range}.")
        if self.target_offset_low.shape != (3,):
            raise ValueError(f"target_offset_low must have shape (3,), got {self.target_offset_low.shape}.")
        if self.target_offset_high.shape != (3,):
            raise ValueError(f"target_offset_high must have shape (3,), got {self.target_offset_high.shape}.")
        if np.any(self.target_offset_low > self.target_offset_high):
            raise ValueError(
                f"target_offset_low must be <= target_offset_high, got {self.target_offset_low} "
                f"and {self.target_offset_high}."
            )

        self.step_count = 0
        self.contact_count = 0
        self.successful_bounce_count = 0
        self._contact_active_previous_step = False
        self._rng = np.random.default_rng()
        self.controller = RacketCartesianController(
            self.sim,
            max_position_step=self.action_limit,
            target_offset_low=self.target_offset_low,
            target_offset_high=self.target_offset_high,
        )

    @property
    def target_position(self) -> np.ndarray:
        return self.controller.target_position

    @property
    def observation_size(self) -> int:
        return _OBSERVATION_SIZE

    @property
    def observation_slices(self) -> dict[str, slice]:
        return _OBSERVATION_SLICES.copy()

    def observation_dict(self) -> dict[str, np.ndarray]:
        return {
            "joint_positions": self.sim.joint_positions,
            "joint_velocities": self.sim.joint_velocities,
            "racket_position": self.sim.racket_position,
            "racket_velocity": self._racket_velocity(),
            "target_position": self.controller.target_position,
            "ball_position": self.sim.ball_position,
            "ball_velocity": self.sim.ball_velocity,
        }

    def observation(self) -> np.ndarray:
        observation_dict = self.observation_dict()
        return np.concatenate([observation_dict[name] for name, _ in _OBSERVATION_COMPONENTS])

    @classmethod
    def unflatten_observation(cls, observation: Sequence[float]) -> dict[str, np.ndarray]:
        observation_array = np.asarray(observation, dtype=float)
        if observation_array.shape != (_OBSERVATION_SIZE,):
            raise ValueError(f"Flat observation must have shape ({_OBSERVATION_SIZE},), got {observation_array.shape}.")

        return {
            name: observation_array[component_slice].copy()
            for name, component_slice in _OBSERVATION_SLICES.items()
        }

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(
        self,
        ball_height: float | None = None,
        ball_velocity: Sequence[float] | None = None,
        ball_xy_offset: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        spawn_height = self.ball_height if ball_height is None else float(ball_height)
        spawn_velocity = self._sample_reset_velocity() if ball_velocity is None else np.asarray(ball_velocity, dtype=float)
        spawn_xy_offset = self._sample_reset_xy_offset() if ball_xy_offset is None else np.asarray(ball_xy_offset, dtype=float)
        if spawn_velocity.shape != (3,):
            raise ValueError(f"ball_velocity must have shape (3,), got {spawn_velocity.shape}.")
        if spawn_xy_offset.shape != (2,):
            raise ValueError(f"ball_xy_offset must have shape (2,), got {spawn_xy_offset.shape}.")
        try:
            self.sim.reset(ball_height=spawn_height, ball_velocity=spawn_velocity, ball_xy_offset=spawn_xy_offset)
        except TypeError:
            self.sim.reset(ball_height=spawn_height, ball_velocity=spawn_velocity)
        self.controller.reset()
        self.step_count = 0
        self.contact_count = 0
        self.successful_bounce_count = 0
        self._contact_active_previous_step = False
        reward_terms = self._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=False,
            contact_active=False,
        )
        reward_logging = self._reward_logging(reward_terms)
        info: dict[str, object] = {
            "failure_reason": None,
            "success_reason": None,
            "episode_success_reason": None,
            "step_count": self.step_count,
            "episode_steps": self.step_count,
            "contact_count": self.contact_count,
            "successful_bounce_count": self.successful_bounce_count,
            "target_position": self.controller.target_position,
            "time_limit_reached": False,
            "terminated": False,
            "truncated": False,
            "contact_event_during_step": False,
            "racket_velocity_x": float(self._racket_velocity()[0]),
            "racket_velocity_y": float(self._racket_velocity()[1]),
            "racket_velocity_z": float(self._racket_velocity()[2]),
            "racket_speed_norm": float(np.linalg.norm(self._racket_velocity())),
            "ball_height_above_racket": self._ball_height_above_racket(),
            "reward_height_target_term": float(self._height_target_term()),
            "reward_lift_term": 0.0,
            "reward_active_hit_term": 0.0,
            "reward_terms": reward_terms,
            **reward_logging,
        }
        return self.observation(), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (3,):
            raise ValueError(f"EE delta action must have shape (3,), got {action_array.shape}.")

        applied_action = np.clip(action_array, -self.action_limit, self.action_limit)
        self.controller.add_target_offset(applied_action)
        joint_targets = self.controller.compute_joint_targets()
        if hasattr(self.sim, "step_with_contact_trace"):
            contact_trace = self.sim.step_with_contact_trace(
                joint_targets=joint_targets,
                n_substeps=self.sim.n_substeps,
            )
        else:
            self.sim.step(joint_targets=joint_targets, n_substeps=self.sim.n_substeps)
            contact_trace = {
                "contact_observed": self.sim.has_contact("ball_geom", "racket_head"),
                "contact_substep": None,
                "contact_ball_velocity_x": None,
                "contact_ball_velocity_y": None,
                "contact_ball_velocity_z": None,
                "contact_ball_speed_norm": None,
                "contact_racket_velocity_x": None,
                "contact_racket_velocity_y": None,
                "contact_racket_velocity_z": None,
                "contact_racket_speed_norm": None,
                "contact_racket_acceleration_x": None,
                "contact_racket_acceleration_y": None,
                "contact_racket_acceleration_z": None,
                "contact_racket_acceleration_norm": None,
            }
        self.step_count += 1

        ball_velocity = self.sim.ball_velocity
        racket_velocity = self._racket_velocity()
        failure_reason = self.sim.failure_reason()
        contact_observed = bool(contact_trace["contact_observed"])
        contact_active = bool(contact_observed or self.sim.has_contact("ball_geom", "racket_head"))
        contact_event = contact_active and not self._contact_active_previous_step
        success_reason = self._success_reason(failure_reason, contact_trace, contact_event)
        if contact_event:
            self.contact_count += 1
        if success_reason is not None:
            self.successful_bounce_count += 1

        reward_terms = self._reward_terms(
            failure_reason,
            success_reason,
            contact_event,
            contact_active,
            contact_trace,
        )
        reward_logging = self._reward_logging(reward_terms)
        reward = float(sum(reward_terms.values()))
        terminated = failure_reason is not None
        truncated = (not terminated) and self.step_count >= self.max_episode_steps
        episode_success_reason = None
        if truncated and self.successful_bounce_count > 0:
            episode_success_reason = "keepup_time_limit"
        info: dict[str, object] = {
            "applied_action": applied_action.copy(),
            "target_position": self.controller.target_position,
            "failure_reason": failure_reason,
            "success_reason": success_reason,
            "episode_success_reason": episode_success_reason,
            "reward_terms": reward_terms,
            "racket_contact": self.sim.has_contact("ball_geom", "racket_head"),
            "ball_velocity_x": float(ball_velocity[0]),
            "ball_velocity_y": float(ball_velocity[1]),
            "ball_velocity_z": float(ball_velocity[2]),
            "ball_vertical_velocity": float(ball_velocity[2]),
            "ball_speed_norm": float(np.linalg.norm(ball_velocity)),
            "racket_velocity_x": float(racket_velocity[0]),
            "racket_velocity_y": float(racket_velocity[1]),
            "racket_velocity_z": float(racket_velocity[2]),
            "racket_speed_norm": float(np.linalg.norm(racket_velocity)),
            "ball_height_above_racket": self._ball_height_above_racket(),
            "contact_observed_during_step": bool(contact_trace["contact_observed"]),
            "contact_event_during_step": contact_event,
            "contact_substep": contact_trace.get("contact_substep"),
            "contact_ball_velocity_x": contact_trace.get("contact_ball_velocity_x"),
            "contact_ball_velocity_y": contact_trace.get("contact_ball_velocity_y"),
            "contact_ball_velocity_z": contact_trace.get("contact_ball_velocity_z"),
            "contact_ball_speed_norm": contact_trace.get("contact_ball_speed_norm"),
            "contact_racket_velocity_x": contact_trace.get("contact_racket_velocity_x"),
            "contact_racket_velocity_y": contact_trace.get("contact_racket_velocity_y"),
            "contact_racket_velocity_z": contact_trace.get("contact_racket_velocity_z"),
            "contact_racket_speed_norm": contact_trace.get("contact_racket_speed_norm"),
            "contact_racket_acceleration_x": contact_trace.get("contact_racket_acceleration_x"),
            "contact_racket_acceleration_y": contact_trace.get("contact_racket_acceleration_y"),
            "contact_racket_acceleration_z": contact_trace.get("contact_racket_acceleration_z"),
            "contact_racket_acceleration_norm": contact_trace.get("contact_racket_acceleration_norm"),
            "active_hit_score": float(self._active_hit_score(contact_trace)),
            "reward_height_target_term": float(self._height_target_term()),
            "reward_lift_term": float(self._lift_term(contact_event, contact_trace)),
            "reward_active_hit_term": float(self._active_hit_term(contact_event, contact_trace)),
            "step_count": self.step_count,
            "episode_steps": self.step_count,
            "contact_count": self.contact_count,
            "successful_bounce_count": self.successful_bounce_count,
            "time_limit_reached": truncated,
            "terminated": terminated,
            "truncated": truncated,
            **reward_logging,
        }
        self._contact_active_previous_step = contact_active
        return self.observation(), reward, terminated, truncated, info

    def _success_reason(
        self,
        failure_reason: str | None,
        contact_trace: dict[str, object],
        contact_event: bool,
    ) -> str | None:
        if failure_reason is not None:
            return None

        if not contact_event:
            return None

        if self._active_hit_score(contact_trace) <= 0.0:
            return None

        contact_ball_velocity_z = contact_trace.get("contact_ball_velocity_z")
        if contact_ball_velocity_z is None:
            if not self.sim.has_contact("ball_geom", "racket_head"):
                return None
            contact_ball_velocity_z = float(self.sim.ball_velocity[2])

        if float(contact_ball_velocity_z) <= self.success_velocity_threshold:
            return None
        return "upward_racket_bounce"

    def _reward_terms(
        self,
        failure_reason: str | None,
        success_reason: str | None,
        contact_event: bool,
        contact_active: bool,
        contact_trace: dict[str, object] | None = None,
    ) -> dict[str, float]:
        xy_alignment_error = float(np.linalg.norm(self.sim.ball_position[:2] - self.sim.racket_position[:2]))
        height_target_term = self._height_target_term()
        lift_term = self._lift_term(contact_event, contact_trace)
        active_hit_term = self._active_hit_term(contact_event, contact_trace)
        active_hit_score = self._active_hit_score(contact_trace)
        reward_terms: dict[str, float] = {
            "contact_bonus": 0.0,
            "height_term": height_target_term + lift_term,
            "distance_term": -self.xy_alignment_weight * xy_alignment_error,
            "active_hit_term": active_hit_term,
            "success_bonus": 0.0,
            "failure_penalty": 0.0,
        }
        if contact_event and active_hit_score > 0.0:
            reward_terms["contact_bonus"] = self.contact_bonus
        elif contact_active:
            reward_terms["contact_bonus"] = self.stale_contact_penalty
        if success_reason is not None:
            reward_terms["success_bonus"] = self.success_bonus + self.rally_progress_bonus * max(
                self.successful_bounce_count - 1,
                0,
            )
        if failure_reason == "floor_contact":
            reward_terms["failure_penalty"] = self.floor_penalty
        elif failure_reason is not None:
            reward_terms["failure_penalty"] = self.failure_penalty
        return reward_terms

    def _ball_height_above_racket(self) -> float:
        return float(self.sim.ball_position[2] - self.sim.racket_position[2])

    def _height_target_term(self) -> float:
        ball_height_above_racket = self._ball_height_above_racket()
        height_error = abs(ball_height_above_racket - self.target_ball_height)
        height_match = max(1.0 - height_error / self.height_tolerance, 0.0)
        overshoot_penalty = self.height_overshoot_penalty_weight * max(
            ball_height_above_racket - (self.target_ball_height + self.height_tolerance),
            0.0,
        )
        return self.height_reward_weight * height_match - overshoot_penalty

    def _sample_reset_xy_offset(self) -> np.ndarray:
        if self.reset_xy_range <= 0.0:
            return np.zeros(2, dtype=float)
        return self._rng.uniform(-self.reset_xy_range, self.reset_xy_range, size=2)

    def _sample_reset_velocity(self) -> np.ndarray:
        velocity = np.zeros(3, dtype=float)
        if self.reset_velocity_xy_range > 0.0:
            velocity[:2] = self._rng.uniform(-self.reset_velocity_xy_range, self.reset_velocity_xy_range, size=2)
        velocity[2] = self._rng.uniform(self.reset_velocity_z_range[0], self.reset_velocity_z_range[1])
        return velocity

    def _racket_velocity(self) -> np.ndarray:
        if not hasattr(self.sim, "racket_velocity"):
            return np.zeros(3, dtype=float)
        return np.asarray(self.sim.racket_velocity, dtype=float)

    @staticmethod
    def _contact_float(contact_trace: dict[str, object] | None, key: str, default: float = 0.0) -> float:
        if contact_trace is None:
            return default
        value = contact_trace.get(key)
        return default if value is None else float(value)

    @staticmethod
    def _range_score(value: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0 if value >= lower else 0.0
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    def _active_hit_score(self, contact_trace: dict[str, object] | None) -> float:
        racket_velocity_z = self._contact_float(contact_trace, "contact_racket_velocity_z")
        racket_acceleration_z = self._contact_float(contact_trace, "contact_racket_acceleration_z")
        velocity_score = self._range_score(
            racket_velocity_z,
            self.min_active_racket_velocity_z,
            self.target_active_racket_velocity_z,
        )
        if velocity_score <= 0.0:
            return 0.0

        acceleration_score = self._range_score(
            racket_acceleration_z,
            self.min_active_racket_acceleration_z,
            self.target_active_racket_acceleration_z,
        )
        return velocity_score * (0.7 + 0.3 * acceleration_score)

    def _active_hit_term(self, contact_event: bool, contact_trace: dict[str, object] | None) -> float:
        preparation_term = self._pre_contact_upward_term(contact_event)
        if not contact_event:
            return preparation_term
        active_hit_score = self._active_hit_score(contact_trace)
        if active_hit_score <= 0.0:
            return preparation_term + self.passive_contact_penalty
        return preparation_term + self.active_hit_reward_weight * active_hit_score

    def _pre_contact_upward_term(self, contact_event: bool) -> float:
        if contact_event:
            return 0.0

        strike_zone_score = self._strike_zone_score()
        if strike_zone_score <= 0.0:
            return 0.0

        racket_velocity_z = float(self._racket_velocity()[2])
        upward_motion_score = self._range_score(
            racket_velocity_z,
            0.0,
            self.target_active_racket_velocity_z,
        )
        downward_motion_score = self._range_score(
            -racket_velocity_z,
            0.0,
            self.target_active_racket_velocity_z,
        )
        return (
            self.preparation_reward_weight * strike_zone_score * upward_motion_score
            - self.downward_motion_penalty_weight * strike_zone_score * downward_motion_score
        )

    def _strike_zone_score(self) -> float:
        if float(self.sim.ball_velocity[2]) >= self.descending_ball_velocity_threshold:
            return 0.0

        ball_height_above_racket = self._ball_height_above_racket()
        if ball_height_above_racket < 0.0:
            return 0.0

        xy_alignment_error = float(np.linalg.norm(self.sim.ball_position[:2] - self.sim.racket_position[:2]))
        xy_score = max(1.0 - xy_alignment_error / self.strike_zone_xy_radius, 0.0)
        height_score = max(
            1.0 - abs(ball_height_above_racket - self.target_ball_height) / self.strike_zone_height_tolerance,
            0.0,
        )
        descent_speed_score = self._range_score(
            -float(self.sim.ball_velocity[2]),
            abs(self.descending_ball_velocity_threshold),
            self.useful_contact_velocity_z,
        )
        return xy_score * height_score * descent_speed_score

    def _lift_term(self, contact_event: bool, contact_trace: dict[str, object] | None) -> float:
        if not contact_event:
            return 0.0

        active_hit_score = self._active_hit_score(contact_trace)
        if active_hit_score <= 0.0:
            return 0.0

        contact_ball_velocity_z: float | None = None
        if contact_trace is not None:
            raw_contact_velocity_z = contact_trace.get("contact_ball_velocity_z")
            if raw_contact_velocity_z is not None:
                contact_ball_velocity_z = float(raw_contact_velocity_z)
        if contact_ball_velocity_z is None:
            contact_ball_velocity_z = float(self.sim.ball_velocity[2])

        if contact_ball_velocity_z <= 0.0:
            return -active_hit_score * self.lift_reward_weight
        if contact_ball_velocity_z < self.useful_contact_velocity_z:
            velocity_shortfall = self.useful_contact_velocity_z - contact_ball_velocity_z
            return -active_hit_score * self.lift_reward_weight * velocity_shortfall / self.useful_contact_velocity_z
        if contact_ball_velocity_z <= self.target_contact_velocity_z:
            usable_velocity_range = max(self.target_contact_velocity_z - self.useful_contact_velocity_z, 1.0e-6)
            return (
                active_hit_score
                * self.lift_reward_weight
                * (contact_ball_velocity_z - self.useful_contact_velocity_z)
                / usable_velocity_range
            )

        velocity_overshoot = contact_ball_velocity_z - self.target_contact_velocity_z
        return active_hit_score * max(
            self.lift_reward_weight - self.lift_overshoot_penalty_weight * velocity_overshoot,
            -self.lift_reward_weight,
        )

    def _reward_logging(self, reward_terms: dict[str, float]) -> dict[str, float]:
        return {
            "reward_total": float(sum(reward_terms.values())),
            "reward_height": float(reward_terms["height_term"]),
            "reward_distance": float(reward_terms["distance_term"]),
            "reward_contact": float(reward_terms["contact_bonus"]),
            "reward_active_hit": float(reward_terms["active_hit_term"]),
            "reward_success": float(reward_terms["success_bonus"]),
            "reward_failure": float(reward_terms["failure_penalty"]),
        }
