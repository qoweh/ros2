from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from pingpong_rl3.defaults import DEFAULT_CONTROL_DT
from pingpong_rl3.utils.paths import SCENE_XML_PATH, resolve_input_path


@dataclass(frozen=True)
class BallHandle:
    index: int
    body_name: str
    joint_name: str
    geom_name: str
    body_id: int
    joint_id: int
    geom_id: int
    qpos_adr: int
    dof_adr: int


class TwoBallPingPongSim:
    ball_count = 2

    def __init__(self, scene_path: Path | str | None = None, control_dt: float = DEFAULT_CONTROL_DT) -> None:
        scene_file = resolve_input_path(Path(scene_path)) if scene_path is not None else SCENE_XML_PATH
        self.scene_path = scene_file.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.n_substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self.racket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "racket")
        self.racket_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
        self.racket_head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "racket_head")
        if self.racket_body_id < 0 or self.racket_site_id < 0 or self.racket_head_geom_id < 0:
            raise ValueError("Scene is missing racket, racket_center, or racket_head.")

        self.balls = tuple(self._resolve_ball(index) for index in range(self.ball_count))
        self.ball_geom_names = tuple(ball.geom_name for ball in self.balls)
        self._home_ctrl = self.model.key_ctrl[0].copy()
        self._home_joint_targets = self._home_ctrl[:7].copy()
        self._racket_jacobian = np.zeros((3, self.model.nv), dtype=float)
        self.reset()

    def _resolve_ball(self, index: int) -> BallHandle:
        body_name = f"ball_{index}"
        joint_name = f"ball_{index}_joint"
        geom_name = f"ball_{index}_geom"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if body_id < 0 or joint_id < 0 or geom_id < 0:
            raise ValueError(f"Scene is missing {body_name}, {joint_name}, or {geom_name}.")
        return BallHandle(
            index=index,
            body_name=body_name,
            joint_name=joint_name,
            geom_name=geom_name,
            body_id=body_id,
            joint_id=joint_id,
            geom_id=geom_id,
            qpos_adr=int(self.model.jnt_qposadr[joint_id]),
            dof_adr=int(self.model.jnt_dofadr[joint_id]),
        )

    @property
    def home_joint_targets(self) -> np.ndarray:
        return self._home_joint_targets.copy()

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[:7].copy()

    @property
    def ball_positions(self) -> np.ndarray:
        return np.vstack([self.ball_position(index) for index in range(self.ball_count)])

    @property
    def ball_velocities(self) -> np.ndarray:
        return np.vstack([self.ball_velocity(index) for index in range(self.ball_count)])

    @property
    def racket_position(self) -> np.ndarray:
        return self.data.site_xpos[self.racket_site_id].copy()

    @property
    def racket_velocity(self) -> np.ndarray:
        mujoco.mj_jacSite(self.model, self.data, self._racket_jacobian, None, self.racket_site_id)
        return self._racket_jacobian @ self.data.qvel

    @property
    def racket_face_normal(self) -> np.ndarray:
        racket_xmat = np.asarray(self.data.geom_xmat[self.racket_head_geom_id], dtype=float).reshape(3, 3)
        return racket_xmat[:, 2].copy()

    def ball_position(self, index: int) -> np.ndarray:
        return self.data.xpos[self.balls[index].body_id].copy()

    def ball_velocity(self, index: int) -> np.ndarray:
        ball = self.balls[index]
        return self.data.qvel[ball.dof_adr:ball.dof_adr + 3].copy()

    def ball_angular_velocity(self, index: int) -> np.ndarray:
        ball = self.balls[index]
        return self.data.qvel[ball.dof_adr + 3:ball.dof_adr + 6].copy()

    def reset(
        self,
        ball_positions: Sequence[Sequence[float]] | None = None,
        ball_velocities: Sequence[Sequence[float]] | None = None,
        ball_angular_velocities: Sequence[Sequence[float]] | None = None,
    ) -> mujoco.MjData:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self._home_ctrl
        mujoco.mj_forward(self.model, self.data)

        if ball_positions is None:
            anchor = self.racket_position.copy()
            ball_positions = (
                anchor + np.array([-0.035, -0.025, 0.42], dtype=float),
                anchor + np.array([0.035, 0.025, 0.68], dtype=float),
            )
        if ball_velocities is None:
            ball_velocities = ((0.0, 0.0, -0.10), (0.0, 0.0, -0.20))
        if ball_angular_velocities is None:
            ball_angular_velocities = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        if len(ball_positions) != self.ball_count:
            raise ValueError(f"Expected {self.ball_count} ball positions, got {len(ball_positions)}.")
        if len(ball_velocities) != self.ball_count:
            raise ValueError(f"Expected {self.ball_count} ball velocities, got {len(ball_velocities)}.")
        if len(ball_angular_velocities) != self.ball_count:
            raise ValueError(
                f"Expected {self.ball_count} ball angular velocities, got {len(ball_angular_velocities)}."
            )

        for index in range(self.ball_count):
            self.spawn_ball(
                index,
                ball_positions[index],
                ball_velocities[index],
                angular_velocity=ball_angular_velocities[index],
            )
        return self.data

    def spawn_ball(
        self,
        index: int,
        position: Sequence[float],
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
        angular_velocity: Sequence[float] | None = None,
    ) -> np.ndarray:
        position_array = np.asarray(position, dtype=float)
        velocity_array = np.asarray(velocity, dtype=float)
        angular_velocity_array = (
            np.zeros(3, dtype=float)
            if angular_velocity is None
            else np.asarray(angular_velocity, dtype=float)
        )
        if position_array.shape != (3,):
            raise ValueError(f"Ball position must have shape (3,), got {position_array.shape}.")
        if velocity_array.shape != (3,):
            raise ValueError(f"Ball velocity must have shape (3,), got {velocity_array.shape}.")
        if angular_velocity_array.shape != (3,):
            raise ValueError(f"Ball angular velocity must have shape (3,), got {angular_velocity_array.shape}.")

        ball = self.balls[index]
        self.data.qpos[ball.qpos_adr:ball.qpos_adr + 3] = position_array
        self.data.qpos[ball.qpos_adr + 3:ball.qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[ball.dof_adr:ball.dof_adr + 3] = velocity_array
        self.data.qvel[ball.dof_adr + 3:ball.dof_adr + 6] = angular_velocity_array
        mujoco.mj_forward(self.model, self.data)
        return self.ball_position(index)

    def set_arm_joint_targets(self, joint_targets: Sequence[float], gripper_target: float | None = None) -> np.ndarray:
        joint_targets_array = np.asarray(joint_targets, dtype=float)
        if joint_targets_array.shape != (7,):
            raise ValueError(f"Arm targets must have shape (7,), got {joint_targets_array.shape}.")
        self.data.ctrl[:7] = joint_targets_array
        if gripper_target is not None:
            self.data.ctrl[7] = gripper_target
        return self.data.ctrl[:8].copy()

    def contact_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = self.model.geom(int(contact.geom1)).name
            geom2 = self.model.geom(int(contact.geom2)).name
            pairs.append(tuple(sorted((geom1, geom2))))
        return pairs

    def has_contact(self, geom_a: str, geom_b: str) -> bool:
        target = tuple(sorted((geom_a, geom_b)))
        return target in self.contact_pairs()

    def ball_robot_body_contact(self) -> str | None:
        excluded_bodies = {"racket", *(ball.body_name for ball in self.balls)}
        excluded_geoms = {"floor", "racket_head", *self.ball_geom_names}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = self.model.geom(int(contact.geom1))
            geom2 = self.model.geom(int(contact.geom2))
            geom1_name = geom1.name or ""
            geom2_name = geom2.name or ""
            if geom1_name not in self.ball_geom_names and geom2_name not in self.ball_geom_names:
                continue
            other_geom = geom2 if geom1_name in self.ball_geom_names else geom1
            other_geom_name = other_geom.name or ""
            other_body_name = self.model.body(int(other_geom.bodyid)).name
            if other_geom_name in excluded_geoms or other_body_name in excluded_bodies:
                continue
            return other_body_name
        return None

    def state_is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def failure_reason(
        self,
        x_bounds: tuple[float, float] = (0.0, 1.35),
        y_bounds: tuple[float, float] = (-0.65, 0.65),
        z_bounds: tuple[float, float] = (-0.05, 2.1),
        max_ball_speed: float = 8.0,
        terminate_on_ball_ball_contact: bool = True,
    ) -> str | None:
        if not self.state_is_finite():
            return "nonfinite_state"
        if terminate_on_ball_ball_contact and self.has_contact("ball_0_geom", "ball_1_geom"):
            return "ball_ball_contact"
        robot_body = self.ball_robot_body_contact()
        if robot_body is not None:
            return f"robot_body_contact:{robot_body}"

        for ball in self.balls:
            if self.has_contact(ball.geom_name, "floor"):
                return f"{ball.body_name}_floor_contact"
            position = self.ball_position(ball.index)
            velocity = self.ball_velocity(ball.index)
            within_x = x_bounds[0] <= position[0] <= x_bounds[1]
            within_y = y_bounds[0] <= position[1] <= y_bounds[1]
            within_z = z_bounds[0] <= position[2] <= z_bounds[1]
            if not (within_x and within_y and within_z):
                return f"{ball.body_name}_out_of_bounds"
            if np.linalg.norm(velocity) > max_ball_speed:
                return f"{ball.body_name}_speed_limit"
        return None

    def step(
        self,
        joint_targets: Sequence[float] | None = None,
        gripper_target: float | None = None,
        n_substeps: int | None = None,
    ) -> mujoco.MjData:
        if joint_targets is not None:
            self.set_arm_joint_targets(joint_targets, gripper_target)
        step_count = self.n_substeps if n_substeps is None else max(1, int(n_substeps))
        for _ in range(step_count):
            mujoco.mj_step(self.model, self.data)
        return self.data

    def _matching_racket_contact(self, ball_index: int) -> dict[str, object] | None:
        ball = self.balls[ball_index]
        target = tuple(sorted((ball.geom_name, "racket_head")))
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1_name = self.model.geom(int(contact.geom1)).name or ""
            geom2_name = self.model.geom(int(contact.geom2)).name or ""
            if tuple(sorted((geom1_name, geom2_name))) != target:
                continue
            return {
                "contact_index": index,
                "geom1_name": geom1_name,
                "geom2_name": geom2_name,
                "contact_position": np.asarray(contact.pos[:3], dtype=float).copy(),
            }
        return None

    def step_with_contact_trace(
        self,
        joint_targets: Sequence[float] | None = None,
        gripper_target: float | None = None,
        n_substeps: int | None = None,
    ) -> dict[str, object]:
        if joint_targets is not None:
            self.set_arm_joint_targets(joint_targets, gripper_target)

        step_count = self.n_substeps if n_substeps is None else max(1, int(n_substeps))
        previous_ball_positions = self.ball_positions
        previous_ball_velocities = self.ball_velocities
        previous_racket_velocity = self.racket_velocity
        contact_events: list[dict[str, object]] = []
        observed_balls: set[int] = set()

        for substep_index in range(1, step_count + 1):
            mujoco.mj_step(self.model, self.data)
            racket_velocity = self.racket_velocity
            for ball in self.balls:
                if ball.index in observed_balls:
                    continue
                matching_contact = self._matching_racket_contact(ball.index)
                if matching_contact is None:
                    continue
                ball_position = self.ball_position(ball.index)
                ball_velocity = self.ball_velocity(ball.index)
                racket_position = self.racket_position
                contact_events.append(
                    {
                        "ball_index": ball.index,
                        "ball_name": ball.body_name,
                        "contact_substep": substep_index,
                        "contact_position": matching_contact["contact_position"],
                        "pre_contact_ball_position": previous_ball_positions[ball.index].copy(),
                        "pre_contact_ball_velocity": previous_ball_velocities[ball.index].copy(),
                        "pre_contact_racket_velocity": previous_racket_velocity.copy(),
                        "contact_ball_position": ball_position,
                        "contact_ball_velocity": ball_velocity,
                        "contact_racket_velocity": racket_velocity.copy(),
                        "contact_ball_height_above_racket": float(ball_position[2] - racket_position[2]),
                        "contact_xy_alignment_error": float(np.linalg.norm(ball_position[:2] - racket_position[:2])),
                    }
                )
                observed_balls.add(ball.index)

            previous_ball_positions = self.ball_positions
            previous_ball_velocities = self.ball_velocities
            previous_racket_velocity = racket_velocity

        return {
            "contact_observed": bool(contact_events),
            "contact_events": contact_events,
        }
