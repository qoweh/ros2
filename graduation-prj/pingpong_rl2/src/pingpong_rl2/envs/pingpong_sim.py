from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from pingpong_rl2.defaults import DEFAULT_BALL_HEIGHT, DEFAULT_CONTROL_DT
from pingpong_rl2.utils.paths import SCENE_XML_PATH


class PingPongSim:
    def __init__(self, scene_path: Path | str | None = None, control_dt: float = DEFAULT_CONTROL_DT) -> None:
        scene_file = Path(scene_path) if scene_path is not None else SCENE_XML_PATH
        self.scene_path = scene_file.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.n_substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.racket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "racket")
        self.racket_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
        self.racket_head_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "racket_head")

        if (
            self.ball_joint_id < 0
            or self.ball_body_id < 0
            or self.racket_body_id < 0
            or self.racket_site_id < 0
            or self.racket_head_geom_id < 0
        ):
            raise ValueError(
                "Scene is missing one of the required objects: ball_joint, ball, racket, racket_center, racket_head."
            )

        self._ball_qpos_adr = self.model.jnt_qposadr[self.ball_joint_id]
        self._ball_dof_adr = self.model.jnt_dofadr[self.ball_joint_id]
        self._home_ctrl = self.model.key_ctrl[0].copy()
        self._home_joint_targets = self._home_ctrl[:7].copy()
        self._default_ball_height = DEFAULT_BALL_HEIGHT
        self._racket_jacobian = np.zeros((3, self.model.nv), dtype=float)
        self.reset()

    @property
    def home_joint_targets(self) -> np.ndarray:
        return self._home_joint_targets.copy()

    @property
    def home_gripper_target(self) -> float:
        return float(self._home_ctrl[7])

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[:7].copy()

    @property
    def ball_position(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].copy()

    @property
    def racket_position(self) -> np.ndarray:
        return self.data.site_xpos[self.racket_site_id].copy()

    @property
    def racket_velocity(self) -> np.ndarray:
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._racket_jacobian,
            None,
            self.racket_site_id,
        )
        return self._racket_jacobian @ self.data.qvel

    @property
    def racket_grip_position(self) -> np.ndarray:
        return self.data.xpos[self.racket_body_id].copy()

    @property
    def racket_face_normal(self) -> np.ndarray:
        racket_xmat = np.asarray(self.data.geom_xmat[self.racket_head_geom_id], dtype=float).reshape(3, 3)
        return racket_xmat[:, 2].copy()

    @property
    def ball_velocity(self) -> np.ndarray:
        return self.data.qvel[self._ball_dof_adr:self._ball_dof_adr + 3].copy()

    def reset(
        self,
        ball_position: Sequence[float] | None = None,
        ball_velocity: Sequence[float] = (0.0, 0.0, 0.0),
        ball_height: float | None = None,
        ball_xy_offset: Sequence[float] = (0.0, 0.0),
    ) -> mujoco.MjData:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self._home_ctrl
        mujoco.mj_forward(self.model, self.data)

        if ball_position is None:
            spawn_height = self._default_ball_height if ball_height is None else float(ball_height)
            self.reset_ball_above_racket(height=spawn_height, xy_offset=ball_xy_offset, velocity=ball_velocity)
        else:
            self.spawn_ball(ball_position, ball_velocity)

        return self.data

    def spawn_ball(self, position: Sequence[float], velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        position_array = np.asarray(position, dtype=float)
        velocity_array = np.asarray(velocity, dtype=float)

        if position_array.shape != (3,):
            raise ValueError(f"Ball position must have shape (3,), got {position_array.shape}.")
        if velocity_array.shape != (3,):
            raise ValueError(f"Ball velocity must have shape (3,), got {velocity_array.shape}.")

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3] = position_array
        qpos[self._ball_qpos_adr + 3:self._ball_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel[self._ball_dof_adr:self._ball_dof_adr + 3] = velocity_array
        qvel[self._ball_dof_adr + 3:self._ball_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.ball_position

    def reset_ball_above_racket(
        self,
        height: float = 0.25,
        xy_offset: Sequence[float] = (0.0, 0.0),
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        xy_offset_array = np.asarray(xy_offset, dtype=float)
        if xy_offset_array.shape != (2,):
            raise ValueError(f"xy_offset must have shape (2,), got {xy_offset_array.shape}.")

        spawn_position = self.racket_position + np.array([xy_offset_array[0], xy_offset_array[1], height])
        return self.spawn_ball(spawn_position, velocity)

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
            geom1 = self.model.geom(contact.geom1).name
            geom2 = self.model.geom(contact.geom2).name
            pairs.append(tuple(sorted((geom1, geom2))))
        return pairs

    def ball_robot_body_contact(self) -> str | None:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = self.model.geom(int(contact.geom1))
            geom2 = self.model.geom(int(contact.geom2))
            geom1_name = geom1.name or ""
            geom2_name = geom2.name or ""
            if geom1_name != "ball_geom" and geom2_name != "ball_geom":
                continue

            other_geom = geom2 if geom1_name == "ball_geom" else geom1
            other_geom_name = other_geom.name or ""
            other_body_name = self.model.body(int(np.asarray(other_geom.bodyid).item())).name
            if other_geom_name == "floor" or other_body_name in {"ball", "racket"}:
                continue
            return other_body_name
        return None

    def has_contact(self, geom_a: str, geom_b: str) -> bool:
        target = tuple(sorted((geom_a, geom_b)))
        return target in self.contact_pairs()

    def state_is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def failure_reason(
        self,
        x_bounds: tuple[float, float] = (0.0, 1.35),
        y_bounds: tuple[float, float] = (-0.6, 0.6),
        z_bounds: tuple[float, float] = (-0.05, 2.0),
        max_ball_speed: float = 8.0,
    ) -> str | None:
        if not self.state_is_finite():
            return "nonfinite_state"
        if self.has_contact("ball_geom", "floor"):
            return "floor_contact"
        if self.ball_robot_body_contact() is not None:
            return "robot_body_contact"

        ball_position = self.ball_position
        within_x = x_bounds[0] <= ball_position[0] <= x_bounds[1]
        within_y = y_bounds[0] <= ball_position[1] <= y_bounds[1]
        within_z = z_bounds[0] <= ball_position[2] <= z_bounds[1]
        if not (within_x and within_y and within_z):
            return "ball_out_of_bounds"
        if np.linalg.norm(self.ball_velocity) > max_ball_speed:
            return "ball_speed_limit"
        return None

    def reset_if_failed(
        self,
        ball_height: float | None = None,
        x_bounds: tuple[float, float] = (0.0, 1.35),
        y_bounds: tuple[float, float] = (-0.6, 0.6),
        z_bounds: tuple[float, float] = (-0.05, 2.0),
        max_ball_speed: float = 8.0,
    ) -> str | None:
        reason = self.failure_reason(
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            max_ball_speed=max_ball_speed,
        )
        if reason is None:
            return None

        self.reset(ball_height=self._default_ball_height if ball_height is None else float(ball_height))
        return reason

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

    def step_with_contact_trace(
        self,
        joint_targets: Sequence[float] | None = None,
        gripper_target: float | None = None,
        n_substeps: int | None = None,
        contact_geoms: tuple[str, str] = ("ball_geom", "racket_head"),
    ) -> dict[str, object]:
        if joint_targets is not None:
            self.set_arm_joint_targets(joint_targets, gripper_target)

        step_count = self.n_substeps if n_substeps is None else max(1, int(n_substeps))
        target_pair = tuple(sorted(contact_geoms))
        contact_trace: dict[str, object] = {
            "contact_observed": False,
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
        previous_racket_velocity = self.racket_velocity
        for substep_index in range(1, step_count + 1):
            mujoco.mj_step(self.model, self.data)
            racket_velocity = self.racket_velocity
            racket_acceleration = (racket_velocity - previous_racket_velocity) / self.model.opt.timestep
            previous_racket_velocity = racket_velocity
            if contact_trace["contact_observed"]:
                continue
            if target_pair not in self.contact_pairs():
                continue

            ball_velocity = self.ball_velocity
            contact_trace = {
                "contact_observed": True,
                "contact_substep": substep_index,
                "contact_ball_velocity_x": float(ball_velocity[0]),
                "contact_ball_velocity_y": float(ball_velocity[1]),
                "contact_ball_velocity_z": float(ball_velocity[2]),
                "contact_ball_speed_norm": float(np.linalg.norm(ball_velocity)),
                "contact_racket_velocity_x": float(racket_velocity[0]),
                "contact_racket_velocity_y": float(racket_velocity[1]),
                "contact_racket_velocity_z": float(racket_velocity[2]),
                "contact_racket_speed_norm": float(np.linalg.norm(racket_velocity)),
                "contact_racket_acceleration_x": float(racket_acceleration[0]),
                "contact_racket_acceleration_y": float(racket_acceleration[1]),
                "contact_racket_acceleration_z": float(racket_acceleration[2]),
                "contact_racket_acceleration_norm": float(np.linalg.norm(racket_acceleration)),
            }
        return contact_trace
