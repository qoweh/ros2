from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl.envs import PingPongEEDeltaEnv, PingPongSim
from pingpong_rl.controllers import RacketCartesianController, compute_keepup_target
from pingpong_rl.viewer import _ee_demo_target_position, _passive_viewer_is_running, parse_args


class PingPongSimTest(unittest.TestCase):
    def test_ball_bounces_off_floor_after_tuning(self) -> None:
        sim = PingPongSim()
        sim.reset(ball_position=(0.2, -0.25, 1.0), ball_velocity=(0.0, 0.0, 0.0))

        impact_seen = False
        post_impact_peak = None
        previous_vz = float(sim.ball_velocity[2])
        for _ in range(5000):
            sim.step(n_substeps=1)
            ball_height = float(sim.ball_position[2])
            vertical_velocity = float(sim.ball_velocity[2])
            if not impact_seen and sim.has_contact("ball_geom", "floor"):
                impact_seen = True
            elif impact_seen:
                if vertical_velocity > 0.0:
                    post_impact_peak = ball_height if post_impact_peak is None else max(post_impact_peak, ball_height)
                if post_impact_peak is not None and previous_vz > 0.0 and vertical_velocity <= 0.0:
                    break
            previous_vz = vertical_velocity

        self.assertTrue(impact_seen)
        self.assertIsNotNone(post_impact_peak)
        self.assertGreater(post_impact_peak, 0.2)

    def test_scene_loads_and_ball_resets_above_racket(self) -> None:
        sim = PingPongSim()
        sim.reset()
        ball_position = sim.reset_ball_above_racket(height=0.22)

        self.assertEqual(sim.model.nbody, 14)
        self.assertGreater(ball_position[2], sim.racket_position[2])
        self.assertAlmostEqual(sim.ball_position[0], ball_position[0], places=6)
        self.assertAlmostEqual(sim.ball_position[1], ball_position[1], places=6)
        self.assertAlmostEqual(sim.data.ctrl[7], sim.home_gripper_target, places=6)
        self.assertAlmostEqual(float(sim.data.joint("finger_joint1").qpos[0]), 0.012, places=6)
        self.assertAlmostEqual(float(sim.data.joint("finger_joint2").qpos[0]), 0.012, places=6)
        self.assertEqual(sim.data.ncon, 0)

    def test_racket_grip_sits_between_fingers(self) -> None:
        sim = PingPongSim()
        sim.reset()

        left_finger_position = sim.data.xpos[sim.model.body("left_finger").id]
        right_finger_position = sim.data.xpos[sim.model.body("right_finger").id]
        racket_grip_position = sim.racket_grip_position
        racket_center_position = sim.racket_position

        self.assertLess(right_finger_position[0], racket_grip_position[0])
        self.assertLess(racket_grip_position[0], left_finger_position[0])
        self.assertNotAlmostEqual(racket_center_position[1], racket_grip_position[1], places=3)
        self.assertGreater(sim.ball_position[2], racket_center_position[2])
        self.assertEqual(sim.data.ncon, 0)

    def test_joint_target_update_changes_ctrl_buffer(self) -> None:
        sim = PingPongSim()
        targets = sim.home_joint_targets
        targets[0] += 0.1
        sim.step(joint_targets=targets, n_substeps=1)

        self.assertAlmostEqual(sim.data.ctrl[0], targets[0], places=6)

    def test_ball_hits_racket_before_floor(self) -> None:
        sim = PingPongSim()
        sim.reset()

        first_target_contact: tuple[str, str] | None = None
        for _ in range(800):
            sim.step(n_substeps=1)
            if sim.has_contact("ball_geom", "racket_head"):
                first_target_contact = ("ball_geom", "racket_head")
                break
            if sim.has_contact("ball_geom", "floor"):
                first_target_contact = ("ball_geom", "floor")
                break

        self.assertEqual(first_target_contact, ("ball_geom", "racket_head"))

    def test_failure_reason_reports_out_of_bounds(self) -> None:
        sim = PingPongSim()
        sim.reset()
        sim.spawn_ball((1.8, 0.0, 0.5))

        self.assertEqual(sim.failure_reason(), "ball_out_of_bounds")

    def test_reset_if_failed_respawns_after_floor_contact(self) -> None:
        sim = PingPongSim()
        sim.reset()

        failure_reason = None
        for _ in range(1200):
            sim.step(n_substeps=1)
            failure_reason = sim.failure_reason()
            if failure_reason is not None:
                break

        self.assertEqual(failure_reason, "floor_contact")

        reset_reason = sim.reset_if_failed()
        self.assertEqual(reset_reason, "floor_contact")
        self.assertEqual(sim.data.ncon, 0)
        self.assertGreater(sim.ball_position[2], sim.racket_position[2])
        self.assertIsNone(sim.failure_reason())

    def test_racket_cartesian_controller_reduces_position_error(self) -> None:
        sim = PingPongSim(control_dt=0.02)
        sim.reset()
        controller = RacketCartesianController(sim)

        target_position = sim.racket_position + np.array([0.02, 0.0, 0.01])
        controller.set_target_position(target_position)
        initial_error = np.linalg.norm(target_position - sim.racket_position)

        for _ in range(15):
            joint_targets = controller.compute_joint_targets()
            sim.step(joint_targets=joint_targets, n_substeps=sim.n_substeps)

        final_error = np.linalg.norm(target_position - sim.racket_position)
        self.assertLess(final_error, initial_error)

    def test_passive_viewer_pause_helper_uses_run_flag(self) -> None:
        class FakeSimState:
            def __init__(self, run: int) -> None:
                self.run = run

        class FakeViewer:
            def __init__(self, run: int) -> None:
                self._sim_state = FakeSimState(run)

            def _get_sim(self) -> FakeSimState:
                return self._sim_state

        self.assertTrue(_passive_viewer_is_running(FakeViewer(1)))
        self.assertFalse(_passive_viewer_is_running(FakeViewer(0)))

    def test_viewer_parse_args_accepts_ee_demo(self) -> None:
        args = parse_args(["--demo-controller", "ee", "--ee-axis", "x", "--demo-amplitude", "0.03"])

        self.assertEqual(args.mode, "passive")
        self.assertEqual(args.demo_controller, "ee")
        self.assertEqual(args.ee_axis, "x")
        self.assertAlmostEqual(args.demo_amplitude, 0.03, places=6)

    def test_ee_demo_target_position_only_moves_selected_axis(self) -> None:
        anchor = np.array([0.5, 0.1, 0.6])
        target = _ee_demo_target_position(anchor, "z", amplitude=0.04, frequency=0.5, time_seconds=0.5)

        self.assertAlmostEqual(target[0], anchor[0], places=6)
        self.assertAlmostEqual(target[1], anchor[1], places=6)
        self.assertGreater(target[2], anchor[2])

    def test_keepup_target_tracks_descending_ball_with_xy_lead(self) -> None:
        anchor = np.array([0.55, 0.0, 0.52])
        ball_position = np.array([0.61, -0.03, 0.70])
        ball_velocity = np.array([0.20, -0.10, -0.40])

        target = compute_keepup_target(
            anchor,
            ball_position,
            ball_velocity,
            preview_time=0.10,
            strike_plane_offset=0.02,
            max_xy_offset=0.18,
            min_z_offset=-0.03,
            max_z_offset=0.10,
        )

        np.testing.assert_allclose(target[:2], np.array([0.63889, -0.04444]), atol=2.0e-5)
        self.assertAlmostEqual(float(target[2]), 0.54, places=6)

    def test_keepup_target_recenters_height_when_ball_is_rising(self) -> None:
        anchor = np.array([0.55, 0.0, 0.52])
        ball_position = np.array([0.72, 0.40, 0.80])
        ball_velocity = np.array([0.40, 0.30, 0.60])

        target = compute_keepup_target(
            anchor,
            ball_position,
            ball_velocity,
            preview_time=0.10,
            return_height_offset=0.0,
            max_xy_offset=0.18,
            min_z_offset=-0.03,
            max_z_offset=0.10,
        )

        np.testing.assert_allclose(target[:2], anchor[:2], atol=1.0e-6)
        self.assertAlmostEqual(float(target[2]), 0.52, places=6)

    def test_ee_delta_env_step_clips_action_and_returns_flat_contract(self) -> None:
        env = PingPongEEDeltaEnv()
        observation, reset_info = env.reset()
        unpacked_observation = env.unflatten_observation(observation)

        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertEqual(
            set(unpacked_observation),
            {
                "joint_positions",
                "joint_velocities",
                "racket_position",
                "racket_velocity",
                "target_position",
                "ball_position",
                "ball_velocity",
            },
        )
        self.assertIsNone(reset_info["failure_reason"])
        self.assertIsNone(reset_info["success_reason"])
        self.assertEqual(reset_info["step_count"], 0)
        self.assertEqual(reset_info["episode_steps"], 0)
        self.assertEqual(reset_info["contact_count"], 0)
        self.assertEqual(reset_info["successful_bounce_count"], 0)
        self.assertFalse(reset_info["time_limit_reached"])
        self.assertFalse(reset_info["terminated"])
        self.assertFalse(reset_info["truncated"])
        self.assertFalse(reset_info["contact_event_during_step"])
        self.assertIn("reward_height_target_term", reset_info)
        self.assertIn("reward_lift_term", reset_info)
        self.assertEqual(float(reset_info["reward_lift_term"]), 0.0)
        self.assertIn("reward_terms", reset_info)
        self.assertAlmostEqual(float(reset_info["reward_total"]), sum(reset_info["reward_terms"].values()), places=6)
        self.assertAlmostEqual(float(reset_info["reward_height"]), float(reset_info["reward_terms"]["height_term"]), places=6)
        self.assertAlmostEqual(float(reset_info["reward_distance"]), float(reset_info["reward_terms"]["distance_term"]), places=6)
        self.assertAlmostEqual(float(reset_info["reward_contact"]), float(reset_info["reward_terms"]["contact_bonus"]), places=6)
        self.assertAlmostEqual(float(reset_info["reward_failure"]), float(reset_info["reward_terms"]["failure_penalty"]), places=6)
        self.assertEqual(float(reset_info["reward_success"]), 0.0)
        np.testing.assert_allclose(unpacked_observation["target_position"], env.target_position)
        initial_target = env.target_position.copy()
        next_observation, reward, terminated, truncated, info = env.step((0.0, 0.0, 0.1))
        next_unpacked_observation = env.unflatten_observation(next_observation)

        self.assertEqual(next_unpacked_observation["joint_positions"].shape, (7,))
        self.assertEqual(next_unpacked_observation["joint_velocities"].shape, (7,))
        self.assertEqual(next_unpacked_observation["racket_position"].shape, (3,))
        self.assertEqual(next_unpacked_observation["racket_velocity"].shape, (3,))
        self.assertEqual(next_unpacked_observation["target_position"].shape, (3,))
        self.assertEqual(next_unpacked_observation["ball_position"].shape, (3,))
        self.assertEqual(next_unpacked_observation["ball_velocity"].shape, (3,))
        self.assertAlmostEqual(float(info["applied_action"][2]), env.action_limit, places=6)
        self.assertGreater(float(info["target_position"][2]), float(initial_target[2]))
        self.assertAlmostEqual(
            float(next_unpacked_observation["target_position"][2]),
            float(info["target_position"][2]),
            places=6,
        )
        self.assertIsNone(info["success_reason"])
        self.assertEqual(info["step_count"], 1)
        self.assertEqual(info["episode_steps"], 1)
        self.assertFalse(info["time_limit_reached"])
        self.assertFalse(info["terminated"])
        self.assertFalse(info["truncated"])
        self.assertFalse(info["contact_event_during_step"])
        self.assertIn("reward_height_target_term", info)
        self.assertIn("reward_lift_term", info)
        self.assertEqual(float(info["reward_lift_term"]), 0.0)
        self.assertAlmostEqual(float(info["reward_total"]), sum(info["reward_terms"].values()), places=6)
        self.assertAlmostEqual(float(info["reward_height"]), float(info["reward_terms"]["height_term"]), places=6)
        self.assertAlmostEqual(float(info["reward_distance"]), float(info["reward_terms"]["distance_term"]), places=6)
        self.assertAlmostEqual(float(info["reward_contact"]), float(info["reward_terms"]["contact_bonus"]), places=6)
        self.assertAlmostEqual(float(info["reward_failure"]), float(info["reward_terms"]["failure_penalty"]), places=6)
        self.assertEqual(float(info["reward_success"]), 0.0)
        self.assertAlmostEqual(float(info["ball_vertical_velocity"]), float(info["ball_velocity_z"]), places=6)
        self.assertGreaterEqual(float(info["ball_speed_norm"]), 0.0)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_ee_delta_env_truncates_at_time_limit_and_reset_clears_counter(self) -> None:
        env = PingPongEEDeltaEnv(max_episode_steps=2)
        _, reset_info = env.reset()

        self.assertEqual(reset_info["step_count"], 0)
        self.assertEqual(env.step_count, 0)

        _, _, terminated_1, truncated_1, info_1 = env.step((0.0, 0.0, 0.0))
        _, _, terminated_2, truncated_2, info_2 = env.step((0.0, 0.0, 0.0))

        self.assertFalse(terminated_1)
        self.assertFalse(truncated_1)
        self.assertEqual(info_1["step_count"], 1)
        self.assertEqual(info_1["episode_steps"], 1)
        self.assertFalse(info_1["time_limit_reached"])
        self.assertFalse(info_1["terminated"])
        self.assertFalse(info_1["truncated"])
        self.assertFalse(terminated_2)
        self.assertTrue(truncated_2)
        self.assertEqual(info_2["step_count"], 2)
        self.assertEqual(info_2["episode_steps"], 2)
        self.assertTrue(info_2["time_limit_reached"])
        self.assertFalse(info_2["terminated"])
        self.assertTrue(info_2["truncated"])
        self.assertEqual(env.step_count, 2)

        _, reset_info_after = env.reset()

        self.assertEqual(env.step_count, 0)
        self.assertEqual(reset_info_after["step_count"], 0)
        self.assertEqual(reset_info_after["episode_steps"], 0)
        self.assertFalse(reset_info_after["time_limit_reached"])

    def test_ee_delta_env_success_requires_racket_contact_and_upward_ball_velocity(self) -> None:
        class FakeSim:
            def __init__(self) -> None:
                self.n_substeps = 1
                self._joint_positions = np.zeros(7, dtype=float)
                self._joint_velocities = np.zeros(7, dtype=float)
                self._racket_position = np.array([0.55, 0.125, 0.52], dtype=float)
                self._ball_position = np.array([0.55, 0.125, 0.62], dtype=float)
                self._ball_velocity = np.zeros(3, dtype=float)
                self._racket_contact = False

            @property
            def joint_positions(self) -> np.ndarray:
                return self._joint_positions.copy()

            @property
            def joint_velocities(self) -> np.ndarray:
                return self._joint_velocities.copy()

            @property
            def racket_position(self) -> np.ndarray:
                return self._racket_position.copy()

            @property
            def ball_position(self) -> np.ndarray:
                return self._ball_position.copy()

            @property
            def ball_velocity(self) -> np.ndarray:
                return self._ball_velocity.copy()

            def reset(self, ball_height: float | None = None, ball_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
                self._ball_position = np.array([0.55, 0.125, 0.52 + (0.22 if ball_height is None else ball_height)], dtype=float)
                self._ball_velocity = np.asarray(ball_velocity, dtype=float)
                self._racket_contact = False

            def step(self, joint_targets: np.ndarray | None = None, n_substeps: int | None = None) -> None:
                self._racket_contact = True
                self._ball_velocity = np.array([0.02, 0.0, 0.8], dtype=float)

            def step_with_contact_trace(self, joint_targets: np.ndarray | None = None, n_substeps: int | None = None) -> dict[str, object]:
                self.step(joint_targets=joint_targets, n_substeps=n_substeps)
                return {
                    "contact_observed": True,
                    "contact_substep": 1,
                    "contact_ball_velocity_x": 0.02,
                    "contact_ball_velocity_y": 0.0,
                    "contact_ball_velocity_z": 0.8,
                    "contact_ball_speed_norm": 0.8,
                    "contact_racket_velocity_x": 0.0,
                    "contact_racket_velocity_y": 0.0,
                    "contact_racket_velocity_z": 0.3,
                    "contact_racket_speed_norm": 0.3,
                    "contact_racket_acceleration_x": 0.0,
                    "contact_racket_acceleration_y": 0.0,
                    "contact_racket_acceleration_z": 4.0,
                    "contact_racket_acceleration_norm": 4.0,
                }

            def failure_reason(self) -> None:
                return None

            def has_contact(self, geom_a: str, geom_b: str) -> bool:
                return self._racket_contact and {geom_a, geom_b} == {"ball_geom", "racket_head"}

        class FakeController:
            def __init__(self, target_position: np.ndarray) -> None:
                self._target_position = target_position.copy()

            @property
            def target_position(self) -> np.ndarray:
                return self._target_position.copy()

            def reset(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

            def add_target_offset(self, delta: tuple[float, float, float] | np.ndarray) -> np.ndarray:
                self._target_position = self._target_position + np.asarray(delta, dtype=float)
                return self.target_position

            def compute_joint_targets(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

        env = PingPongEEDeltaEnv(success_velocity_threshold=0.5, max_episode_steps=200)
        fake_sim = FakeSim()
        env.sim = fake_sim
        env.controller = FakeController(fake_sim.racket_position)

        env.reset()
        _, _, terminated, truncated, success_info = env.step((0.0, 0.0, 0.0))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(success_info["failure_reason"])
        self.assertEqual(success_info["success_reason"], "upward_racket_bounce")
        self.assertTrue(success_info["racket_contact"])
        self.assertFalse(success_info["terminated"])
        self.assertFalse(success_info["truncated"])
        self.assertTrue(success_info["contact_event_during_step"])
        self.assertEqual(success_info["contact_count"], 1)
        self.assertEqual(success_info["successful_bounce_count"], 1)
        self.assertGreater(success_info["ball_vertical_velocity"], env.success_velocity_threshold)
        self.assertEqual(float(success_info["reward_success"]), env.success_bonus)

    def test_ee_delta_env_success_uses_transient_contact_trace(self) -> None:
        class FakeSim:
            def __init__(self) -> None:
                self.n_substeps = 1
                self._joint_positions = np.zeros(7, dtype=float)
                self._joint_velocities = np.zeros(7, dtype=float)
                self._racket_position = np.array([0.55, 0.125, 0.52], dtype=float)
                self._ball_position = np.array([0.55, 0.125, 0.62], dtype=float)
                self._ball_velocity = np.zeros(3, dtype=float)

            @property
            def joint_positions(self) -> np.ndarray:
                return self._joint_positions.copy()

            @property
            def joint_velocities(self) -> np.ndarray:
                return self._joint_velocities.copy()

            @property
            def racket_position(self) -> np.ndarray:
                return self._racket_position.copy()

            @property
            def ball_position(self) -> np.ndarray:
                return self._ball_position.copy()

            @property
            def ball_velocity(self) -> np.ndarray:
                return self._ball_velocity.copy()

            def reset(self, ball_height: float | None = None, ball_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
                self._ball_position = np.array([0.55, 0.125, 0.52 + (0.22 if ball_height is None else ball_height)], dtype=float)
                self._ball_velocity = np.asarray(ball_velocity, dtype=float)

            def step_with_contact_trace(self, joint_targets: np.ndarray | None = None, n_substeps: int | None = None) -> dict[str, object]:
                self._ball_velocity = np.array([0.01, 0.0, 0.2], dtype=float)
                return {
                    "contact_observed": True,
                    "contact_substep": 1,
                    "contact_ball_velocity_x": 0.02,
                    "contact_ball_velocity_y": 0.0,
                    "contact_ball_velocity_z": 0.8,
                    "contact_ball_speed_norm": 0.8,
                    "contact_racket_velocity_x": 0.0,
                    "contact_racket_velocity_y": 0.0,
                    "contact_racket_velocity_z": 0.3,
                    "contact_racket_speed_norm": 0.3,
                    "contact_racket_acceleration_x": 0.0,
                    "contact_racket_acceleration_y": 0.0,
                    "contact_racket_acceleration_z": 4.0,
                    "contact_racket_acceleration_norm": 4.0,
                }

            def failure_reason(self) -> None:
                return None

            def has_contact(self, geom_a: str, geom_b: str) -> bool:
                return False

        class FakeController:
            def __init__(self, target_position: np.ndarray) -> None:
                self._target_position = target_position.copy()

            @property
            def target_position(self) -> np.ndarray:
                return self._target_position.copy()

            def reset(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

            def add_target_offset(self, delta: tuple[float, float, float] | np.ndarray) -> np.ndarray:
                self._target_position = self._target_position + np.asarray(delta, dtype=float)
                return self.target_position

            def compute_joint_targets(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

        env = PingPongEEDeltaEnv(success_velocity_threshold=0.5, max_episode_steps=200)
        fake_sim = FakeSim()
        env.sim = fake_sim
        env.controller = FakeController(fake_sim.racket_position)

        env.reset()
        _, _, terminated, truncated, success_info = env.step((0.0, 0.0, 0.0))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(success_info["success_reason"], "upward_racket_bounce")
        self.assertTrue(success_info["contact_observed_during_step"])
        self.assertTrue(success_info["contact_event_during_step"])
        self.assertEqual(float(success_info["reward_success"]), env.success_bonus)
        self.assertLess(float(success_info["reward_lift_term"]), 0.0)

    def test_ee_delta_env_height_reward_prefers_target_band_over_excessive_height(self) -> None:
        class FakeSim:
            def __init__(self) -> None:
                self._racket_position = np.array([0.55, 0.125, 0.52], dtype=float)
                self._ball_position = np.array([0.55, 0.125, 0.80], dtype=float)
                self._ball_velocity = np.zeros(3, dtype=float)

            @property
            def racket_position(self) -> np.ndarray:
                return self._racket_position.copy()

            @property
            def ball_position(self) -> np.ndarray:
                return self._ball_position.copy()

            @property
            def ball_velocity(self) -> np.ndarray:
                return self._ball_velocity.copy()

        env = PingPongEEDeltaEnv()
        fake_sim = FakeSim()
        env.sim = fake_sim

        target_band_reward = env._height_target_term()
        fake_sim._ball_position = np.array([0.55, 0.125, 1.60], dtype=float)
        overshoot_reward = env._height_target_term()

        self.assertGreater(target_band_reward, overshoot_reward)
        self.assertGreater(target_band_reward, 0.0)
        self.assertLess(overshoot_reward, 0.0)

    def test_ee_delta_env_lift_reward_prefers_useful_upward_velocity_over_weak_or_excessive_hits(self) -> None:
        class FakeSim:
            def __init__(self) -> None:
                self._racket_position = np.array([0.55, 0.125, 0.52], dtype=float)
                self._ball_position = np.array([0.55, 0.125, 0.80], dtype=float)
                self._ball_velocity = np.zeros(3, dtype=float)

            @property
            def racket_position(self) -> np.ndarray:
                return self._racket_position.copy()

            @property
            def ball_position(self) -> np.ndarray:
                return self._ball_position.copy()

            @property
            def ball_velocity(self) -> np.ndarray:
                return self._ball_velocity.copy()

        env = PingPongEEDeltaEnv()
        env.sim = FakeSim()
        active_trace = {
            "contact_racket_velocity_z": env.target_active_racket_velocity_z,
            "contact_racket_acceleration_z": env.target_active_racket_acceleration_z,
        }

        weak_lift_reward = env._lift_term(True, {"contact_ball_velocity_z": 0.4, **active_trace})
        target_lift_reward = env._lift_term(True, {"contact_ball_velocity_z": env.target_contact_velocity_z, **active_trace})
        overshoot_lift_reward = env._lift_term(True, {"contact_ball_velocity_z": 4.5, **active_trace})

        self.assertLess(weak_lift_reward, 0.0)
        self.assertGreater(target_lift_reward, weak_lift_reward)
        self.assertGreater(target_lift_reward, overshoot_lift_reward)

    def test_ee_delta_env_persistent_contact_counts_once_and_penalizes_stale_contact(self) -> None:
        class FakeSim:
            def __init__(self) -> None:
                self.n_substeps = 1
                self._joint_positions = np.zeros(7, dtype=float)
                self._joint_velocities = np.zeros(7, dtype=float)
                self._racket_position = np.array([0.55, 0.125, 0.52], dtype=float)
                self._ball_position = np.array([0.55, 0.125, 0.62], dtype=float)
                self._ball_velocity = np.zeros(3, dtype=float)

            @property
            def joint_positions(self) -> np.ndarray:
                return self._joint_positions.copy()

            @property
            def joint_velocities(self) -> np.ndarray:
                return self._joint_velocities.copy()

            @property
            def racket_position(self) -> np.ndarray:
                return self._racket_position.copy()

            @property
            def ball_position(self) -> np.ndarray:
                return self._ball_position.copy()

            @property
            def ball_velocity(self) -> np.ndarray:
                return self._ball_velocity.copy()

            def reset(self, ball_height: float | None = None, ball_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
                self._ball_position = np.array([0.55, 0.125, 0.52 + (0.22 if ball_height is None else ball_height)], dtype=float)
                self._ball_velocity = np.asarray(ball_velocity, dtype=float)

            def step_with_contact_trace(self, joint_targets: np.ndarray | None = None, n_substeps: int | None = None) -> dict[str, object]:
                self._ball_velocity = np.array([0.0, 0.0, 0.9], dtype=float)
                return {
                    "contact_observed": True,
                    "contact_substep": 1,
                    "contact_ball_velocity_x": 0.0,
                    "contact_ball_velocity_y": 0.0,
                    "contact_ball_velocity_z": 0.9,
                    "contact_ball_speed_norm": 0.9,
                    "contact_racket_velocity_x": 0.0,
                    "contact_racket_velocity_y": 0.0,
                    "contact_racket_velocity_z": 0.3,
                    "contact_racket_speed_norm": 0.3,
                    "contact_racket_acceleration_x": 0.0,
                    "contact_racket_acceleration_y": 0.0,
                    "contact_racket_acceleration_z": 4.0,
                    "contact_racket_acceleration_norm": 4.0,
                }

            def failure_reason(self) -> None:
                return None

            def has_contact(self, geom_a: str, geom_b: str) -> bool:
                return False

        class FakeController:
            def __init__(self, target_position: np.ndarray) -> None:
                self._target_position = target_position.copy()

            @property
            def target_position(self) -> np.ndarray:
                return self._target_position.copy()

            def reset(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

            def add_target_offset(self, delta: tuple[float, float, float] | np.ndarray) -> np.ndarray:
                self._target_position = self._target_position + np.asarray(delta, dtype=float)
                return self.target_position

            def compute_joint_targets(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

        env = PingPongEEDeltaEnv(success_velocity_threshold=0.5, max_episode_steps=200)
        fake_sim = FakeSim()
        env.sim = fake_sim
        env.controller = FakeController(fake_sim.racket_position)

        env.reset()
        _, _, terminated_1, truncated_1, info_1 = env.step((0.0, 0.0, 0.0))
        _, _, terminated_2, truncated_2, info_2 = env.step((0.0, 0.0, 0.0))

        self.assertFalse(terminated_1)
        self.assertFalse(truncated_1)
        self.assertFalse(terminated_2)
        self.assertFalse(truncated_2)
        self.assertTrue(info_1["contact_event_during_step"])
        self.assertFalse(info_2["contact_event_during_step"])
        self.assertEqual(info_2["contact_count"], 1)
        self.assertEqual(info_2["successful_bounce_count"], 1)
        self.assertEqual(float(info_1["reward_contact"]), env.contact_bonus)
        self.assertEqual(float(info_2["reward_contact"]), env.stale_contact_penalty)
        self.assertEqual(float(info_2["reward_success"]), 0.0)

    def test_ee_delta_env_counts_multiple_bounces_only_after_contact_releases(self) -> None:
        class FakeSim:
            def __init__(self) -> None:
                self.n_substeps = 1
                self._joint_positions = np.zeros(7, dtype=float)
                self._joint_velocities = np.zeros(7, dtype=float)
                self._racket_position = np.array([0.55, 0.125, 0.52], dtype=float)
                self._ball_position = np.array([0.55, 0.125, 0.62], dtype=float)
                self._ball_velocity = np.zeros(3, dtype=float)
                self._step_index = 0

            @property
            def joint_positions(self) -> np.ndarray:
                return self._joint_positions.copy()

            @property
            def joint_velocities(self) -> np.ndarray:
                return self._joint_velocities.copy()

            @property
            def racket_position(self) -> np.ndarray:
                return self._racket_position.copy()

            @property
            def ball_position(self) -> np.ndarray:
                return self._ball_position.copy()

            @property
            def ball_velocity(self) -> np.ndarray:
                return self._ball_velocity.copy()

            def reset(self, ball_height: float | None = None, ball_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
                self._ball_position = np.array([0.55, 0.125, 0.52 + (0.22 if ball_height is None else ball_height)], dtype=float)
                self._ball_velocity = np.asarray(ball_velocity, dtype=float)
                self._step_index = 0

            def step_with_contact_trace(self, joint_targets: np.ndarray | None = None, n_substeps: int | None = None) -> dict[str, object]:
                self._step_index += 1
                if self._step_index == 1:
                    self._ball_velocity = np.array([0.0, 0.0, 0.9], dtype=float)
                    return {
                        "contact_observed": True,
                        "contact_substep": 1,
                        "contact_ball_velocity_x": 0.0,
                        "contact_ball_velocity_y": 0.0,
                        "contact_ball_velocity_z": 0.9,
                        "contact_ball_speed_norm": 0.9,
                        "contact_racket_velocity_x": 0.0,
                        "contact_racket_velocity_y": 0.0,
                        "contact_racket_velocity_z": 0.3,
                        "contact_racket_speed_norm": 0.3,
                        "contact_racket_acceleration_x": 0.0,
                        "contact_racket_acceleration_y": 0.0,
                        "contact_racket_acceleration_z": 4.0,
                        "contact_racket_acceleration_norm": 4.0,
                    }
                if self._step_index == 2:
                    self._ball_velocity = np.array([0.0, 0.0, -0.2], dtype=float)
                    return {
                        "contact_observed": False,
                        "contact_substep": None,
                        "contact_ball_velocity_x": None,
                        "contact_ball_velocity_y": None,
                        "contact_ball_velocity_z": None,
                        "contact_ball_speed_norm": None,
                    }
                self._ball_velocity = np.array([0.0, 0.0, 1.0], dtype=float)
                return {
                    "contact_observed": True,
                    "contact_substep": 1,
                    "contact_ball_velocity_x": 0.0,
                    "contact_ball_velocity_y": 0.0,
                    "contact_ball_velocity_z": 1.0,
                    "contact_ball_speed_norm": 1.0,
                    "contact_racket_velocity_x": 0.0,
                    "contact_racket_velocity_y": 0.0,
                    "contact_racket_velocity_z": 0.3,
                    "contact_racket_speed_norm": 0.3,
                    "contact_racket_acceleration_x": 0.0,
                    "contact_racket_acceleration_y": 0.0,
                    "contact_racket_acceleration_z": 4.0,
                    "contact_racket_acceleration_norm": 4.0,
                }

            def failure_reason(self) -> None:
                return None

            def has_contact(self, geom_a: str, geom_b: str) -> bool:
                return False

        class FakeController:
            def __init__(self, target_position: np.ndarray) -> None:
                self._target_position = target_position.copy()

            @property
            def target_position(self) -> np.ndarray:
                return self._target_position.copy()

            def reset(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

            def add_target_offset(self, delta: tuple[float, float, float] | np.ndarray) -> np.ndarray:
                self._target_position = self._target_position + np.asarray(delta, dtype=float)
                return self.target_position

            def compute_joint_targets(self) -> np.ndarray:
                return np.zeros(7, dtype=float)

        env = PingPongEEDeltaEnv(success_velocity_threshold=0.5, max_episode_steps=200)
        fake_sim = FakeSim()
        env.sim = fake_sim
        env.controller = FakeController(fake_sim.racket_position)

        env.reset()
        _, _, _, _, info_1 = env.step((0.0, 0.0, 0.0))
        _, _, _, _, info_2 = env.step((0.0, 0.0, 0.0))
        _, _, _, _, info_3 = env.step((0.0, 0.0, 0.0))

        self.assertEqual(info_1["contact_count"], 1)
        self.assertEqual(info_2["contact_count"], 1)
        self.assertEqual(info_3["contact_count"], 2)
        self.assertEqual(info_3["successful_bounce_count"], 2)
        self.assertGreater(float(info_3["reward_success"]), float(info_1["reward_success"]))


if __name__ == "__main__":
    unittest.main()
