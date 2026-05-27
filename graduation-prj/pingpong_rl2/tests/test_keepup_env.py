from __future__ import annotations

import unittest

import numpy as np

from pingpong_rl2.envs import PingPongKeepUpEnv, PingPongKeepUpGymEnv


class PingPongKeepUpEnvTests(unittest.TestCase):
    def test_keepup_env_reset_returns_expected_shape(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertEqual(info["contact_count"], 0)
        self.assertEqual(info["successful_bounce_count"], 0)

    def test_observation_includes_racket_velocity_slice(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        observation, _ = env.reset()
        racket_velocity_slice = env.observation_slices["racket_velocity"]
        self.assertTrue(np.allclose(observation[racket_velocity_slice], env.sim.racket_velocity))

    def test_default_action_limits_bias_vertical_motion(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0)
        self.assertLess(env.action_high[0], env.action_high[2])
        self.assertAlmostEqual(env.action_high[0], env.training_config()["lateral_action_limit"])
        self.assertAlmostEqual(env.action_high[2], env.training_config()["vertical_action_limit"])

    def test_position_tilt_mode_exposes_tilt_state(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_tilt", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        observation, _ = env.reset()
        self.assertEqual(env.action_size, 5)
        self.assertIn("racket_face_normal", env.observation_slices)
        self.assertIn("target_tilt", env.observation_slices)
        self.assertTrue(
            np.allclose(observation[env.observation_slices["racket_face_normal"]], env.sim.racket_face_normal)
        )
        self.assertTrue(np.allclose(observation[env.observation_slices["target_tilt"]], env.controller.target_tilt))

    def test_position_tilt_step_accepts_five_dim_action(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_tilt", reset_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        observation, _, _, _, info = env.step(np.zeros(5, dtype=float))
        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertIn("target_tilt", info)

    def test_position_tilt_target_pitch_range_clips_outward_pitch(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_tilt",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            target_pitch_range=(0.0, 0.06),
        )
        env.reset(ball_height=env.ball_height)
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, -0.015, 0.0], dtype=float))
        self.assertAlmostEqual(float(info["target_tilt"][0]), 0.0)

    def test_position_tilt_target_pitch_range_is_reported_in_training_config(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_tilt",
            reset_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            target_pitch_range=(0.0, 0.06),
        )
        self.assertEqual(env.training_config()["target_pitch_range"], [0.0, 0.06])

    def test_position_tilt_reset_applies_initial_target_tilt(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_tilt",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            target_pitch_range=(0.0, 0.06),
            initial_target_tilt=(0.03, 0.0),
        )
        observation, info = env.reset(ball_height=env.ball_height)
        self.assertTrue(np.allclose(observation[env.observation_slices["target_tilt"]], np.array([0.03, 0.0])))
        self.assertTrue(np.allclose(info["target_tilt"], np.array([0.03, 0.0])))


    def test_keepup_env_seed_repeats_reset_sample(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.05, reset_velocity_xy_range=0.01)
        env.seed(123)
        env.reset()
        first_ball_position = env.sim.ball_position.copy()
        first_ball_velocity = env.sim.ball_velocity.copy()
        env.seed(123)
        env.reset()
        second_ball_position = env.sim.ball_position.copy()
        second_ball_velocity = env.sim.ball_velocity.copy()
        self.assertTrue(np.allclose(first_ball_position, second_ball_position))
        self.assertTrue(np.allclose(first_ball_velocity, second_ball_velocity))


    def test_pre_contact_guard_limits_xy_target(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        guarded_target = env._guarded_target_position(anchor_position + np.array([0.25, -0.25, 0.05]))
        xy_limit = env._pre_contact_xy_limit()
        self.assertTrue(np.all(np.abs(guarded_target[:2] - anchor_position[:2]) <= xy_limit + 1.0e-9))

    def test_pre_contact_xy_limit_expands_for_low_descending_ball_even_when_misaligned(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.08, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        self.assertGreater(env._pre_contact_xy_limit(), 0.11)

    def test_position_tilt_xy_limit_stays_more_conservative_than_position(self) -> None:
        position_env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        tilt_env = PingPongKeepUpEnv(action_mode="position_tilt", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        for env in (position_env, tilt_env):
            env.reset(ball_height=env.ball_height)
            ball_position = env.sim.racket_position + np.array([0.08, 0.0, env._preparation_target_height_above_racket()])
            env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        self.assertGreater(position_env._pre_contact_xy_limit(), tilt_env._pre_contact_xy_limit())


    def test_tracking_term_stays_active_after_first_contact(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset()
        env.contact_count = 1
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        self.assertGreater(env._tracking_term(), 0.0)

    def test_contact_active_suppresses_tracking_reward(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset()
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=False,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={},
        )
        self.assertEqual(reward_terms["tracking_term"], 0.0)

    def test_position_tilt_penalties_are_negative_when_tilt_changes(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_tilt", reset_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        env.controller.set_target_tilt((0.03, -0.03))
        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=False,
            contact_active=False,
            applied_action=np.array([0.0, 0.0, 0.0, 0.015, -0.015], dtype=float),
            contact_trace={},
        )
        self.assertLess(reward_terms["tilt_angle_penalty"], 0.0)
        self.assertLess(reward_terms["tilt_action_delta_penalty"], 0.0)

    def test_success_reason_requires_centered_contact(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        success_reason = env._success_reason(
            failure_reason=None,
            contact_trace={
                "contact_ball_velocity_z": 4.0,
                "contact_racket_velocity_z": 0.2,
                "contact_xy_alignment_error": env.contact_centering_radius + 0.01,
                "contact_ball_height_above_racket": 0.02,
            },
            contact_event=True,
        )
        self.assertIsNone(success_reason)

    def test_success_reason_accepts_centered_upward_contact(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        success_reason = env._success_reason(
            failure_reason=None,
            contact_trace={
                "contact_ball_velocity_z": 4.0,
                "contact_racket_velocity_z": 0.2,
                "contact_xy_alignment_error": env.contact_centering_radius - 0.01,
                "contact_ball_height_above_racket": 0.02,
            },
            contact_event=True,
        )
        self.assertEqual(success_reason, "useful_keepup_bounce")


    def test_strike_guard_reapplies_after_first_contact(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0)
        env.reset(ball_height=env.ball_height, ball_velocity=(0.0, 0.0, 0.0))
        env.contact_count = 1
        anchor_position = env._controller_anchor_position()
        guarded_target = env._guarded_target_position(anchor_position + np.array([0.0, 0.0, 0.12]))
        self.assertLessEqual(guarded_target[2], anchor_position[2] + 0.02 + 1.0e-9)

    def test_post_contact_guard_preserves_xy_limit(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        baseline_limit = env._pre_contact_xy_limit()
        env.contact_count = 1
        guarded_limit = env._pre_contact_xy_limit()
        self.assertAlmostEqual(guarded_limit, baseline_limit)


    def test_gym_wrapper_exposes_box_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (3,))

    def test_gym_wrapper_exposes_position_tilt_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(action_mode="position_tilt", reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (5,))


if __name__ == "__main__":
    unittest.main()
