from __future__ import annotations

import unittest

import numpy as np

from pingpong_rl2.controllers import HeuristicKeepUpPolicy
from pingpong_rl2.envs import PingPongKeepUpEnv, PingPongKeepUpGymEnv


class PingPongKeepUpEnvTests(unittest.TestCase):
    def test_keepup_env_reset_returns_expected_shape(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        observation, info = env.reset()
        self.assertEqual(observation.shape, (env.observation_size,))
        self.assertEqual(info["contact_count"], 0)
        self.assertEqual(info["successful_bounce_count"], 0)

    def test_step_info_exposes_controller_anchor_position(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))
        self.assertIn("controller_anchor_position", info)
        self.assertIn("contact_ball_position_x", info)
        self.assertIn("contact_ball_position_y", info)
        self.assertTrue(np.allclose(info["controller_anchor_position"], env._controller_anchor_position()))

    def test_observation_includes_racket_velocity_slice(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        observation, _ = env.reset()
        racket_velocity_slice = env.observation_slices["racket_velocity"]
        self.assertTrue(np.allclose(observation[racket_velocity_slice], env.sim.racket_velocity))

    def test_observation_includes_predicted_intercept_relative_xy_slice(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.4, 0.0, -1.0))
        observation = env.observation()
        predicted_intercept_slice = env.observation_slices["predicted_intercept_relative_xy"]
        expected_value = env._predicted_intercept_xy() - env.sim.racket_position[:2]
        self.assertTrue(np.allclose(observation[predicted_intercept_slice], expected_value))
        self.assertGreater(float(observation[predicted_intercept_slice][0]), 0.0)

    def test_observation_includes_predicted_intercept_time_slice(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.4, 0.0, -1.0))
        observation = env.observation()
        predicted_intercept_time_slice = env.observation_slices["predicted_intercept_time"]
        expected_value = env._predicted_intercept_time()
        self.assertAlmostEqual(float(observation[predicted_intercept_time_slice][0]), expected_value)
        self.assertGreater(expected_value, 0.0)

    def test_velocity_domain_observation_is_opt_in(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        self.assertNotIn("relative_velocity", env.observation_slices)
        self.assertNotIn("racket_face_normal", env.observation_slices)

    def test_position_strike_velocity_domain_observation_includes_relative_velocity_and_face_normal(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            include_velocity_domain_observation=True,
        )
        observation, _ = env.reset(ball_height=env.ball_height)
        self.assertIn("relative_velocity", env.observation_slices)
        self.assertIn("racket_face_normal", env.observation_slices)
        relative_velocity_slice = env.observation_slices["relative_velocity"]
        self.assertTrue(
            np.allclose(observation[relative_velocity_slice], env.sim.ball_velocity - env.sim.racket_velocity)
        )
        self.assertTrue(
            np.allclose(observation[env.observation_slices["racket_face_normal"]], env.sim.racket_face_normal)
        )

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

    def test_position_strike_mode_keeps_three_dim_action(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_strike", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        observation, _ = env.reset()
        self.assertEqual(env.action_size, 3)
        self.assertEqual(observation.shape, (env.observation_size,))

    def test_position_strike_reset_applies_initial_target_tilt(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            initial_target_tilt=(0.03, 0.0),
        )
        _, info = env.reset(ball_height=env.ball_height)
        self.assertTrue(np.allclose(info["target_tilt"], np.array([0.03, 0.0])))

    def test_position_strike_step_preserves_initial_target_tilt(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            initial_target_tilt=(0.03, 0.0),
        )
        env.reset(ball_height=env.ball_height)
        _, _, _, _, info = env.step(np.zeros(3, dtype=float))
        self.assertTrue(np.allclose(info["target_tilt"], np.array([0.03, 0.0])))

    def test_position_strike_tilt_assist_targets_robot_center_in_xy(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            strike_tilt_assist_limit=(0.03, 0.03),
            strike_tilt_assist_deadband=0.0,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.03, -0.03, env._preparation_target_height_above_racket() + 0.01])
        env.sim.spawn_ball(ball_position, velocity=(0.2, 0.0, -1.0))
        target_tilt = env._strike_tilt_assist_target()
        self.assertLess(float(target_tilt[0]), -0.01)
        self.assertLess(float(target_tilt[1]), -0.01)

    def test_position_strike_tilt_assist_returns_neutral_after_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            strike_tilt_assist_limit=(0.03, 0.03),
            strike_tilt_assist_deadband=0.0,
        )
        env.reset(ball_height=env.ball_height)
        env._contact_active_previous_step = True
        self.assertTrue(np.allclose(env._strike_tilt_assist_target(), np.zeros(2, dtype=float)))

    def test_position_strike_step_applies_tilt_assist(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            strike_tilt_assist_limit=(0.03, 0.03),
            strike_tilt_assist_deadband=0.0,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.03, 0.0, env._preparation_target_height_above_racket() + 0.01])
        env.sim.spawn_ball(ball_position, velocity=(0.2, 0.0, -1.0))
        _, _, _, _, info = env.step(np.zeros(3, dtype=float))
        self.assertLess(float(info["target_tilt"][0]), -0.01)

    def test_position_strike_timed_negative_pitch_ramp_targets_inward_pitch(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            strike_tilt_ramp_pitch=-0.03,
            strike_tilt_ramp_xy_tolerance=0.05,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.02, 0.0, env._preparation_target_height_above_racket() + 0.01])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        target_tilt = env._strike_tilt_ramp_target()
        self.assertLess(float(target_tilt[0]), -0.01)
        self.assertAlmostEqual(float(target_tilt[1]), 0.0)

    def test_position_strike_timed_negative_pitch_ramp_stays_neutral_when_misaligned(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            strike_tilt_ramp_pitch=-0.03,
            strike_tilt_ramp_xy_tolerance=0.03,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.06, 0.0, env._preparation_target_height_above_racket() + 0.01])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        self.assertTrue(np.allclose(env._strike_tilt_ramp_target(), np.zeros(2, dtype=float)))

    def test_position_strike_step_applies_timed_negative_pitch_ramp(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            strike_tilt_ramp_pitch=-0.03,
            strike_tilt_ramp_xy_tolerance=0.05,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.02, 0.0, env._preparation_target_height_above_racket() + 0.01])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        _, _, _, _, info = env.step(np.zeros(3, dtype=float))
        self.assertLess(float(info["target_tilt"][0]), -0.01)

    def test_position_strike_timed_negative_pitch_ramp_conflicts_with_center_assist(self) -> None:
        with self.assertRaises(ValueError):
            PingPongKeepUpEnv(
                action_mode="position_strike",
                strike_tilt_assist_limit=(0.03, 0.03),
                strike_tilt_ramp_pitch=-0.03,
            )

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

    def test_tracking_term_uses_predicted_intercept_xy(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        stationary_tracking_term = env._tracking_term()
        env.sim.spawn_ball(ball_position, velocity=(0.4, 0.0, -1.0))
        moving_tracking_term = env._tracking_term()
        self.assertLess(moving_tracking_term, stationary_tracking_term)

    def test_pre_contact_readiness_uses_predicted_intercept_xy(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        stationary_readiness = env._pre_contact_readiness()
        env.sim.spawn_ball(ball_position, velocity=(0.4, 0.0, -1.0))
        moving_readiness = env._pre_contact_readiness()
        self.assertLess(moving_readiness, stationary_readiness)

    def test_position_strike_step_anchors_xy_to_predicted_intercept(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_strike", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket() + 0.01])
        env.sim.spawn_ball(ball_position, velocity=(0.3, -0.1, -1.0))
        expected_xy = env._predicted_intercept_xy()
        _, _, _, _, info = env.step(np.zeros(3, dtype=float))
        self.assertTrue(np.allclose(info["target_position"][:2], expected_xy, atol=1.0e-6))

    def test_position_strike_post_contact_return_assist_biases_xy_toward_future_intercept(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            post_contact_return_assist_weight=0.5,
            post_contact_return_max_intercept_time=0.6,
        )
        env.reset(ball_height=env.ball_height)
        env.successful_bounce_count = 1
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, 0.10])
        env.sim.spawn_ball(ball_position, velocity=(0.3, -0.1, 1.0))
        anchor_xy = env._controller_anchor_position()[:2]
        predicted_return_xy = env._predicted_intercept_xy(max_intercept_time=0.6)
        target_position = env._strike_action_target_position(np.zeros(3, dtype=float))
        expected_xy = 0.5 * anchor_xy + 0.5 * predicted_return_xy
        self.assertTrue(np.allclose(target_position[:2], expected_xy, atol=1.0e-6))

    def test_position_strike_post_contact_return_assist_stays_neutral_without_successful_bounce(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            post_contact_return_assist_weight=0.5,
            post_contact_return_max_intercept_time=0.6,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, 0.10])
        env.sim.spawn_ball(ball_position, velocity=(0.3, -0.1, 1.0))
        target_position = env._strike_action_target_position(np.zeros(3, dtype=float))
        self.assertTrue(np.allclose(target_position[:2], env._controller_anchor_position()[:2], atol=1.0e-6))

    def test_strike_lift_feedforward_raises_pre_contact_z_cap(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket() + 0.04])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.4))
        anchor_position = env._controller_anchor_position()
        strike_lift_feedforward = env._strike_lift_feedforward()
        guarded_target = env._guarded_target_position(anchor_position + np.array([0.0, 0.0, 0.12]))
        self.assertGreater(strike_lift_feedforward, 0.0)
        self.assertAlmostEqual(float(guarded_target[2]), float(anchor_position[2] + 0.02 + strike_lift_feedforward))

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

    def test_outgoing_x_term_applies_on_upward_contact_event(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            useful_contact_outgoing_x_penalty_weight=0.5,
            desired_outgoing_ball_velocity_x=-0.15,
        )
        env.reset(ball_height=env.ball_height)
        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason="useful_keepup_bounce",
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={"contact_ball_velocity_x": 0.2, "contact_ball_velocity_z": 0.8},
        )
        self.assertAlmostEqual(reward_terms["outgoing_x_term"], -0.175)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={"contact_ball_velocity_x": 0.2, "contact_ball_velocity_z": 0.8},
        )
        self.assertAlmostEqual(reward_terms["outgoing_x_term"], -0.175)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason="useful_keepup_bounce",
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={"contact_ball_velocity_x": -0.2, "contact_ball_velocity_z": 0.8},
        )
        self.assertEqual(reward_terms["outgoing_x_term"], 0.0)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason="useful_keepup_bounce",
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={"contact_ball_velocity_x": 0.2, "contact_ball_velocity_z": -0.1},
        )
        self.assertEqual(reward_terms["outgoing_x_term"], 0.0)

    def test_outgoing_x_term_stays_zero_when_weight_disabled(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            useful_contact_outgoing_x_penalty_weight=0.0,
            desired_outgoing_ball_velocity_x=-0.15,
        )
        env.reset(ball_height=env.ball_height)
        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason="useful_keepup_bounce",
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={"contact_ball_velocity_x": 0.2, "contact_ball_velocity_z": 0.8},
        )
        self.assertEqual(reward_terms["outgoing_x_term"], 0.0)

    def test_trajectory_error_penalty_applies_on_upward_contact_event(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            trajectory_error_penalty_weight=0.5,
        )
        env.reset(ball_height=env.ball_height)
        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={
                "contact_ball_position_x": float(env._controller_anchor_position()[0]),
                "contact_ball_position_y": float(env._controller_anchor_position()[1]),
                "contact_ball_position_z": float(env._controller_anchor_position()[2] + 0.02),
                "contact_ball_velocity_x": 1.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 1.0,
            },
        )

        self.assertLess(reward_terms["trajectory_error_penalty"], 0.0)

    def test_return_target_xy_term_applies_only_on_useful_contact(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            useful_contact_return_target_xy_reward_weight=1.25,
            return_target_xy_source="controller_anchor",
            return_target_xy_tolerance=0.1,
        )
        env.reset(ball_height=env.ball_height)
        anchor_xy = env._controller_anchor_position()[:2]
        contact_trace = {
            "contact_ball_position_x": float(anchor_xy[0]),
            "contact_ball_position_y": float(anchor_xy[1]),
            "contact_ball_velocity_x": 0.0,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": 0.8,
        }

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason="useful_keepup_bounce",
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace=contact_trace,
        )
        self.assertAlmostEqual(reward_terms["return_target_xy_term"], 1.25)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace=contact_trace,
        )
        self.assertEqual(reward_terms["return_target_xy_term"], 0.0)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason="useful_keepup_bounce",
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={"contact_ball_velocity_x": 0.0, "contact_ball_velocity_y": 0.0, "contact_ball_velocity_z": 0.8},
        )
        self.assertEqual(reward_terms["return_target_xy_term"], 0.0)

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
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 4.0,
                "contact_racket_velocity_z": 0.2,
                "contact_xy_alignment_error": env.contact_centering_radius - 0.01,
                "contact_ball_height_above_racket": 0.02,
            },
            contact_event=True,
        )
        self.assertEqual(success_reason, "useful_keepup_bounce")

    def test_contact_apex_height_uses_anchor_reference_when_position_is_available(self) -> None:
        env = PingPongKeepUpEnv(
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        gravity = abs(env._gravity_z())
        contact_height_above_anchor = 0.02
        velocity_z = float(np.sqrt(2.0 * gravity * (env.target_ball_height - contact_height_above_anchor)))
        anchor_position = env._controller_anchor_position()
        projected_apex = env._projected_contact_apex_height_above_racket(
            {
                "contact_ball_position_z": float(anchor_position[2] + contact_height_above_anchor),
                "contact_ball_height_above_racket": 0.0,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": velocity_z,
            }
        )

        self.assertAlmostEqual(projected_apex, env.target_ball_height)


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

    def test_gym_wrapper_exposes_position_strike_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(action_mode="position_strike", reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (3,))

    def test_target_ball_height_can_be_lower_than_spawn_height(self) -> None:
        env = PingPongKeepUpEnv(
            ball_height=0.5,
            target_ball_height=0.4,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset()
        self.assertAlmostEqual(env._target_ball_height_above_racket(), 0.4)

    def test_position_contact_frame_mode_exposes_tilt_state(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_contact_frame", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        observation, _ = env.reset()
        self.assertEqual(env.action_size, 5)
        self.assertIn("racket_face_normal", env.observation_slices)
        self.assertIn("target_tilt", env.observation_slices)
        self.assertEqual(observation.shape, (env.observation_size,))

    def test_contact_frame_radial_action_moves_target_toward_anchor(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_contact_frame", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        ball_position = anchor_position + np.array([0.04, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        base_target = env._contact_frame_action_target_position(np.zeros(3, dtype=float))
        radial_target = env._contact_frame_action_target_position(np.array([0.02, 0.0, 0.0], dtype=float))
        self.assertLess(float(radial_target[0]), float(base_target[0]))
        self.assertAlmostEqual(float(radial_target[1]), float(base_target[1]))

    def test_heuristic_policy_supports_contact_frame_mode(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_contact_frame", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        action = HeuristicKeepUpPolicy().predict(env)
        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue(np.all(action <= env.action_high + 1.0e-9))
        self.assertTrue(np.all(action >= env.action_low - 1.0e-9))

    def test_contact_frame_base_strike_lift_makes_zero_action_nonempty(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_base_strike_z_boost=0.024,
            contact_frame_base_strike_z_offset=0.01,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        base_target = env._contact_frame_action_target_position(np.zeros(3, dtype=float))
        no_base_env = PingPongKeepUpEnv(action_mode="position_contact_frame", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        no_base_env.reset(ball_height=no_base_env.ball_height)
        no_base_env.sim.spawn_ball(
            no_base_env.sim.racket_position + np.array([0.0, 0.0, no_base_env._preparation_target_height_above_racket()]),
            velocity=(0.0, 0.0, -1.0),
        )
        no_base_target = no_base_env._contact_frame_action_target_position(np.zeros(3, dtype=float))
        self.assertGreater(float(base_target[2]), float(no_base_target[2]))

    def test_contact_frame_base_tilt_residual_applies_during_strike(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            contact_frame_base_tilt_residual=(-0.02, 0.0),
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))
        self.assertLess(float(info["target_tilt"][0]), -0.01)

    def test_contact_frame_apex_lift_increases_for_slower_descent(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_apex_lift_gain=0.06,
            contact_frame_apex_lift_max=0.03,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -0.2))
        slow_lift = env._contact_frame_apex_lift()
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -2.0))
        fast_lift = env._contact_frame_apex_lift()
        self.assertGreater(slow_lift, fast_lift)

    def test_contact_frame_velocity_lead_tracks_required_impact_velocity(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_velocity_lead_gain=0.05,
            contact_frame_velocity_lead_max=0.03,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -0.2))
        slow_descent_lead = env._contact_frame_velocity_lead()

        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -2.0))
        fast_descent_lead = env._contact_frame_velocity_lead()

        self.assertGreater(slow_descent_lead, 0.0)
        self.assertLess(fast_descent_lead, slow_descent_lead)

    def test_cartesian_controller_velocity_target_changes_joint_targets(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            controller_velocity_gain=1.0,
            controller_max_velocity_step=0.05,
        )
        env.reset(ball_height=env.ball_height)
        current_position = env.sim.racket_position.copy()
        env.controller.set_target_position(current_position)
        env.controller.set_target_velocity((0.0, 0.0, 0.0))
        baseline_targets = env.controller.compute_joint_targets()

        env.controller.set_target_position(current_position)
        env.controller.set_target_velocity((0.0, 0.0, 1.0))
        velocity_targets = env.controller.compute_joint_targets()

        self.assertTrue(np.allclose(env.controller.target_velocity, np.array([0.0, 0.0, 1.0])))
        self.assertGreater(float(np.linalg.norm(velocity_targets - baseline_targets)), 1.0e-6)

    def test_contact_frame_velocity_target_tracks_required_impact_velocity(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_velocity_target_gain=1.0,
            contact_frame_velocity_target_max=2.0,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -0.2))
        env.controller.set_target_tilt((0.0, 0.0))
        slow_descent_target = env._contact_frame_velocity_target()

        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -2.0))
        env.controller.set_target_tilt((0.0, 0.0))
        fast_descent_target = env._contact_frame_velocity_target()

        self.assertGreater(float(slow_descent_target[2]), 0.0)
        self.assertLess(float(fast_descent_target[2]), float(slow_descent_target[2]))

    def test_contact_frame_followthrough_offset_tracks_required_impact_velocity(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_followthrough_gain=1.0,
            contact_frame_followthrough_time=0.06,
            contact_frame_followthrough_max=0.04,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -0.2))
        slow_descent_offset = env._contact_frame_followthrough_offset()

        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -2.0))
        fast_descent_offset = env._contact_frame_followthrough_offset()

        self.assertGreater(float(slow_descent_offset[2]), 0.0)
        self.assertLess(float(fast_descent_offset[2]), float(slow_descent_offset[2]))

    def test_contact_frame_centering_tilt_uses_pitch_and_roll_toward_anchor(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            contact_frame_centering_tilt_limit=(0.03, 0.03),
            contact_frame_centering_tilt_deadband=0.0,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.03, -0.03, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.2, 0.0, -1.0))
        target_tilt = env._contact_frame_centering_tilt()
        self.assertLess(float(target_tilt[0]), -0.01)
        self.assertLess(float(target_tilt[1]), -0.01)

    def test_contact_frame_trajectory_tilt_uses_desired_impulse_direction(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.09, 0.09),
            contact_frame_trajectory_tilt_gain=1.0,
            contact_frame_trajectory_tilt_limit=(0.03, 0.03),
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.03, -0.03, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))

        target_tilt = env._contact_frame_trajectory_tilt()

        self.assertLess(float(target_tilt[0]), -0.01)
        self.assertLess(float(target_tilt[1]), -0.01)

    def test_contact_frame_action_penalty_is_negative_for_nonzero_action(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_action_penalty_weight=0.05,
        )
        env.reset(ball_height=env.ball_height)
        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=False,
            contact_active=False,
            applied_action=env.action_high.copy(),
            contact_trace={},
        )
        self.assertLess(reward_terms["contact_frame_action_penalty"], 0.0)

    def test_gym_wrapper_exposes_position_contact_frame_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(action_mode="position_contact_frame", reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (5,))


if __name__ == "__main__":
    unittest.main()
