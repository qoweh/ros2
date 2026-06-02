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

    def test_scene_path_variant_moves_racket_center_farther_from_hand(self) -> None:
        default_env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        outward_env = PingPongKeepUpEnv(
            scene_path="assets/scene_racket_outward.xml",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )

        default_hand_id = default_env.sim.model.body("hand").id
        outward_hand_id = outward_env.sim.model.body("hand").id
        default_hand_distance = np.linalg.norm(
            default_env.sim.racket_position[:2] - default_env.sim.data.xpos[default_hand_id][:2]
        )
        outward_hand_distance = np.linalg.norm(
            outward_env.sim.racket_position[:2] - outward_env.sim.data.xpos[outward_hand_id][:2]
        )

        self.assertGreater(outward_hand_distance, default_hand_distance + 0.015)
        self.assertTrue(outward_env.training_config()["scene_path"].endswith("scene_racket_outward.xml"))

    def test_step_info_exposes_controller_anchor_position(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))
        self.assertIn("controller_anchor_position", info)
        self.assertIn("contact_ball_position_x", info)
        self.assertIn("contact_ball_position_y", info)
        self.assertTrue(np.allclose(info["controller_anchor_position"], env._controller_anchor_position()))

    def test_step_info_exposes_applied_action_norms(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        action = np.array([0.01, -0.005, 0.02, 0.002, -0.003], dtype=float)

        _, _, _, _, info = env.step(action)

        self.assertTrue(np.allclose(info["applied_action"], action))
        self.assertGreater(float(info["applied_action_norm"]), 0.0)
        self.assertGreater(float(info["applied_action_normalized_norm"]), 0.0)
        self.assertGreater(float(info["applied_position_action_norm"]), 0.0)
        self.assertGreater(float(info["applied_tilt_action_norm"]), 0.0)

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

    def test_success_reason_rejects_apex_overshoot_when_window_required(self) -> None:
        env = PingPongKeepUpEnv(
            target_ball_height=0.25,
            height_tolerance=0.10,
            require_apex_height_window_for_success=True,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        gravity = abs(env._gravity_z())
        contact_height_above_racket = 0.02
        overshoot_apex_height = env.target_ball_height + env.height_tolerance + 0.05
        velocity_z = float(np.sqrt(2.0 * gravity * (overshoot_apex_height - contact_height_above_racket)))

        success_reason = env._success_reason(
            failure_reason=None,
            contact_trace={
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": velocity_z,
                "contact_racket_velocity_z": 0.2,
                "contact_xy_alignment_error": env.contact_centering_radius - 0.01,
                "contact_ball_height_above_racket": contact_height_above_racket,
            },
            contact_event=True,
        )

        self.assertIsNone(success_reason)

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

    def test_desired_outgoing_velocity_targets_next_descending_intercept_by_default(self) -> None:
        env = PingPongKeepUpEnv(
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        contact_position = anchor_position + np.array([0.04, -0.02, env._tracking_strike_plane_offset()])

        desired_velocity, desired_time_to_apex, desired_target_xy = env._desired_outgoing_velocity(contact_position)
        next_time, next_xy = env._predicted_descending_intercept_from_velocity(
            contact_position,
            desired_velocity,
            env._desired_outgoing_target_z(),
        )
        desired_apex_xy = contact_position[:2] + desired_velocity[:2] * desired_time_to_apex

        self.assertIsNotNone(next_time)
        self.assertTrue(np.allclose(desired_target_xy, anchor_position[:2]))
        self.assertTrue(np.allclose(next_xy, anchor_position[:2], atol=1.0e-6))
        self.assertGreater(float(np.linalg.norm(desired_apex_xy - anchor_position[:2])), 1.0e-4)

    def test_keepup_target_xy_offset_shifts_next_intercept_target(self) -> None:
        env = PingPongKeepUpEnv(
            target_ball_height=0.25,
            keepup_target_xy_offset=(0.0, 0.04),
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        target_xy = env._keepup_target_xy()
        contact_position = env._controller_anchor_position() + np.array(
            [0.04, -0.02, env._tracking_strike_plane_offset()]
        )

        desired_velocity, _, desired_target_xy = env._desired_outgoing_velocity(contact_position)
        _, next_xy = env._predicted_descending_intercept_from_velocity(
            contact_position,
            desired_velocity,
            env._desired_outgoing_target_z(),
        )

        self.assertTrue(np.allclose(desired_target_xy, target_xy))
        self.assertTrue(np.allclose(next_xy, target_xy, atol=1.0e-6))

    def test_apex_desired_outgoing_mode_keeps_legacy_apex_xy_target(self) -> None:
        env = PingPongKeepUpEnv(
            target_ball_height=0.25,
            desired_outgoing_xy_mode="apex",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        contact_position = anchor_position + np.array([0.04, 0.0, env._tracking_strike_plane_offset()])

        desired_velocity, desired_time_to_apex, _ = env._desired_outgoing_velocity(contact_position)
        desired_apex_xy = contact_position[:2] + desired_velocity[:2] * desired_time_to_apex
        _, next_xy = env._predicted_descending_intercept_from_velocity(
            contact_position,
            desired_velocity,
            env._desired_outgoing_target_z(),
        )

        self.assertTrue(np.allclose(desired_apex_xy, anchor_position[:2], atol=1.0e-6))
        self.assertGreater(float(np.linalg.norm(next_xy - anchor_position[:2])), 1.0e-3)

    def test_trajectory_metrics_report_next_intercept_error(self) -> None:
        env = PingPongKeepUpEnv(
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        contact_position = anchor_position + np.array([0.03, 0.02, env._tracking_strike_plane_offset()])
        desired_velocity, desired_time_to_apex, desired_target_xy = env._desired_outgoing_velocity(contact_position)
        metrics = env._trajectory_metrics_from_velocity(
            contact_position,
            desired_velocity,
            desired_velocity,
            desired_time_to_apex,
            desired_target_xy,
            env._desired_outgoing_target_z(),
        )

        self.assertAlmostEqual(metrics["predicted_apex_xy_error"], 0.0)
        self.assertAlmostEqual(metrics["predicted_next_intercept_xy_error"], 0.0)

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

    def test_position_contact_frame_velocity_residual_mode_exposes_8d_action_space(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            lateral_action_limit=0.02,
            vertical_action_limit=0.03,
            tilt_action_limit=0.004,
            contact_frame_velocity_scale_action_limit=0.25,
            contact_frame_outgoing_xy_action_limit=0.40,
        )
        observation, _ = env.reset()
        self.assertEqual(env.action_size, 8)
        self.assertTrue(np.allclose(env.action_high, np.array([0.02, 0.02, 0.03, 0.004, 0.004, 0.25, 0.40, 0.40])))
        self.assertIn("racket_face_normal", env.observation_slices)
        self.assertIn("target_tilt", env.observation_slices)
        self.assertEqual(observation.shape, (env.observation_size,))
        config = env.training_config()
        self.assertEqual(config["action_mode"], "position_contact_frame_velocity_residual")
        self.assertAlmostEqual(config["contact_frame_velocity_scale_action_limit"], 0.25)
        self.assertAlmostEqual(config["contact_frame_outgoing_xy_action_limit"], 0.40)

    def test_position_contact_frame_velocity_tilt_residual_mode_exposes_11d_action_space(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            lateral_action_limit=0.02,
            vertical_action_limit=0.03,
            tilt_action_limit=0.006,
            contact_frame_velocity_scale_action_limit=0.25,
            contact_frame_outgoing_xy_action_limit=0.40,
            contact_frame_racket_vz_action_limit=0.50,
            contact_frame_tilt_scale_action_limit=0.60,
            tracking_strike_plane_offset=0.06,
        )
        observation, _ = env.reset()
        expected_high = np.array(
            [0.02, 0.02, 0.03, 0.006, 0.006, 0.25, 0.40, 0.40, 0.50, 0.60, 0.60],
            dtype=float,
        )
        self.assertEqual(env.action_size, 11)
        self.assertTrue(np.allclose(env.action_high, expected_high))
        self.assertAlmostEqual(env._tracking_strike_plane_offset(), 0.06)
        self.assertEqual(observation.shape, (env.observation_size,))
        config = env.training_config()
        self.assertEqual(config["action_mode"], "position_contact_frame_velocity_tilt_residual")
        self.assertAlmostEqual(config["contact_frame_racket_vz_action_limit"], 0.50)
        self.assertAlmostEqual(config["contact_frame_tilt_scale_action_limit"], 0.60)
        self.assertAlmostEqual(config["tracking_strike_plane_offset"], 0.06)

    def test_position_contact_frame_velocity_tilt_lateral_residual_mode_exposes_13d_action_space(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_lateral_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            lateral_action_limit=0.02,
            vertical_action_limit=0.03,
            tilt_action_limit=0.008,
            contact_frame_velocity_scale_action_limit=0.25,
            contact_frame_outgoing_xy_action_limit=0.40,
            contact_frame_racket_vz_action_limit=0.50,
            contact_frame_tilt_scale_action_limit=0.60,
            contact_frame_racket_xy_action_limit=0.30,
        )
        observation, _ = env.reset()
        expected_high = np.array(
            [0.02, 0.02, 0.03, 0.008, 0.008, 0.25, 0.40, 0.40, 0.50, 0.60, 0.60, 0.30, 0.30],
            dtype=float,
        )
        self.assertEqual(env.action_size, 13)
        self.assertTrue(np.allclose(env.action_high, expected_high))
        self.assertEqual(observation.shape, (env.observation_size,))
        config = env.training_config()
        self.assertEqual(config["action_mode"], "position_contact_frame_velocity_tilt_lateral_residual")
        self.assertAlmostEqual(config["contact_frame_racket_xy_action_limit"], 0.30)

    def test_contact_frame_velocity_residual_changes_controller_desired_velocity_only(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_residual",
            target_ball_height=0.30,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        contact_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(contact_position, velocity=(0.0, 0.0, -1.0))

        base_desired_velocity, _, _ = env._contact_frame_planned_desired_velocity(contact_position)
        zero_residual_velocity, _, _ = env._contact_frame_controller_desired_velocity(contact_position)
        self.assertTrue(np.allclose(zero_residual_velocity, base_desired_velocity))

        env._contact_frame_velocity_residual_action = np.array([0.20, 0.30, -0.10], dtype=float)
        residual_velocity, _, _ = env._contact_frame_controller_desired_velocity(contact_position)
        unchanged_base_desired_velocity, _, _ = env._contact_frame_planned_desired_velocity(contact_position)

        self.assertTrue(np.allclose(unchanged_base_desired_velocity, base_desired_velocity))
        self.assertAlmostEqual(float(residual_velocity[0]), float(base_desired_velocity[0] + 0.30))
        self.assertAlmostEqual(float(residual_velocity[1]), float(base_desired_velocity[1] - 0.10))
        self.assertAlmostEqual(float(residual_velocity[2]), float(base_desired_velocity[2] * 1.20))

    def test_contact_frame_velocity_residual_step_info_exposes_residuals(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        action = np.zeros(env.action_size, dtype=float)
        action[5:] = np.array([0.20, 0.10, -0.12], dtype=float)

        _, _, _, _, info = env.step(action)

        self.assertTrue(np.allclose(info["contact_frame_velocity_residual_action"], action[5:]))
        self.assertAlmostEqual(float(info["contact_frame_vz_scale_action"]), 0.20)
        self.assertAlmostEqual(float(info["contact_frame_vz_scale"]), 1.20)
        self.assertAlmostEqual(float(info["contact_frame_outgoing_x_residual_action"]), 0.10)
        self.assertAlmostEqual(float(info["contact_frame_outgoing_y_residual_action"]), -0.12)
        self.assertIsNotNone(info["contact_frame_controller_desired_velocity"])

    def test_contact_frame_racket_vz_residual_changes_velocity_target_directly(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_velocity_target_gain=0.0,
            contact_frame_velocity_target_max=2.0,
            contact_frame_racket_vz_action_limit=0.5,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))

        baseline_target = env._contact_frame_velocity_target()
        env._contact_frame_racket_vz_residual_action = 0.35
        residual_target = env._contact_frame_velocity_target()

        self.assertAlmostEqual(float(baseline_target[2]), 0.0)
        self.assertAlmostEqual(float(residual_target[2]), 0.35)

    def test_contact_frame_racket_xy_residual_changes_velocity_target_directly(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_lateral_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_velocity_target_gain=0.0,
            contact_frame_velocity_target_max=2.0,
            contact_frame_racket_xy_action_limit=0.4,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))

        baseline_target = env._contact_frame_velocity_target()
        env._contact_frame_racket_xy_residual_action = np.array([-0.25, 0.15], dtype=float)
        residual_target = env._contact_frame_velocity_target()

        self.assertTrue(np.allclose(baseline_target[:2], np.zeros(2, dtype=float)))
        self.assertAlmostEqual(float(residual_target[0]), -0.25)
        self.assertAlmostEqual(float(residual_target[1]), 0.15)

    def test_contact_frame_velocity_tilt_residual_step_info_exposes_extra_residuals(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        action = np.zeros(env.action_size, dtype=float)
        action[5:] = np.array([0.10, -0.12, 0.08, 0.20, 0.30, -0.25], dtype=float)

        _, _, _, _, info = env.step(action)

        self.assertTrue(np.allclose(info["contact_frame_velocity_residual_action"], action[5:8]))
        self.assertAlmostEqual(float(info["contact_frame_racket_vz_residual_action"]), 0.20)
        self.assertAlmostEqual(float(info["contact_frame_trajectory_tilt_scale"]), 1.30)
        self.assertAlmostEqual(float(info["contact_frame_centering_tilt_scale"]), 0.75)
        self.assertTrue(np.allclose(info["contact_frame_tilt_scale_residual_action"], action[9:11]))

    def test_contact_frame_velocity_tilt_lateral_residual_step_info_exposes_xy_residuals(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_lateral_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        action = np.zeros(env.action_size, dtype=float)
        action[11:] = np.array([-0.18, 0.11], dtype=float)

        _, _, _, _, info = env.step(action)

        self.assertTrue(np.allclose(info["contact_frame_racket_xy_residual_action"], action[11:13]))
        self.assertAlmostEqual(float(info["contact_frame_racket_x_residual_action"]), -0.18)
        self.assertAlmostEqual(float(info["contact_frame_racket_y_residual_action"]), 0.11)

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

    def test_heuristic_policy_supports_contact_frame_velocity_residual_mode(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        action = HeuristicKeepUpPolicy().predict(env)
        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue(np.all(action <= env.action_high + 1.0e-9))
        self.assertTrue(np.all(action >= env.action_low - 1.0e-9))
        self.assertTrue(np.allclose(action[5:], np.zeros(3, dtype=float)))

    def test_heuristic_policy_supports_contact_frame_velocity_tilt_residual_mode(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        action = HeuristicKeepUpPolicy().predict(env)
        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue(np.all(action <= env.action_high + 1.0e-9))
        self.assertTrue(np.all(action >= env.action_low - 1.0e-9))
        self.assertTrue(np.allclose(action[5:], np.zeros(6, dtype=float)))

    def test_heuristic_policy_supports_contact_frame_velocity_tilt_lateral_residual_mode(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame_velocity_tilt_lateral_residual",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        action = HeuristicKeepUpPolicy().predict(env)
        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue(np.all(action <= env.action_high + 1.0e-9))
        self.assertTrue(np.all(action >= env.action_low - 1.0e-9))
        self.assertTrue(np.allclose(action[5:], np.zeros(8, dtype=float)))

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

    def test_contact_frame_post_contact_return_z_offset_lowers_racket_target(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            post_contact_return_z_offset=-0.03,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        env.sim.spawn_ball(anchor_position + np.array([0.0, 0.0, 0.08]), velocity=(0.0, 0.0, 1.0))

        target = env._contact_frame_action_target_position(np.zeros(3, dtype=float))

        self.assertAlmostEqual(float(target[2]), float(anchor_position[2] - 0.03))

    def test_post_contact_return_can_avoid_chasing_ball_during_rise(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            post_contact_return_assist_weight=0.8,
            post_contact_return_predict_during_rise=False,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        env.successful_bounce_count = 1
        env.sim.spawn_ball(anchor_position + np.array([0.08, 0.0, 0.08]), velocity=(0.5, 0.0, 1.0))

        target = env._contact_frame_action_target_position(np.zeros(3, dtype=float))

        self.assertAlmostEqual(float(target[0]), float(anchor_position[0]))
        self.assertAlmostEqual(float(target[1]), float(anchor_position[1]))

    def test_next_intercept_success_radius_can_be_sweet_spot_strict(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            next_intercept_success_radius=0.04,
            easy_next_ball_xy_radius=0.04,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        env.sim.spawn_ball(anchor_position + np.array([0.05, 0.0, 0.10]), velocity=(0.0, 0.0, -1.0))

        metrics = env._next_intercept_metrics()

        self.assertFalse(metrics["reachable"])
        self.assertGreater(float(metrics["info_xy_error"]), env.next_intercept_success_radius)

    def test_controller_nullspace_options_are_recorded(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            controller_nullspace_posture_gain=0.2,
            controller_nullspace_posture_max_step=0.01,
            controller_body_clearance_gain=0.5,
            controller_body_clearance_margin=0.14,
            controller_body_clearance_max_step=0.01,
            controller_body_clearance_body_names=("link5", "link6"),
        )

        config = env.training_config()

        self.assertAlmostEqual(config["controller_nullspace_posture_gain"], 0.2)
        self.assertEqual(config["controller_body_clearance_body_names"], ["link5", "link6"])

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

    def test_contact_frame_low_apex_recovery_only_runs_on_followup_descent(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            height_tolerance=0.10,
            contact_frame_base_strike_z_boost=0.0,
            contact_frame_base_strike_z_offset=0.0,
            contact_frame_apex_lift_gain=0.0,
            contact_frame_velocity_lead_gain=0.0,
            contact_frame_low_apex_recovery_lift_gain=0.10,
            contact_frame_low_apex_recovery_lift_max=0.04,
            contact_frame_low_apex_recovery_velocity_gain=0.80,
            contact_frame_low_apex_recovery_velocity_max=0.30,
        )
        env.reset(ball_height=env.ball_height)
        config = env.training_config()
        self.assertAlmostEqual(config["contact_frame_low_apex_recovery_lift_gain"], 0.10)
        self.assertAlmostEqual(config["contact_frame_low_apex_recovery_velocity_max"], 0.30)
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env._last_contact_apex_shortfall = 0.15

        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        recovery_lift = env._contact_frame_low_apex_recovery_lift()
        recovery_velocity = env._contact_frame_low_apex_recovery_velocity()
        base_lift = env._contact_frame_base_strike_lift()

        self.assertGreater(recovery_lift, 0.0)
        self.assertLessEqual(recovery_lift, env.contact_frame_low_apex_recovery_lift_max)
        self.assertGreater(recovery_velocity, 0.0)
        self.assertLessEqual(recovery_velocity, env.contact_frame_low_apex_recovery_velocity_max)
        self.assertAlmostEqual(base_lift, recovery_lift)

        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, 0.2))
        self.assertEqual(env._contact_frame_low_apex_recovery_lift(), 0.0)
        self.assertEqual(env._contact_frame_low_apex_recovery_velocity(), 0.0)

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

    def test_contact_frame_intercept_velocity_targets_contact_position(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_intercept_velocity_gain=1.0,
            contact_frame_intercept_velocity_max=2.0,
        )
        env.reset(ball_height=env.ball_height)
        current_position = env.sim.racket_position.copy()
        ball_position = current_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        target_position = current_position + np.array([0.04, 0.0, 0.0])

        target_velocity = env._contact_frame_velocity_target(target_position)

        self.assertGreater(float(target_velocity[0]), 0.0)
        self.assertAlmostEqual(float(target_velocity[1]), 0.0, places=6)

    def test_contact_frame_intercept_velocity_is_inactive_while_ball_rises(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_intercept_velocity_gain=1.0,
            contact_frame_intercept_velocity_max=2.0,
        )
        env.reset(ball_height=env.ball_height)
        current_position = env.sim.racket_position.copy()
        ball_position = current_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, 1.0))

        target_velocity = env._contact_frame_velocity_target(current_position + np.array([0.04, 0.0, 0.0]))

        self.assertTrue(np.allclose(target_velocity, np.zeros(3, dtype=float)))

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

    def test_contact_frame_trajectory_tilt_remains_active_for_followup_recovery_strike(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_tilt_limit=(0.06, 0.06),
            contact_frame_trajectory_tilt_gain=1.0,
            contact_frame_trajectory_tilt_limit=(0.0, 0.03),
        )
        env.reset(ball_height=env.ball_height)
        env.successful_bounce_count = 1
        env._last_contact_step = env.step_count
        ball_position = env.sim.racket_position + np.array([0.0, 0.05, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))

        target_tilt = env._contact_frame_trajectory_tilt()

        self.assertEqual(env._phase_name(), "recovery")
        self.assertAlmostEqual(float(target_tilt[0]), 0.0)
        self.assertGreater(float(target_tilt[1]), 0.01)

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

    def test_contact_frame_planner_fixes_next_target_near_anchor(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            target_ball_height=0.25,
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            keepup_target_xy_offset=(0.01, -0.02),
            contact_frame_planner_enabled=True,
            contact_frame_planner_hold_during_descent=True,
        )
        env.reset(ball_height=env.ball_height)
        anchor_target_xy = env._keepup_target_xy().copy()
        ball_position = env._controller_anchor_position() + np.array([0.04, -0.03, 0.20])
        env.sim.spawn_ball(ball_position, velocity=(0.15, 0.0, -1.0))

        env._update_contact_frame_plan()

        self.assertTrue(env._contact_frame_plan_active)
        np.testing.assert_allclose(env._contact_frame_plan_target_xy, anchor_target_xy)
        self.assertGreater(float(env._contact_frame_plan_desired_velocity[2]), 0.0)

        env.keepup_target_xy_offset = np.array([0.04, 0.04], dtype=float)
        env._update_contact_frame_plan()

        np.testing.assert_allclose(env._contact_frame_plan_target_xy, anchor_target_xy)

    def test_contact_frame_planner_base_target_uses_planned_contact_position(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_planner_enabled=True,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env._controller_anchor_position() + np.array([0.05, -0.02, 0.20])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        env._update_contact_frame_plan()

        target_position = env._contact_frame_action_target_position(np.zeros(3, dtype=float))

        self.assertTrue(env._contact_frame_plan_active)
        np.testing.assert_allclose(target_position[:2], env._contact_frame_plan_contact_position[:2])

    def test_contact_frame_strike_hold_freezes_final_contact_position(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_planner_enabled=True,
            contact_frame_strike_hold_time=0.20,
            contact_frame_strike_hold_min_readiness=0.0,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        ball_position = anchor_position + np.array([0.02, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))

        env._update_contact_frame_plan()
        held_position = env._contact_frame_plan_contact_position.copy()

        env.sim.spawn_ball(ball_position + np.array([0.04, 0.0, 0.0]), velocity=(0.0, 0.0, -1.0))
        env._update_contact_frame_plan()

        self.assertTrue(env._contact_frame_strike_hold_active)
        np.testing.assert_allclose(env._contact_frame_plan_contact_position, held_position)

    def test_contact_frame_strike_hold_suppresses_lateral_intercept_velocity(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_frame_intercept_velocity_gain=1.0,
            contact_frame_intercept_velocity_max=2.0,
            contact_frame_velocity_target_gain=0.0,
            contact_frame_strike_hold_time=0.10,
        )
        env.reset(ball_height=env.ball_height)
        env._contact_frame_strike_hold_active = True
        target_velocity = env._contact_frame_velocity_target(env.sim.racket_position + np.array([0.04, 0.0, 0.0]))

        self.assertTrue(np.allclose(target_velocity, np.zeros(3, dtype=float)))

    def test_body_clearance_active_after_any_recent_contact_while_ball_is_close(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            controller_body_clearance_gain=0.75,
            controller_body_clearance_margin=0.14,
            controller_body_clearance_max_step=0.018,
        )
        env.reset(ball_height=env.ball_height)
        env.contact_count = 1
        env._last_contact_step = env.step_count
        env.sim.spawn_ball(
            env.sim.racket_position + np.array([0.0, 0.0, env.target_ball_height]),
            velocity=(0.0, 0.0, 0.8),
        )

        self.assertTrue(env._controller_body_clearance_active())

    def test_success_reason_rejects_side_sweeping_contact_when_lateral_speed_limited(self) -> None:
        env = PingPongKeepUpEnv(
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            max_contact_racket_lateral_speed_for_success=0.2,
        )
        env.reset(ball_height=env.ball_height)
        success_reason = env._success_reason(
            failure_reason=None,
            contact_trace={
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 4.0,
                "contact_racket_velocity_x": 0.3,
                "contact_racket_velocity_y": 0.0,
                "contact_racket_velocity_z": 0.2,
                "contact_xy_alignment_error": env.contact_centering_radius - 0.01,
                "contact_ball_height_above_racket": 0.02,
            },
            contact_event=True,
        )

        self.assertIsNone(success_reason)

    def test_contact_racket_lateral_velocity_penalty_applies_on_sweeping_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_racket_lateral_velocity_penalty_weight=0.5,
            contact_racket_lateral_velocity_tolerance=0.1,
        )
        env.reset(ball_height=env.ball_height)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={
                "contact_ball_velocity_z": 1.0,
                "contact_racket_velocity_x": 0.2,
                "contact_racket_velocity_y": 0.0,
            },
        )

        self.assertLess(reward_terms["contact_racket_lateral_velocity_penalty"], 0.0)

    def test_contact_quality_penalties_apply_to_bad_upward_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            next_intercept_success_radius=0.04,
            next_intercept_xy_error_penalty_weight=0.5,
            post_contact_lateral_velocity_penalty_weight=0.5,
            contact_xy_error_penalty_weight=0.5,
            nonuseful_contact_penalty_weight=1.0,
            contact_apex_under_target_penalty_weight=0.5,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        env.reset(ball_height=env.ball_height)
        ball_position = env.sim.racket_position + np.array([0.08, 0.0, 0.03])
        env.sim.spawn_ball(ball_position, velocity=(0.6, 0.0, 1.0))
        contact_trace = {
            "contact_ball_position_x": float(ball_position[0]),
            "contact_ball_position_y": float(ball_position[1]),
            "contact_ball_position_z": float(ball_position[2]),
            "contact_ball_velocity_x": 0.6,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": 1.0,
            "contact_xy_alignment_error": 0.08,
        }
        outgoing_metrics = {
            "actual_outgoing_velocity_x": 0.6,
            "actual_outgoing_velocity_y": 0.0,
            "actual_outgoing_velocity_z": 1.0,
        }

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace=contact_trace,
            outgoing_trajectory_metrics=outgoing_metrics,
        )

        self.assertLess(reward_terms["next_intercept_xy_error_penalty"], 0.0)
        self.assertLess(reward_terms["post_contact_lateral_velocity_penalty"], 0.0)
        self.assertLess(reward_terms["contact_xy_error_penalty"], 0.0)
        self.assertEqual(reward_terms["nonuseful_contact_penalty"], -1.0)
        self.assertLess(reward_terms["contact_apex_under_target_penalty"], 0.0)

    def test_contact_apex_progress_reward_scales_with_projected_height(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_apex_progress_reward_weight=0.8,
            target_ball_height=0.30,
        )
        env.reset(ball_height=env.ball_height)

        low_reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={
                "contact_ball_height_above_racket": 0.02,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 1.0,
            },
            outgoing_trajectory_metrics={"actual_outgoing_velocity_z": 1.0},
        )
        target_or_higher_reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={
                "contact_ball_height_above_racket": 0.02,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 3.0,
            },
            outgoing_trajectory_metrics={"actual_outgoing_velocity_z": 3.0},
        )

        self.assertGreater(low_reward_terms["contact_apex_progress_term"], 0.0)
        self.assertLess(low_reward_terms["contact_apex_progress_term"], 0.8)
        self.assertAlmostEqual(target_or_higher_reward_terms["contact_apex_progress_term"], 0.8)

    def test_contact_apex_progress_can_be_gated_by_easy_next_ball(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            gate_contact_apex_progress_by_easy_next_ball=True,
            contact_apex_progress_min_easy_next_ball_score=0.50,
        )
        env.reset(ball_height=env.ball_height)
        config = env.training_config()

        self.assertTrue(config["gate_contact_apex_progress_by_easy_next_ball"])
        self.assertAlmostEqual(config["contact_apex_progress_min_easy_next_ball_score"], 0.50)
        self.assertEqual(env._contact_apex_progress_easy_next_ball_gate({"easy_next_ball_score": 0.40}), 0.0)
        self.assertAlmostEqual(
            env._contact_apex_progress_easy_next_ball_gate({"easy_next_ball_score": 0.75}),
            0.75,
        )
        self.assertAlmostEqual(
            env._contact_apex_progress_easy_next_ball_gate({"easy_next_ball_score": 1.40}),
            1.0,
        )

    def test_contact_apex_recovery_progress_rewards_improvement_after_low_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_apex_recovery_progress_reward_weight=0.7,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        env.reset(ball_height=env.ball_height)
        env._last_projected_contact_apex_height = 0.12

        improved_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={
                "contact_ball_height_above_racket": 0.02,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 1.6,
            },
            outgoing_trajectory_metrics={"actual_outgoing_velocity_z": 1.6},
        )
        worse_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={
                "contact_ball_height_above_racket": 0.02,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 0.8,
            },
            outgoing_trajectory_metrics={"actual_outgoing_velocity_z": 0.8},
        )
        env._last_projected_contact_apex_height = 0.31
        already_recovered_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={
                "contact_ball_height_above_racket": 0.02,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": 1.6,
            },
            outgoing_trajectory_metrics={"actual_outgoing_velocity_z": 1.6},
        )

        self.assertGreater(improved_terms["contact_apex_recovery_progress_term"], 0.0)
        self.assertEqual(worse_terms["contact_apex_recovery_progress_term"], 0.0)
        self.assertEqual(already_recovered_terms["contact_apex_recovery_progress_term"], 0.0)

    def test_contact_apex_potential_rewards_improvement_and_penalizes_drops(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_apex_potential_reward_weight=0.5,
            contact_apex_potential_gamma=0.99,
            contact_apex_potential_cap=2.0,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        env.reset(ball_height=env.ball_height)

        def contact_trace_for_apex(apex_height: float) -> dict[str, float]:
            contact_height = 0.02
            velocity_z = float(np.sqrt(2.0 * abs(env._gravity_z()) * max(apex_height - contact_height, 0.0)))
            return {
                "contact_ball_height_above_racket": contact_height,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": velocity_z,
            }

        env._last_projected_contact_apex_height = 0.12
        improved_term = env._contact_apex_potential_term(contact_trace_for_apex(0.22))
        env._last_projected_contact_apex_height = 0.25
        dropped_term = env._contact_apex_potential_term(contact_trace_for_apex(0.16))

        self.assertGreater(improved_term, 0.0)
        self.assertLess(dropped_term, 0.0)

    def test_lateral_stability_reward_can_require_minimum_apex_ratio(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_lateral_stability_reward_weight=0.5,
            contact_lateral_stability_speed_tolerance=0.25,
            contact_lateral_stability_xy_tolerance=0.08,
            contact_lateral_stability_min_apex_ratio=0.85,
            target_ball_height=0.30,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()

        def centered_trace_for_apex(apex_height: float) -> dict[str, float]:
            contact_height = 0.02
            velocity_z = float(np.sqrt(2.0 * abs(env._gravity_z()) * max(apex_height - contact_height, 0.0)))
            return {
                "contact_ball_position_x": float(anchor_position[0]),
                "contact_ball_position_y": float(anchor_position[1]),
                "contact_ball_height_above_racket": contact_height,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": velocity_z,
            }

        self.assertEqual(env._contact_lateral_stability_term(centered_trace_for_apex(0.24)), 0.0)
        self.assertGreater(env._contact_lateral_stability_term(centered_trace_for_apex(0.27)), 0.3)

    def test_stable_contact_reward_can_require_minimum_apex_ratio(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            stable_contact_reward_weight=1.0,
            stable_contact_min_apex_ratio=0.90,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        env.reset(ball_height=env.ball_height)

        def contact_trace_for_apex(apex_height: float) -> dict[str, float]:
            contact_height = 0.02
            velocity_z = float(np.sqrt(2.0 * abs(env._gravity_z()) * max(apex_height - contact_height, 0.0)))
            return {
                "contact_ball_height_above_racket": contact_height,
                "contact_ball_velocity_x": 0.0,
                "contact_ball_velocity_y": 0.0,
                "contact_ball_velocity_z": velocity_z,
            }

        next_intercept_metrics = {"easy_next_ball_score": 1.0}

        self.assertEqual(env._stable_contact_term(contact_trace_for_apex(0.25), next_intercept_metrics), 0.0)
        self.assertGreater(env._stable_contact_term(contact_trace_for_apex(0.30), next_intercept_metrics), 0.9)

    def test_contact_lateral_stability_rewards_centered_vertical_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            contact_lateral_stability_reward_weight=0.5,
            contact_lateral_stability_speed_tolerance=0.25,
            contact_lateral_stability_xy_tolerance=0.08,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        centered_trace = {
            "contact_ball_position_x": float(anchor_position[0]),
            "contact_ball_position_y": float(anchor_position[1]),
            "contact_ball_velocity_x": 0.02,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": 1.6,
        }
        sweeping_trace = {
            **centered_trace,
            "contact_ball_velocity_x": 0.50,
        }
        off_center_trace = {
            **centered_trace,
            "contact_ball_position_x": float(anchor_position[0] + 0.20),
        }

        self.assertGreater(env._contact_lateral_stability_term(centered_trace), 0.3)
        self.assertEqual(env._contact_lateral_stability_term(sweeping_trace), 0.0)
        self.assertEqual(env._contact_lateral_stability_term(off_center_trace), 0.0)

    def test_stable_contact_reward_requires_target_apex_and_easy_next_ball(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            stable_contact_reward_weight=1.6,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        env.reset(ball_height=env.ball_height)
        target_velocity_z = float(np.sqrt(2.0 * abs(env._gravity_z()) * (0.30 - 0.02)))
        good_contact_trace = {
            "contact_ball_height_above_racket": 0.02,
            "contact_ball_velocity_x": 0.0,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": target_velocity_z,
        }
        low_contact_trace = {
            "contact_ball_height_above_racket": 0.02,
            "contact_ball_velocity_x": 0.0,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": 1.0,
        }
        next_intercept_metrics = {"easy_next_ball_score": 0.5}

        self.assertAlmostEqual(
            env._stable_contact_term(good_contact_trace, next_intercept_metrics),
            0.8,
        )
        self.assertEqual(env._stable_contact_term(low_contact_trace, next_intercept_metrics), 0.0)

    def test_stable_cycle_observed_requires_useful_height_and_easy_next_intercept(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            stable_cycle_min_easy_next_ball_score=0.45,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        env.reset(ball_height=env.ball_height)
        target_velocity_z = float(np.sqrt(2.0 * abs(env._gravity_z()) * 0.30))
        contact_trace = {
            "contact_ball_height_above_racket": 0.0,
            "contact_ball_velocity_x": 0.0,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": target_velocity_z,
        }

        self.assertTrue(
            env._stable_cycle_observed(
                success_reason="useful_keepup_bounce",
                contact_event=True,
                contact_trace=contact_trace,
                next_intercept_metrics={"reachable": True, "easy_next_ball_score": 0.50},
            )
        )
        self.assertFalse(
            env._stable_cycle_observed(
                success_reason="useful_keepup_bounce",
                contact_event=True,
                contact_trace=contact_trace,
                next_intercept_metrics={"reachable": True, "easy_next_ball_score": 0.40},
            )
        )
        self.assertFalse(
            env._stable_cycle_observed(
                success_reason=None,
                contact_event=True,
                contact_trace=contact_trace,
                next_intercept_metrics={"reachable": True, "easy_next_ball_score": 0.50},
            )
        )

    def test_stable_cycle_state_and_reward_scale_with_streak(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            stable_cycle_reward_weight=0.8,
            stable_cycle_reward_cap=3,
        )
        env.reset(ball_height=env.ball_height)

        env._update_stable_cycle_state(contact_event=True, stable_cycle_observed=True)
        self.assertEqual(env.stable_cycle_count, 1)
        self.assertEqual(env._consecutive_stable_cycle_count, 1)
        self.assertAlmostEqual(
            env._stable_cycle_term(stable_cycle_observed=True, consecutive_stable_cycle_count=1),
            0.8,
        )

        env._update_stable_cycle_state(contact_event=True, stable_cycle_observed=True)
        self.assertEqual(env.stable_cycle_count, 2)
        self.assertEqual(env._consecutive_stable_cycle_count, 2)
        self.assertAlmostEqual(
            env._stable_cycle_term(stable_cycle_observed=True, consecutive_stable_cycle_count=2),
            1.2,
        )

        env._update_stable_cycle_state(contact_event=True, stable_cycle_observed=False)
        self.assertEqual(env.stable_cycle_count, 2)
        self.assertEqual(env._consecutive_stable_cycle_count, 0)
        self.assertEqual(env._stable_cycle_term(stable_cycle_observed=False), 0.0)

    def test_apex_gate_removes_easy_next_ball_reward_from_low_nonuseful_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            easy_next_ball_reward_weight=1.0,
            reward_contact_quality_on_any_upward_contact=True,
            gate_nonuseful_easy_next_ball_by_apex=True,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        ungated_env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            easy_next_ball_reward_weight=1.0,
            reward_contact_quality_on_any_upward_contact=True,
            gate_nonuseful_easy_next_ball_by_apex=False,
            target_ball_height=0.30,
            height_tolerance=0.10,
        )
        contact_trace = {
            "contact_ball_height_above_racket": 0.02,
            "contact_ball_velocity_x": 0.0,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": 1.6,
        }
        outgoing_metrics = {"actual_outgoing_velocity_z": 1.6}
        for current_env in (env, ungated_env):
            current_env.reset(ball_height=current_env.ball_height)
            ball_position = current_env.sim.racket_position + np.array([0.0, 0.0, 0.02])
            current_env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, 1.6))

        gated_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace=contact_trace,
            outgoing_trajectory_metrics=outgoing_metrics,
        )
        ungated_terms = ungated_env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(ungated_env.action_size, dtype=float),
            contact_trace=contact_trace,
            outgoing_trajectory_metrics=outgoing_metrics,
        )

        self.assertEqual(gated_terms["easy_next_ball_term"], 0.0)
        self.assertGreater(ungated_terms["easy_next_ball_term"], 0.0)

    def test_nonuseful_contact_penalty_applies_to_weak_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            nonuseful_contact_penalty_weight=1.0,
        )
        env.reset(ball_height=env.ball_height)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=False,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={"contact_ball_velocity_z": 0.1},
        )

        self.assertEqual(reward_terms["nonuseful_contact_penalty"], -1.0)

    def test_low_apex_contact_failure_only_catches_under_target_upward_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_contact_frame",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            target_ball_height=0.30,
            height_tolerance=0.10,
            terminate_on_low_apex_contact=True,
        )
        env.reset(ball_height=env.ball_height)
        anchor_position = env._controller_anchor_position()
        low_contact_trace = {
            "contact_ball_position_z": float(anchor_position[2] + 0.02),
            "contact_ball_velocity_x": 0.0,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": 1.0,
        }
        high_contact_trace = {
            "contact_ball_position_z": float(anchor_position[2] + 0.02),
            "contact_ball_velocity_x": 0.0,
            "contact_ball_velocity_y": 0.0,
            "contact_ball_velocity_z": 2.0,
        }

        self.assertTrue(
            env._is_low_apex_contact(
                low_contact_trace,
                {"actual_outgoing_velocity_z": 1.0},
                contact_event=True,
                success_reason=None,
            )
        )
        self.assertFalse(
            env._is_low_apex_contact(
                high_contact_trace,
                {"actual_outgoing_velocity_z": 2.0},
                contact_event=True,
                success_reason=None,
            )
        )

    def test_gym_wrapper_exposes_position_contact_frame_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(action_mode="position_contact_frame", reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (5,))

    def test_gym_wrapper_exposes_contact_frame_velocity_residual_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(action_mode="position_contact_frame_velocity_residual", reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (8,))

    def test_gym_wrapper_exposes_contact_frame_velocity_tilt_residual_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(action_mode="position_contact_frame_velocity_tilt_residual", reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (11,))

    def test_gym_wrapper_exposes_contact_frame_velocity_tilt_lateral_residual_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(
            action_mode="position_contact_frame_velocity_tilt_lateral_residual",
            reset_xy_range=0.0,
        )
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (13,))


if __name__ == "__main__":
    unittest.main()
