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


    def test_tracking_term_stays_active_after_first_contact(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset()
        env.contact_count = 1
        ball_position = env.sim.racket_position + np.array([0.0, 0.0, env._preparation_target_height_above_racket()])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, -1.0))
        self.assertGreater(env._tracking_term(), 0.0)


    def test_strike_guard_reapplies_after_first_contact(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0)
        env.reset(ball_height=env.ball_height, ball_velocity=(0.0, 0.0, 0.0))
        env.contact_count = 1
        anchor_position = env._controller_anchor_position()
        guarded_target = env._guarded_target_position(anchor_position + np.array([0.0, 0.0, 0.12]))
        self.assertLessEqual(guarded_target[2], anchor_position[2] + 0.02 + 1.0e-9)


    def test_gym_wrapper_exposes_box_spaces(self) -> None:
        env = PingPongKeepUpGymEnv(reset_xy_range=0.0)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertEqual(env.action_space.shape, (3,))


if __name__ == "__main__":
    unittest.main()
