from __future__ import annotations

import unittest

import numpy as np

from pingpong_rl2.controllers import HeuristicKeepUpPolicy
from pingpong_rl2.envs import PingPongKeepUpEnv


class PingPongKeepUpContractFeatureTests(unittest.TestCase):
    def test_contract_observation_flags_are_opt_in(self) -> None:
        env = PingPongKeepUpEnv(reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        self.assertNotIn("phase_one_hot", env.observation_slices)
        self.assertNotIn("time_since_contact", env.observation_slices)
        self.assertNotIn("next_intercept_time", env.observation_slices)

    def test_contract_observation_slices_are_present_when_enabled(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            include_task_phase_observation=True,
            include_contact_context_observation=True,
            include_next_intercept_observation=True,
        )
        observation, _ = env.reset(ball_height=env.ball_height)
        self.assertIn("phase_one_hot", env.observation_slices)
        self.assertIn("time_since_contact", env.observation_slices)
        self.assertIn("next_intercept_time", env.observation_slices)
        self.assertEqual(observation[env.observation_slices["phase_one_hot"]].shape, (4,))
        self.assertAlmostEqual(float(observation[env.observation_slices["time_since_contact"]][0]), 0.0)

    def test_next_intercept_observation_reports_reachable_descending_return(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            include_next_intercept_observation=True,
        )
        env.reset(ball_height=env.ball_height)
        env._last_contact_step = 0
        env.step_count = 1
        ball_position = env._controller_anchor_position() + np.array([0.0, 0.0, 0.10])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, 1.0))
        observation = env.observation()
        self.assertGreater(float(observation[env.observation_slices["next_intercept_time"]][0]), 0.0)
        self.assertAlmostEqual(float(observation[env.observation_slices["next_intercept_reachable"]][0]), 1.0)

    def test_phase_name_switches_to_return_shaping_after_recent_contact(self) -> None:
        env = PingPongKeepUpEnv(action_mode="position_strike", reset_xy_range=0.0, reset_velocity_xy_range=0.0)
        env.reset(ball_height=env.ball_height)
        env._last_contact_step = 5
        env.step_count = 5
        env.sim.spawn_ball(env.sim.ball_position, velocity=(0.0, 0.0, 0.8))
        self.assertEqual(env._phase_name(), "return_shaping")

    def test_next_intercept_reachable_bonus_applies_only_on_useful_contact(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            next_intercept_reachable_bonus_weight=0.2,
            easy_next_ball_reward_weight=0.1,
        )
        env.reset(ball_height=env.ball_height)
        env._last_contact_step = 0
        env.step_count = 1
        ball_position = env._controller_anchor_position() + np.array([0.0, 0.0, 0.10])
        env.sim.spawn_ball(ball_position, velocity=(0.0, 0.0, 1.0))
        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason="useful_keepup_bounce",
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={},
        )
        self.assertAlmostEqual(reward_terms["next_intercept_reachable_bonus"], 0.2)
        self.assertGreater(reward_terms["easy_next_ball_term"], 0.0)

        reward_terms = env._reward_terms(
            failure_reason=None,
            success_reason=None,
            contact_event=True,
            contact_active=True,
            applied_action=np.zeros(env.action_size, dtype=float),
            contact_trace={},
        )
        self.assertEqual(reward_terms["next_intercept_reachable_bonus"], 0.0)
        self.assertEqual(reward_terms["easy_next_ball_term"], 0.0)

    def test_heuristic_policy_returns_clipped_action(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
        )
        env.reset(ball_height=env.ball_height)
        policy = HeuristicKeepUpPolicy()
        action = policy.predict(env)
        self.assertEqual(action.shape, (env.action_size,))
        self.assertTrue(np.all(action <= env.action_high + 1.0e-9))
        self.assertTrue(np.all(action >= env.action_low - 1.0e-9))

    def test_heuristic_policy_biases_post_contact_return_before_first_success(self) -> None:
        env = PingPongKeepUpEnv(
            action_mode="position_strike",
            reset_xy_range=0.0,
            reset_velocity_xy_range=0.0,
            post_contact_return_assist_weight=0.0,
        )
        env.reset(ball_height=env.ball_height)
        env._last_contact_step = 0
        env.step_count = 1
        ball_position = env._controller_anchor_position() + np.array([0.0, 0.0, 0.10])
        env.sim.spawn_ball(ball_position, velocity=(0.3, 0.0, 1.0))
        policy = HeuristicKeepUpPolicy(return_blend=0.8, recovery_blend=0.6)
        action = policy.predict(env)
        self.assertGreater(float(action[0]), 0.0)


if __name__ == "__main__":
    unittest.main()