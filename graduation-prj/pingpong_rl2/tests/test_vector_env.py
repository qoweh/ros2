from __future__ import annotations

import unittest

import numpy as np

from pingpong_rl2.training import make_gym_vector_env, make_sb3_async_vector_env


class VectorEnvTests(unittest.TestCase):
    def test_gym_async_vector_env_reset_shape(self) -> None:
        vector_env = make_gym_vector_env(num_envs=2, env_kwargs={"reset_xy_range": 0.0}, vector_mode="async")
        try:
            observations, _ = vector_env.reset(seed=5)
            self.assertEqual(observations.shape[0], 2)
            self.assertEqual(observations.ndim, 2)
        finally:
            vector_env.close()


    def test_sb3_async_vector_env_adapter_step_contract(self) -> None:
        vector_env = make_sb3_async_vector_env(num_envs=2, env_kwargs={"reset_xy_range": 0.0}, seed=5)
        try:
            observations = vector_env.reset()
            actions = np.zeros((2, vector_env.action_space.shape[0]), dtype=np.float32)
            next_observations, rewards, dones, infos = vector_env.step(actions)
            self.assertEqual(observations.shape[0], 2)
            self.assertEqual(next_observations.shape[0], 2)
            self.assertEqual(rewards.shape, (2,))
            self.assertEqual(dones.shape, (2,))
            self.assertEqual(len(infos), 2)
        finally:
            vector_env.close()


if __name__ == "__main__":
    unittest.main()