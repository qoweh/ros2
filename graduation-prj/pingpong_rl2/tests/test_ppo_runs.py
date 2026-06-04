from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_ppo_learning import ResetDistributionCurriculumCallback, scaled_action_log_std
from pingpong_rl2.utils.ppo_runs import load_env_config_for_model, resolve_env_kwargs_for_model


class PpoRunsPathResolutionTests(unittest.TestCase):
    def test_load_env_config_for_best_model_uses_base_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "clean_tnp_ckpt_v1"
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "clean_tnp_ckpt_v1_training_summary.json"
            summary_path.write_text(
                json.dumps({"env_config": {"action_mode": "position_strike", "strike_tilt_ramp_pitch": -0.03}}),
                encoding="utf-8",
            )

            env_config = load_env_config_for_model(run_dir / "clean_tnp_ckpt_v1_best_model.zip")

            self.assertIsNotNone(env_config)
            assert env_config is not None
            self.assertEqual(env_config["action_mode"], "position_strike")
            self.assertAlmostEqual(env_config["strike_tilt_ramp_pitch"], -0.03)

    def test_resolve_env_kwargs_can_override_target_height_separately(self) -> None:
        env_kwargs = resolve_env_kwargs_for_model(
            ball_height=0.42,
            target_ball_height=0.25,
        )

        self.assertAlmostEqual(env_kwargs["ball_height"], 0.42)
        self.assertAlmostEqual(env_kwargs["target_ball_height"], 0.25)

    def test_scaled_action_log_std_uses_action_limit_ratio_and_clamps(self) -> None:
        log_std = scaled_action_log_std(
            action_high=[0.02, 0.006, 0.75],
            ratio=0.35,
            min_std=0.0015,
            max_std=0.08,
        )

        self.assertTrue(np.allclose(np.exp(log_std), np.array([0.007, 0.0021, 0.08])))

    def test_reset_distribution_curriculum_interpolates_all_reset_ranges(self) -> None:
        callback = ResetDistributionCurriculumCallback(
            start_xy_range=0.075,
            end_xy_range=0.16,
            start_velocity_xy_range=0.025,
            end_velocity_xy_range=0.06,
            start_velocity_z_range=(-0.08, 0.02),
            end_velocity_z_range=(-0.16, 0.04),
            start_ball_angular_velocity_range=0.0,
            end_ball_angular_velocity_range=20.0,
            total_timesteps=100,
            curriculum_fraction=0.5,
            update_interval=10,
        )
        callback._base_num_timesteps = 0
        callback.num_timesteps = 25

        distribution = callback.target_distribution()

        self.assertAlmostEqual(distribution["reset_xy_range"], 0.1175)
        self.assertAlmostEqual(distribution["reset_velocity_xy_range"], 0.0425)
        reset_velocity_z_range = distribution["reset_velocity_z_range"]
        self.assertAlmostEqual(reset_velocity_z_range[0], -0.12)
        self.assertAlmostEqual(reset_velocity_z_range[1], 0.03)
        self.assertAlmostEqual(distribution["reset_ball_angular_velocity_range"], 10.0)


if __name__ == "__main__":
    unittest.main()
