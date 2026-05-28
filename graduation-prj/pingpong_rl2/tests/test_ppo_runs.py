from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pingpong_rl2.utils.ppo_runs import load_env_config_for_model


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

    def test_load_env_config_for_checkpoint_model_uses_parent_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "clean_tnp_ckpt_v1"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            summary_path = run_dir / "clean_tnp_ckpt_v1_training_summary.json"
            summary_path.write_text(
                json.dumps({"env_config": {"action_mode": "position_strike", "include_velocity_domain_observation": True}}),
                encoding="utf-8",
            )

            env_config = load_env_config_for_model(
                checkpoint_dir / "clean_tnp_ckpt_v1_step_0030000_model.zip"
            )

            self.assertIsNotNone(env_config)
            assert env_config is not None
            self.assertEqual(env_config["action_mode"], "position_strike")
            self.assertTrue(env_config["include_velocity_domain_observation"])


if __name__ == "__main__":
    unittest.main()