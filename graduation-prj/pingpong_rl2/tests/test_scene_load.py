from __future__ import annotations

import unittest

import numpy as np

from pingpong_rl2.envs import PingPongSim


class PingPongSimTests(unittest.TestCase):
    def test_scene_loads_required_objects(self) -> None:
        sim = PingPongSim()
        self.assertGreaterEqual(sim.ball_joint_id, 0)
        self.assertGreaterEqual(sim.ball_body_id, 0)
        self.assertGreaterEqual(sim.racket_site_id, 0)
        self.assertGreaterEqual(sim.racket_head_geom_id, 0)


    def test_ball_spawns_above_racket_center(self) -> None:
        sim = PingPongSim()
        sim.reset_ball_above_racket(height=0.22, xy_offset=(0.01, -0.01))
        delta = sim.ball_position - sim.racket_position
        self.assertTrue(np.allclose(delta[:2], np.array([0.01, -0.01]), atol=1.0e-6))
        self.assertTrue(np.isclose(delta[2], 0.22, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()