from __future__ import annotations

import numpy as np

from pingpong_rl3.envs import TwoBallKeepUpEnv, TwoBallKeepUpGymEnv, TwoBallPingPongSim
from pingpong_rl3.training import make_sb3_async_vector_env


def test_scene_loads_with_two_balls() -> None:
    sim = TwoBallPingPongSim()

    assert sim.ball_count == 2
    assert sim.ball_positions.shape == (2, 3)
    assert sim.ball_velocities.shape == (2, 3)


def test_env_reset_and_step_zero_action() -> None:
    env = TwoBallKeepUpGymEnv(max_episode_steps=5)
    observation, info = env.reset(seed=123)

    assert observation.shape == env.observation_space.shape
    assert info["contact_count"] == 0

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_observation, reward, terminated, truncated, step_info = env.step(action)

    assert next_observation.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "useful_bounces" in step_info
    env.close()


def test_zero_max_episode_steps_disables_time_limit() -> None:
    env = TwoBallKeepUpEnv(max_episode_steps=0, reset_xy_range=0.0, reset_velocity_xy_range=0.0)
    env.reset(seed=1)

    _, _, terminated, truncated, _ = env.step(np.zeros(env.action_size, dtype=float))

    assert env.max_episode_steps is None
    assert env.training_config()["max_episode_steps"] is None
    assert isinstance(terminated, bool)
    assert truncated is False
    env.close()


def test_reset_distribution_uses_slot_offsets_and_disk_radius() -> None:
    env = TwoBallKeepUpEnv(
        reset_xy_range=0.05,
        reset_height_jitter=0.02,
        reset_velocity_xy_range=0.0,
        reset_spin_range=0.0,
    )

    for seed in range(8):
        env.reset(seed=seed)
        relative_positions = env.sim.ball_positions - env._anchor_position
        for ball_index in range(env.sim.ball_count):
            slot_delta = relative_positions[ball_index, :2] - env.slot_xy_offsets[ball_index]
            assert np.linalg.norm(slot_delta) <= env.reset_xy_range + 1.0e-9

        sorted_heights = np.sort(relative_positions[:, 2])
        assert np.all(sorted_heights >= np.array([0.40, 0.67]) - env.reset_height_jitter - 1.0e-9)
        assert np.all(sorted_heights <= np.array([0.40, 0.67]) + env.reset_height_jitter + 1.0e-9)
        assert np.allclose(env.sim.ball_velocities[:, :2], 0.0)
        assert np.allclose([env.sim.ball_angular_velocity(index) for index in range(env.sim.ball_count)], 0.0)
    env.close()


def test_contact_active_at_step_start_is_not_counted_as_new_contact() -> None:
    env = TwoBallKeepUpEnv(
        max_episode_steps=10,
        reset_xy_range=0.0,
        reset_velocity_xy_range=0.0,
        reset_spin_range=0.0,
    )
    env.reset(seed=2)
    racket_position = env.sim.racket_position.copy()
    env.sim.reset(
        ball_positions=[
            racket_position + np.array([0.0, 0.0, 0.020], dtype=float),
            racket_position + np.array([0.08, 0.0, 0.40], dtype=float),
        ],
        ball_velocities=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ball_angular_velocities=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    )
    env.controller.reset()
    env._anchor_position = env.sim.racket_position.copy()

    assert env.sim.has_contact("ball_0_geom", "racket_head")

    _, _, _, _, info = env.step(np.zeros(env.action_size, dtype=float))

    assert info["contact_count"] == 0
    assert "last_contact_useful" not in info
    env.close()


def test_training_config_captures_reward_and_bounds_parameters() -> None:
    env = TwoBallKeepUpEnv(max_episode_steps=0, terminate_on_ball_ball_contact=False)

    config = env.training_config()

    assert config["max_episode_steps"] is None
    assert config["slot_xy_offsets"] == env.slot_xy_offsets.tolist()
    assert config["x_bounds"] == list(env.x_bounds)
    assert config["tilt_limit"] == env.tilt_limit.tolist()
    assert config["contact_bonus"] == env.contact_bonus
    assert config["terminate_on_ball_ball_contact"] is False
    env.close()


def test_sb3_vector_env_resets_done_envs() -> None:
    vector_env = make_sb3_async_vector_env(
        num_envs=1,
        env_kwargs={"max_episode_steps": 2, "reset_xy_range": 0.0, "reset_velocity_xy_range": 0.0},
        seed=5,
    )
    try:
        observations = vector_env.reset()
        actions = np.zeros((1, vector_env.action_space.shape[0]), dtype=np.float32)

        next_observations, rewards, dones, infos = vector_env.step(actions)
        assert observations.shape == next_observations.shape
        assert rewards.shape == (1,)
        assert dones.tolist() == [False]
        assert infos[0]["elapsed_steps"] == 1

        _, _, dones, infos = vector_env.step(actions)
        assert dones.tolist() == [True]
        assert infos[0]["TimeLimit.truncated"] is True
        assert "terminal_observation" in infos[0]

        _, _, dones, infos = vector_env.step(actions)
        assert dones.tolist() == [False]
        assert infos[0]["elapsed_steps"] == 1
    finally:
        vector_env.close()
