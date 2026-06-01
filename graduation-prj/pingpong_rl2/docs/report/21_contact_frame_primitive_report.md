# 21 Contact-Frame Primitive Report

Date: 2026-06-01

## Goal

Add the first concrete implementation step from the keep-up completion plan:

Train and diagnose a primitive that can express "hit the ball back toward a repeatable hittable region" before spending more time on PPO or reward-weight tuning.

## Implemented Change

Added a new action mode:

`position_contact_frame`

Action shape:

```text
[radial_offset, tangential_offset, strike_z_residual, pitch_residual, roll_residual]
```

The first two action dimensions are no longer raw world XY. They are interpreted in a contact/return frame:

```python
radial = normalize(controller_anchor_xy - predicted_intercept_xy)
tangent = [-radial_y, radial_x]
target_xy = predicted_intercept_xy + radial * radial_offset + tangent * tangential_offset
```

Fallback frame sources are used when the anchor/intercept vector is too small:

- negative ball horizontal velocity,
- anchor minus current ball XY,
- world x-axis fallback.

This keeps the primitive focused on contact geometry instead of making the policy rediscover the return direction from raw XY offsets.

## Files Changed

- `src/pingpong_rl2/envs/keepup_env.py`
- `src/pingpong_rl2/controllers/heuristic_keepup.py`
- `scripts/run_heuristic_keepup_diagnostic.py`
- `scripts/run_ppo_learning.py`
- `src/pingpong_rl2/defaults.py`
- `src/pingpong_rl2/utils/ppo_runs.py`
- `tests/test_keepup_env.py`

## Environment Integration

`position_contact_frame` now:

- has a 5D action space,
- exposes racket face normal and target tilt in observations,
- reuses existing strike tilt ramp/assist behavior,
- reuses follow-up strike contract logic,
- works with `HeuristicKeepUpPolicy`,
- is selectable in `run_heuristic_keepup_diagnostic.py`,
- has a PPO preset named `contact_frame_candidate`, but this should only be used after the scripted gate passes.

## Validation Done

Use the MuJoCo conda environment for project commands:

```bash
conda activate mujoco_env
```

Syntax validation passed:

```bash
python3 -m py_compile \
  pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py \
  pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py \
  pingpong_rl2/scripts/run_heuristic_keepup_diagnostic.py \
  pingpong_rl2/scripts/run_ppo_learning.py \
  pingpong_rl2/src/pingpong_rl2/defaults.py \
  pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py \
  pingpong_rl2/tests/test_keepup_env.py
```

`pytest` was not run successfully in the current shell because the global `python3` environment does not have `pytest` installed.

Unit tests passed in the MuJoCo conda environment:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests/test_keepup_env.py
```

Result:

```text
Ran 50 tests in 1.634s
OK
```

## Diagnostic Results

Smoke result from `contact_frame_smoke`, 5 episodes:

```text
mean_useful_bounces = 1.200
max_useful_bounces = 4
one_or_more_useful_bounce_rate = 0.400
two_or_more_useful_bounce_rate = 0.400
three_or_more_useful_bounce_rate = 0.200
next_intercept_reachable_rate = 0.658
failure_counts = {"ball_out_of_bounds": 3, "ball_speed_limit": 2}
```

30 episode result from `contact_frame_30ep`:

```text
mean_useful_bounces = 0.833
max_useful_bounces = 4
one_or_more_useful_bounce_rate = 0.500
two_or_more_useful_bounce_rate = 0.267
three_or_more_useful_bounce_rate = 0.033
next_intercept_reachable_rate = 0.677
useful_contact_next_intercept_reachable_rate = 0.920
failure_counts = {"ball_out_of_bounds": 17, "ball_speed_limit": 5, "robot_body_contact": 5, "floor_contact": 3}
```

Interpretation:

The contact-frame primitive passes the first scripted gate because it produced `max_useful_bounces >= 3` without a contact oracle. It is not stable enough yet. The next work should reduce failure modes and improve `three_or_more_useful_bounce_rate`, not abandon this primitive.

## Required Next Diagnostic

Run this inside the MuJoCo project environment:

```bash
python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_frame_primitive \
  --variant-name contact_frame_smoke \
  --action-mode position_contact_frame \
  --episodes 5 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-position-residual 0.0 0.0 0.01 \
  --strike-tilt-residual -0.02 0.0 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --post-contact-return-assist-weight 0.5 \
  --post-contact-return-max-intercept-time 0.6 \
  --print-episodes
```

If it does not crash, run:

```bash
python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_frame_primitive \
  --variant-name contact_frame_30ep \
  --action-mode position_contact_frame \
  --episodes 30 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --strike-z-boost 0.024 \
  --strike-position-residual 0.0 0.0 0.01 \
  --strike-tilt-residual -0.02 0.0 \
  --strike-tilt-ramp-pitch -0.06 \
  --followup-strike-target-tilt -0.06 0.0 \
  --post-contact-return-assist-weight 0.5 \
  --post-contact-return-max-intercept-time 0.6
```

## Gate

Do not proceed to long PPO until scripted diagnostics show:

- `max_useful_bounces >= 3`
- `three_or_more_useful_bounce_rate > 0`

Preferably, the 100 episode confirmation should show:

- `three_or_more_useful_bounce_rate >= 0.10`

## Decision Rule

If `position_contact_frame` still plateaus at `max_useful_bounces <= 2`, the next change should be an impact-time primitive, not reward tuning:

```text
[radial_offset, tangential_offset, contact_lead_time, z_velocity_bias, normal_pitch]
```

That primitive should tie upward racket velocity to predicted contact time or closest approach time.
