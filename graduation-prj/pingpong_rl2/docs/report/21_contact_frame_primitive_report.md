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

## Residual-RL Follow-Up

The contact-frame primitive was extended so that `position_contact_frame` can be used as a residual-RL action mode.

New environment parameters:

```text
contact_frame_base_strike_z_boost
contact_frame_base_strike_z_offset
contact_frame_base_strike_time_horizon
contact_frame_base_tilt_residual
contact_frame_action_penalty_weight
```

The `contact_frame_candidate` PPO preset now uses:

```text
contact_frame_base_strike_z_boost = 0.024
contact_frame_base_strike_z_offset = 0.01
contact_frame_base_tilt_residual = (-0.02, 0.0)
contact_frame_action_penalty_weight = 0.05
post_contact_return_assist_weight = 0.8
log_std_init = -3.0
zero_init_action_mean = True
```

This means deterministic zero residual is no longer an empty action. It is the centered scripted contact-frame strike, and PPO learns residuals around it.

Validation:

```text
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests/test_keepup_env.py
Ran 53 tests in 1.773s
OK
```

Zero-residual manual baseline over 100 episodes:

```text
mean_useful_bounces = 0.83
max_useful_bounces = 4
one_or_more_useful_bounce_rate = 0.48
two_or_more_useful_bounce_rate = 0.17
three_or_more_useful_bounce_rate = 0.12
failure_counts = {"ball_out_of_bounds": 77, "ball_speed_limit": 14, "floor_contact": 9}
```

PPO zero-initialized model with `total_timesteps=0`, 100 eval episodes:

```text
run = pmk_cf_zero_init_eval_v2
mean_useful_bounces = 0.67
max_useful_bounces = 4
one_or_more_useful_bounce_rate = 0.40
two_or_more_useful_bounce_rate = 0.16
three_or_more_useful_bounce_rate = 0.07
```

Short PPO training, 20k timesteps, low learning rate:

```text
run = pmk_cf_zero_init_20k
learning_rate = 0.00005
mean_useful_bounces = 0.73
max_useful_bounces = 2
one_or_more_useful_bounce_rate = 0.56
two_or_more_useful_bounce_rate = 0.17
three_or_more_useful_bounce_rate = 0.00
```

Interpretation:

The project now has a PPO-compatible policy/model path that can produce multiple keep-up bounces through residual action structure. However, PPO updates still tend to trade away rare `3+` episodes even when mean one-bounce performance improves. The next learning work should preserve the zero-residual baseline while allowing small residual improvements.

Recommended next learning changes:

- add an evaluation metric for `three_or_more_useful_bounce_rate` to checkpoint ranking,
- use early stopping or model selection based on `max_useful_bounces` and `three_or_more_useful_bounce_rate`, not only mean/two-or-more rate,
- try lower `clip_range` and/or lower learning rate before increasing timesteps,
- consider an explicit KL/action-mean regularizer toward zero residual for early curriculum stages,
- only broaden reset randomization after `3+` survives PPO updates.

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
