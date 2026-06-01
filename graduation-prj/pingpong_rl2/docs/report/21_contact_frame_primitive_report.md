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

## 2026-06-01 Follow-up: Preserve The Primitive Before Longer PPO

User observation:

- `pmk_cf_zero_init_eval_v2` looks usable, but the ball still sometimes drifts away from the robot.
- Some contacts look too low, which shortens the next strike period and makes follow-up hits harder.
- It was unclear whether racket roll/pitch/yaw are being used.

Code-level answer:

- Pitch and roll are already part of the controller through `target_tilt = [pitch, roll]`.
- The `position_contact_frame` action also exposes pitch/roll residuals as action dimensions 4 and 5.
- Yaw is not currently controlled. The controller aims the racket face normal; yaw around that normal does not change the rebound normal for the current round/symmetric racket model, so it is not the first lever to add unless the racket/contact model becomes asymmetric or spin-aware.

Changes made:

- Added `--target-ball-height` to `scripts/run_ppo_learning.py` and `scripts/run_heuristic_keepup_diagnostic.py`.
  - `--ball-height` remains the spawn/reset height.
  - `--target-ball-height` is the desired post-contact apex height above the racket.
  - Default behavior is unchanged: target height falls back to `--ball-height`.
- Updated `contact_frame_candidate` so PPO sees the actual keep-up objective more directly:
  - `trajectory_match_reward_weight = 0.4`
  - `useful_contact_return_target_xy_reward_weight = 0.25`
  - `return_target_xy_tolerance = 0.12`
  - `next_intercept_reachable_bonus_weight = 0.25`
  - `easy_next_ball_reward_weight = 0.3`
  - `include_desired_outgoing_velocity_observation = True`
- Increased the contact-frame base lift slightly:
  - `contact_frame_base_strike_z_boost: 0.024 -> 0.032`
- Increased residual action penalty:
  - `contact_frame_action_penalty_weight: 0.05 -> 0.2`
- Added initial checkpoint evaluation to PPO checkpoint selection.
  - `*_best_model.zip` can now remain the zero-initialized primitive if PPO updates make the policy worse.
  - This prevents a training run from silently replacing the best usable model with a degraded final checkpoint.
- Added PPO stability options:
  - `--n-epochs`
  - `--clip-range`
  - `--ent-coef`
  - `--vf-coef`

Short diagnostics:

```text
coarse zero-action primitive check, 120 episodes:
current-like lift: mean_useful=0.692, max=4, two_or_more=0.183, three_or_more=0.075
zboost=0.032:     mean_useful=0.733, max=4, two_or_more=0.192, three_or_more=0.092
```

```text
rewarded PPO 50k:
final model degraded badly: mean_useful=0.040, max=1
conclusion: reward additions alone are not enough.
```

```text
conservative PPO 10k:
command used n_epochs=1, clip_range=0.05, lr=1e-5
checkpoint history:
  step 0:     mean_useful=0.567, max=3, two_or_more=0.167, three_or_more=0.100
  step 5000:  mean_useful=0.700, max=2, two_or_more=0.100, three_or_more=0.000
  step 10000: mean_useful=0.633, max=3, two_or_more=0.200, three_or_more=0.067

best_model stayed at step 0 because 3+ survival is currently the priority.
```

Interpretation:

The immediate problem is not that pitch/roll are absent. They are present. The problem is that the zero-residual primitive is still only marginally stable, and ordinary PPO updates quickly destroy the rare multi-bounce behavior. Raising height too aggressively also causes more speed-limit/out-of-bounds failures, so height should be increased through a small lift primitive plus conservative model selection, not by simply raising `target_ball_height` first.

Previous recommended command before the bootstrap preset:

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_candidate \
  --run-name pmk_cf_conservative \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --n-envs 1 \
  --n-steps 512 \
  --batch-size 256 \
  --learning-rate 0.00001 \
  --n-epochs 1 \
  --clip-range 0.05 \
  --checkpoint-interval 10000 \
  --checkpoint-eval-episodes 50 \
  --eval-episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --early-stop-patience-evals 4
```

Use `*_best_model.zip`, not `*_model.zip`, for viewer/evaluation unless the final model explicitly beats the best checkpoint.

If this still cannot improve beyond the zero primitive, stop reward tuning and implement the impact-time primitive from the decision rule above.

## 2026-06-01 Second Follow-up: Height Control And Center-Seeking Contact

Question addressed:

- Should the racket apply a fixed upward impulse, or should it strike so the ball reaches a consistent useful apex?
- Can the arm fold inward / strike closer to the robot when the next ball comes close?
- Was pitch/roll rejected too early?
- Does `arXiv:2408.03906v3` suggest a better structure?

Paper takeaway:

`Achieving Human Level Competitive Robot Table Tennis` is not directly the same task, but it strongly supports a modular skill approach: a high-level controller chooses among frozen low-level skills, and the low-level skills have descriptors such as hit velocity, landing location, and success probability. The paper also explicitly notes that monolithic policies are harder to evaluate and can forget, while a good learned skill can be preserved and specialized later. This matches the current project failure mode: PPO updates keep damaging a useful primitive.

Implementation changes:

- Added optional apex-aware contact-frame lift:
  - `contact_frame_apex_lift_gain`
  - `contact_frame_apex_lift_max`
  - `contact_frame_apex_lift_reference_velocity_z`
  - `contact_frame_apex_lift_restitution`
- Added optional continuous center-seeking contact-frame tilt:
  - `contact_frame_centering_tilt_limit`
  - `contact_frame_centering_tilt_radius`
  - `contact_frame_centering_tilt_deadband`
- Added CLI support for these options in:
  - `scripts/run_ppo_learning.py`
  - `scripts/run_heuristic_keepup_diagnostic.py`
- Enabled the already-existing follow-up contact offset in `contact_frame_candidate`:
  - `followup_strike_contact_offset_ratio = 0.25`
  - `followup_strike_contact_offset_max = 0.02`

Important diagnostic result:

```text
100 episode zero-action comparison:
old preset:
  mean_useful=0.65, max=4, two_or_more=0.16, three_or_more=0.04

offset only:
  mean_useful=0.79, max=5, two_or_more=0.19, three_or_more=0.10

apex lift enabled:
  mean_useful=0.54, max=2, two_or_more=0.11, three_or_more=0.00

offset + apex + center tilt:
  mean_useful=0.55, max=3, two_or_more=0.11, three_or_more=0.01
```

Interpretation:

- The user's instinct is right in principle: a fixed force is not ideal, and the useful target is a repeatable next-ball apex.
- However, the current controller is position-target based, not impact-velocity controlled. The simple apex-lift proxy makes the racket over-hit or mistime the contact, increasing speed-limit failures. It should stay available as an experiment but not be on by default yet.
- The user's pitch/roll instinct is also right in principle. The problem is that the current primitive already uses a fixed pitch near the target tilt limit, leaving little headroom for dynamic pitch/roll to help. Reducing fixed pitch to make room for dynamic tilt degraded the current primitive in short tests.
- The most reliable immediate improvement is not extra tilt. It is shifting the follow-up strike contact point slightly toward the robot center, which lets the arm intercept balls in a more repeatable region and improves multi-bounce survival.

Current recommended command:

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_candidate \
  --run-name pmk_cf_offset \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --n-envs 1 \
  --n-steps 512 \
  --batch-size 256 \
  --learning-rate 0.00001 \
  --n-epochs 1 \
  --clip-range 0.05 \
  --checkpoint-interval 10000 \
  --checkpoint-eval-episodes 50 \
  --eval-episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --early-stop-patience-evals 4
```

Short PPO confirmation:

```text
pmk_cf_offset_20k checkpoint history:
  step 0:     mean_useful=0.64, max=4, two_or_more=0.18, three_or_more=0.08
  step 10000: mean_useful=0.58, max=2, two_or_more=0.14, three_or_more=0.00
  step 20000: mean_useful=0.48, max=4, two_or_more=0.14, three_or_more=0.04

final 100 episode evaluation:
  mean_useful=0.67, max=4, two_or_more=0.21, three_or_more=0.11
```

Use the best model first. The final model can be inspected too, but the project should keep selecting by `three_or_more` survival until long keep-up is stable.

Next structural change if PPO still plateaus:

Implement a true impact-time / impact-velocity primitive. The current apex-lift experiment shows that apex control is the correct objective but position offset is the wrong actuator abstraction. The primitive should explicitly choose:

```text
[radial_contact_offset, tangential_contact_offset, target_apex_height, target_apex_xy, impact_time_or_velocity_bias]
```

The controller then needs to convert target apex and incoming ball velocity into required racket velocity near contact, instead of only moving the racket target position upward.

## 2026-06-01 Third Follow-up: Target Height Was Being Clamped Too High

New finding:

`--target-ball-height` was not actually able to request a lower repeatable apex. Internally `_target_ball_height_above_racket()` returned the maximum of `target_ball_height` and the initial spawn height above the racket. With the default spawn height, any value below roughly 0.5 m was silently ignored.

This matters because the user's intuition was correct: the project needs a repeatable next-ball apex, not a fixed upward force. But the useful repeat apex is lower than the initial drop height. The previous conservative PPO runs were still training against an apex that was too high/fast, so the ball often left the easy strike region.

Code changes:

- `_target_ball_height_above_racket()` now returns `target_ball_height` directly.
- `target_ball_height` is validated as positive.
- Added a regression test proving `target_ball_height` can be lower than `ball_height`.
- Updated `contact_frame_candidate`:
  - `target_ball_height = 0.25`
  - `contact_frame_base_strike_z_boost = 0.024`
- Checkpoint evaluation now records floor/body/speed failure rates and, when long-survival is tied, prefers fewer unsafe failures before smaller improvements in two-bounce rate.

Diagnostic comparison after the clamp fix:

```text
contact-frame zero-residual, 200 episodes, seed=9100

target=0.25, z_boost=0.024:
  mean_useful=1.365, max=5, one_or_more=0.665, two_or_more=0.415, three_or_more=0.215
  reachable_rate=0.672, failure_counts={ball_out_of_bounds:126, floor_contact:22, ball_speed_limit:14, robot_body_contact:38}

target=0.28, z_boost=0.024:
  mean_useful=1.165, max=5, one_or_more=0.615, two_or_more=0.340, three_or_more=0.160

target=0.30, z_boost=0.024:
  mean_useful=1.080, max=5, one_or_more=0.585, two_or_more=0.310, three_or_more=0.145

target=0.25, z_boost=0.032:
  mean_useful=1.310, max=5, one_or_more=0.670, two_or_more=0.355, three_or_more=0.195
```

Lower targets around 0.20 m can score even more useful bounces, but they also raise `robot_body_contact` sharply. For now `0.25 m` is the safer default: it improves repeatability without making the policy win mostly by hitting the ball too close to the robot body.

Preset zero-init confirmation:

```text
python scripts/run_ppo_learning.py --preset contact_frame_candidate ... --total-timesteps 0 --eval-episodes 100

evaluation mean_return=7.581
mean_useful_bounces=1.460
max_useful_bounces=4
two_or_more_rate=0.440
three_or_more_rate=0.220
```

Short PPO confirmation:

```text
pmk_cf_lowtarget_20k:
  checkpoint step 0:     mean_useful=1.70, two_or_more=0.50, three_or_more=0.30
  checkpoint step 10000: mean_useful=1.48, two_or_more=0.50, three_or_more=0.24
  checkpoint step 20000: mean_useful=1.74, two_or_more=0.52, three_or_more=0.30

final 100 episode evaluation:
  mean_useful=1.29, max=5, two_or_more=0.39, three_or_more=0.20
```

Interpretation:

- The best immediate answer to "힘을 일정하게 줄까, 높이를 일정하게 맞출까?" is: **height objective first**. The primitive should aim for a repeatable apex; force/velocity should be derived from that.
- The current residual PPO is still fragile. It can preserve the primitive, but it is not yet clearly learning a better long-horizon skill. Do not assume longer training alone will fix the project.
- Pitch/roll are already present and useful as a fixed bias. Dynamic pitch/roll should come back only after the impact primitive can explicitly control outgoing apex/XY. Otherwise tilt changes mostly disturb timing.

Recommended next command:

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_candidate \
  --run-name pmk_cf_lowtarget \
  --run-version v2 \
  --reset-model \
  --total-timesteps 100000 \
  --n-envs 1 \
  --n-steps 512 \
  --batch-size 256 \
  --learning-rate 0.00001 \
  --n-epochs 1 \
  --clip-range 0.05 \
  --checkpoint-interval 10000 \
  --checkpoint-eval-episodes 50 \
  --eval-episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --early-stop-patience-evals 4
```

If this still plateaus around 1-2 useful bounces, the next real work is not more reward tuning. Implement the true impact primitive described above so the agent chooses a target apex/landing point and the low-level controller computes the required contact velocity.

## 2026-06-01 Fourth Follow-up: Anchor-Referenced Apex And Bootstrap Preset

New bugs / mismatches found:

- Desired outgoing velocity was computed against `controller_anchor.z + target_ball_height`.
- The useful-bounce apex check was still effectively comparing against the moving racket frame when contact position was available only as `contact_ball_height_above_racket`.
- When the racket was lifted at impact, a physically correct outgoing velocity could be marked non-useful because the success check demanded `target_ball_height` above the lifted racket, not above the anchor.

Fix:

- `_projected_contact_apex_height_above_racket()` now uses `contact_ball_position_z` when available and subtracts the controller anchor height. This makes the success/reward trajectory target consistent with `_desired_outgoing_velocity()`.

Upper-bound diagnostic:

```text
contact_oracle_mode=desired_outgoing_velocity, blend=1.0
target_ball_height=0.25
episodes=50

mean_useful_bounces=27.48
two_or_more_rate=1.00
three_or_more_rate=1.00
failure_counts={time_limit: 50}
```

Interpretation: the task is possible in this MuJoCo setup. The remaining bottleneck is not reachability in general; it is the physical policy/primitive failing to create the target outgoing velocity without the oracle.

Physical primitive check:

- The optional `contact_frame_apex_lift` calculation had the collision-sign wrong.
- It treated faster downward incoming ball velocity as requiring more upward racket velocity.
- The corrected approximation is:

```text
required_racket_velocity_z = (desired_outgoing_velocity_z + restitution * incoming_ball_velocity_z) / (1 + restitution)
```

This optional primitive is fixed and tested, but it is still not enabled in the main preset because short diagnostics did not improve repeat keep-up robustly.

Best current non-oracle result:

```text
contact_frame_bootstrap_candidate, total_timesteps=0

bootstrap:
  sample_mode=post_success
  bootstrap_heuristic_episodes=120
  bootstrap_min_useful_bounces=2
  bootstrap_max_samples=4000
  bootstrap_epochs=50

evaluation:
  mean_useful_bounces=2.18
  max_useful_bounces=5
  two_or_more_rate=0.65
  three_or_more_rate=0.39
```

Short PPO from this bootstrap did not reliably improve the model. With the current reward, PPO often drifts away from the bootstrapped primitive, so the project should use `*_best_model.zip` and treat checkpoint 0 / bootstrapped actor as a legitimate baseline.

New simple command:

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_bootstrap_candidate \
  --run-name pmk_cf_bootstrap \
  --run-version v1 \
  --reset-model \
  --total-timesteps 0
```

For PPO continuation:

```bash
python scripts/run_ppo_learning.py \
  --preset contact_frame_bootstrap_candidate \
  --run-name pmk_cf_bootstrap \
  --run-version v2 \
  --reset-model \
  --total-timesteps 100000
```

Use the generated `*_best_model.zip`, not necessarily the final `*_model.zip`.

Remaining real gap:

The best non-oracle model still averages only about 2 useful bounces, while the oracle reaches the time limit. Rebound analysis shows outgoing velocity error remains around 1 m/s and failures are mostly `ball_out_of_bounds`. The next structural step is therefore still a true impact-velocity primitive, not more scalar reward tweaking:

```text
policy chooses: target_apex_height, target_apex_xy/residual, contact offset, small tilt residual
low-level primitive computes: required racket velocity near impact from incoming ball velocity and restitution
controller tracks: contact point plus velocity/lead target, rather than only a static position target
```

## 2026-06-01 Fifth Follow-up: Velocity Lead Probe

Implemented but **not enabled by default**:

- `contact_frame_velocity_lead_gain`
- `contact_frame_velocity_lead_max`

Purpose:

The current controller is still position-target based, but the real missing quantity is contact velocity. The new optional probe computes the required racket vertical velocity from desired outgoing ball velocity and incoming ball velocity, compares it with current racket velocity, and converts the error into a small signed z target lead.

Formula used:

```text
required_racket_velocity_z = (desired_outgoing_velocity_z + restitution * incoming_ball_velocity_z) / (1 + restitution)
lead_z = gain * (required_racket_velocity_z - current_racket_velocity_z)
```

Diagnostic result:

```text
120 episodes, seed=10000, target_ball_height=0.25

baseline:
  mean_useful=1.467, two_or_more=0.492, three_or_more=0.233

velocity lead gain=0.03 max=0.02:
  mean_useful=1.258, two_or_more=0.375, three_or_more=0.158

velocity lead gain=0.05 max=0.03:
  mean_useful=0.792, two_or_more=0.225, three_or_more=0.075

velocity lead gain=0.08 max=0.04:
  mean_useful=0.883, two_or_more=0.267, three_or_more=0.083
```

Interpretation:

- The idea is aligned with the real objective, but adding a z lead on top of the current static position controller makes the scripted primitive worse.
- This supports the previous conclusion: the project needs a real impact-velocity primitive or controller-level velocity target, not another z-offset heuristic.
- Keep these options available for future experiments, but do not put them into `contact_frame_candidate` or `contact_frame_bootstrap_candidate` yet.

Contact offset sweep:

```text
150 episodes, seed=10100

offset ratio=0.00 max=0.00:
  mean_useful=1.220, two_or_more=0.360, three_or_more=0.180

offset ratio=0.25 max=0.02:
  mean_useful=1.293, two_or_more=0.373, three_or_more=0.233

offset ratio=0.50 max=0.03:
  mean_useful=1.200, two_or_more=0.373, three_or_more=0.140

offset ratio=0.75 max=0.04:
  mean_useful=1.160, two_or_more=0.353, three_or_more=0.153
```

Interpretation:

- The existing `0.25 / 0.02` follow-up contact offset is still the best of this sweep.
- Stronger center bias reduces out-of-bounds somewhat, but increases robot-body/contact-quality failures and does not improve multi-bounce survival.
