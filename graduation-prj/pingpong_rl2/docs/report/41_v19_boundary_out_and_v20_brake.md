# v19 Boundary-Out Analysis And v20 Brake Fix

## Context

- Run: `artifacts/ppo_runs/pmk_cf_self_rally_v19`
- Model analyzed: `pmk_cf_self_rally_v19_model.zip`
- Analysis: `analysis/pmk_cf_self_rally_v19_final_contact_diagnosis_summary.json`
- v19 preset: `contact_frame_self_rally_v19_anti_low_loop`
- Action mode stayed 13D: `position_contact_frame_velocity_tilt_lateral_residual`

## v19 Result Summary

Compared with v18, v19 is directionally better on lateral control and next-intercept reachability, but still fails through low apex collapse:

| Metric | v18 | v19 |
| --- | ---: | ---: |
| mean useful bounces | 2.42 | 2.21 |
| max useful bounces | 8 | 9 |
| one-or-more useful rate | 0.77 | 0.71 |
| two-or-more useful rate | 0.58 | 0.53 |
| three-or-more useful rate | 0.43 | 0.32 |
| `ball_out_of_bounds` | 30/100 | 8/100 |
| `time_limit` | 48/100 | 22/100 |
| `low_apex_contact` | 18/100 | 69/100 |
| upward contact below 0.20m | 0.505 | 0.380 |
| upward contact below target | 0.725 | 0.617 |
| next intercept reachable rate | 0.726 | 0.863 |
| mean outgoing xy error | 0.093 | 0.055 |

The apparent rise in `low_apex_contact` is partly because v19 made that condition stricter: threshold changed from `0.14m` to `0.20m`, and grace changed from `3` to `2`. The stricter label is useful, because terminal contacts still show a real height problem.

Terminal contact summary in v19:

- mean terminal projected apex above racket: `0.204m`
- median terminal projected apex above racket: `0.164m`
- terminal upward contact below `0.20m`: `0.753`
- terminal outgoing z error: `0.841m/s`

So v19 reduced low contacts overall, but the episodes that die still die through too-low final upward contacts.

## What `time_limit` Means

`time_limit` is not "failed to collect enough useful bounces in time". In `KeepUpEnv.step`, a terminal failure sets `terminated`; otherwise reaching `max_episode_steps=600` sets `truncated`. The rebound analysis script maps `failure_reason=None` plus `truncated=True` to `time_limit`.

So `time_limit` means the episode survived until the configured horizon. If it had one or more successful bounces, env info also marks `episode_success_reason="keepup_time_limit"`. For v19, `time_limit` episodes are the good long-running bucket:

- `time_limit`: 22 episodes, mean useful bounces `5.045`, max `9`, mean contacts `28.9`
- `low_apex_contact`: 69 episodes, mean useful bounces `1.435`, max `6`, mean contacts `13.2`
- `ball_out_of_bounds`: 8 episodes, mean useful bounces `1.125`, max `3`, mean contacts `19.1`

## v19 `ball_out_of_bounds` Pattern

Episodes: `31, 40, 42, 78, 80, 82, 83, 99`.

The common pattern is not random y drift. It is mostly x-axis outward drift:

- terminal contact ball x mean: `0.697`
- terminal projected apex x mean: `0.727`
- terminal target x mean: `0.664`
- terminal next-intercept xy error mean: `0.210`
- terminal actual outgoing x mean: `+0.747m/s`
- terminal desired outgoing x mean: `-0.319m/s`
- terminal outgoing xy error mean: `1.067m/s`
- terminal racket lateral speed mean: `0.459m/s`

In these episodes, the policy already asks for negative outgoing-x residual, but the racket is often still moving outward at contact. The controller target velocity is frequently inward, but the actual racket velocity has not reversed quickly enough. The contact then transfers outward lateral motion to the ball and pushes the next intercept out of the reachable region.

This means the fix should not be "just increase action bounds". v19 action saturation is effectively zero; the failure is a contact execution/control timing issue.

## v20 Changes

Implemented preset: `contact_frame_self_rally_v20_boundary_brake`.

Code changes:

- Added planner contact centering offset:
  - `contact_frame_planner_contact_offset_ratio`
  - `contact_frame_planner_contact_offset_max`
  - With contact-frame planner active, the racket target can be biased slightly from the predicted contact point back toward the anchor.
  - v20 uses ratio `0.35`, max `0.030m`.
- Added lateral brake target velocity:
  - `contact_frame_lateral_brake_gain`
  - `contact_frame_lateral_brake_max`
  - `contact_frame_lateral_brake_radius`
  - When the contact target is far from anchor and the racket is moving farther outward, target velocity receives an inward brake component.
  - v20 uses gain `1.25`, max `0.45m/s`, radius `0.12m`.
- Strengthened lateral/out-of-bounds shaping:
  - `contact_racket_lateral_velocity_penalty_weight: 0.45 -> 0.80`
  - `contact_racket_lateral_velocity_tolerance: 0.18 -> 0.12`
  - `max_contact_racket_lateral_speed_for_success: 0.45 -> 0.35`
  - `next_intercept_xy_error_penalty_weight: 1.35 -> 1.60`
  - `post_contact_lateral_velocity_penalty_weight: 0.90 -> 1.05`
  - `trajectory_error_penalty_weight: 0.75 -> 0.95`
- Strengthened low-apex pressure:
  - `contact_apex_under_target_penalty_weight: 1.15 -> 1.35`
  - low-apex recovery lift gain/max: `0.024/0.045 -> 0.032/0.060`
  - low-apex recovery velocity gain/max: `0.28/0.45 -> 0.38/0.60`

Analysis script now logs:

- `contact_frame_lateral_brake_velocity_x`
- `contact_frame_lateral_brake_velocity_y`
- `contact_frame_planner_contact_target_x`
- `contact_frame_planner_contact_target_y`
- contact summary brake speed mean/max

## Validation

Passed:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  scripts/run_ppo_learning.py \
  scripts/run_ppo_rebound_analysis.py \
  tests/test_keepup_env.py

PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_env

PYTHONPATH=src conda run -n mujoco_env python -m unittest \
  tests.test_ppo_runs tests.test_keepup_contract_features tests.test_vector_env tests.test_scene_load

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v20_boundary_brake \
  --run-name tmp_v20_resume_check2 \
  --run-version codex \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v19/pmk_cf_self_rally_v19_model.zip \
  --total-timesteps 64 \
  --smoke \
  --output-dir artifacts/tmp/tmp_v20_resume_check2_codex

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/tmp/tmp_v20_resume_check2_codex/tmp_v20_resume_check2_codex_model.zip \
  --episodes 2 \
  --analysis-name tmp_v20_resume_check2_analysis \
  --compare-apex-targets
```

## Next Training Command

Use a short fine-tune first because v20 changes controller execution and reward weights:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v20_boundary_brake \
  --run-name pmk_cf_self_rally \
  --run-version v20 \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v19/pmk_cf_self_rally_v19_model.zip \
  --total-timesteps 500000
```

Then analyze:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v20/pmk_cf_self_rally_v20_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v20_final_contact_diagnosis \
  --compare-apex-targets
```

Success criteria for v20:

- `ball_out_of_bounds` should drop below v19 `8/100`, ideally `0-2/100`.
- terminal `racket_lateral_speed` in any remaining out-of-bounds episode should be lower than v19's `~0.46m/s`.
- upward contact below `0.20m` should stay below v19 `0.380`.
- terminal upward contact below `0.20m` should drop from v19 `0.753`.
- mean useful bounces should not fall below v19 by more than a small transient amount.
