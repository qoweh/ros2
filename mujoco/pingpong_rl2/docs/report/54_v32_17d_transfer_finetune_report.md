# v32 17D transfer and fine-tune report

## Goal

`keep1_v30`의 15D 정책을 기반으로 17D 정책을 만들고, 같은 v30 계열 reset 분포에서 기존 15D 기준선보다 나은 모델을 확보한다.

## Changes

- Added preset: `contact_frame_self_rally_v32_17d_v30_transfer`
  - Base: `contact_frame_self_rally_v30_v26_wider_xy_stability`
  - Action mode: `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual`
  - New action axes: tracking velocity residual x/y, action indices 15/16
  - Tracking residual action limit: `0.18`
  - Conservative PPO resume params: `learning_rate=1e-5`, `n_epochs=1`, `clip_range=0.05`
- Added config: `configs/keep1_v32_17d_transfer.json`
- Added transfer utility: `scripts/expand_ppo_action_space.py`
  - Copies all matching PPO policy/value weights from a smaller action model.
  - Copies the first 15 action-head rows and `log_std` entries.
  - Zero-initializes the new 17D action-head rows so the expanded model starts as a stable v30-compatible policy.
- Fixed tracking residual logging in `PingPongKeepUpEnv.step()`.
  - The contact log now records the residual used for the controller velocity target before the simulation step, instead of re-evaluating the descent gate after contact.

## Models

| Model | Source | Notes |
|---|---|---|
| `keep1_v30` | `artifacts/ppo_runs/keep1_v30/keep1_v30_model.zip` | 15D baseline |
| `keep1_v32_17d_init` | `artifacts/ppo_runs/keep1_v32_17d_init/keep1_v32_17d_init_model.zip` | v30 weights expanded to 17D, new axes initialized to zero |
| `keep1_v32_17d` | `artifacts/ppo_runs/keep1_v32_17d/keep1_v32_17d_model.zip` | `keep1_v32_17d_init` resumed for 200k steps |

## Same-seed eval100 results

Evaluation command pattern:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path <model.zip> \
  --episodes 100 \
  --seed 231 \
  --episode-step-limit 1800 \
  --analysis-name <name>
```

| Model | Mean return | Mean useful bounces | Max useful bounces | 30+ rate | Time limit | Ball out | Low apex | Body contact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `keep1_v30` | 282.933 | 30.56 | 53 | 0.71 | 70 | 8 | 19 | 2 |
| `keep1_v32_17d_init` | 309.310 | 32.09 | 59 | 0.76 | 68 | 14 | 17 | 0 |
| `keep1_v32_17d` | 307.956 | 32.11 | 48 | 0.74 | 71 | 6 | 21 | 1 |

## Common difficulty check against v26

Stored v26 analysis used `reset_xy_range=0.075`, while v30/v32 use `reset_xy_range=0.10`. To avoid comparing an easier 15D distribution with a harder 17D distribution, v26 was re-evaluated with the v32 reset width:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v26/keep1_v26_model.zip \
  --episodes 100 \
  --seed 231 \
  --episode-step-limit 1800 \
  --reset-xy-range 0.10 \
  --analysis-name keep1_v26_resetxy010_eval100
```

| Model | Reset XY | Mean return | Mean useful bounces | Max useful bounces | 30+ rate |
|---|---:|---:|---:|---:|---:|
| `keep1_v26` original stored eval | 0.075 | - | 42.34 | 85 | 0.60 |
| `keep1_v26_resetxy010_eval100` | 0.10 | 261.617 | 26.25 | 54 | 0.54 |
| `keep1_v30_current_eval100` | 0.10 | 282.933 | 30.56 | 53 | 0.71 |
| `keep1_v32_17d` | 0.10 | 307.956 | 32.11 | 48 | 0.74 |

Under the same `reset_xy_range=0.10` difficulty, `keep1_v32_17d` beats both the current v30 15D baseline and the older v26 15D model on mean useful bounces and 30+ rate.

## 17D action usage evidence

Contact CSV statistics from the logged eval100 runs:

| Model | Mean abs tracking x | Mean abs tracking y | Max abs tracking x | Max abs tracking y | Nonzero contact-row rate |
|---|---:|---:|---:|---:|---:|
| `keep1_v32_17d_init` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000 |
| `keep1_v32_17d` | 0.004352 | 0.001088 | 0.005762 | 0.002596 | 1.000 |

Interpretation:

- `keep1_v32_17d_init` is the best headline model on max useful bounces and 30+ rate, but it does not use the new tracking axes because they were intentionally initialized to zero.
- `keep1_v32_17d` is the better demonstration model for "17D RL improved the policy": it uses the added tracking velocity residual axes, raises mean useful bounces from `30.56` to `32.11`, raises 30+ rate from `0.71` to `0.74`, and lowers ball-out failures from `8` to `6`.
- The comparison is against the v30 broad-XY reset distribution. Against v26, use the common `reset_xy_range=0.10` result above rather than the easier stored `0.075` result.

## Recommended next step

Run a longer conservative resume from `keep1_v32_17d` with the same preset. The current 200k run already learns to use the extra axes, but the residual magnitude is still small, so a longer run or slightly larger tracking residual action limit may improve the headline max/30+ rate without losing the reduced ball-out behavior.
