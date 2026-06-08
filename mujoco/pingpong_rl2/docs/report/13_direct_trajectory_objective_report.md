# direct trajectory objective report

## 1. why the old tuning direction was insufficient

이번 단계의 핵심 판단은 아래였다.

- `assist weight`, `vx penalty`, `bootstrap filter`를 조금씩 바꾸는 것만으로는 `2+ useful bounce` 문제를 직접 닫지 못한다.
- 지금까지의 보조 metric들은 `다음 공이 다시 strike zone으로 돌아오는 물리적으로 올바른 outgoing velocity`를 직접 정의하지 않았다.
- 특히 `projected_apex_xy`나 `outgoing_x` 하나만 보면, 공이 위로는 가지만 다음 strike zone으로는 잘 돌아오지 않는 경우를 충분히 걸러내지 못한다.

이번 작업에서는 이 missing objective를 아래처럼 다시 잡았다.

- contact 직후 공 위치: `p_contact`
- 목표 apex XY: `controller_anchor_xy`
- 목표 apex Z: `controller_anchor_z + target_ball_height`
- desired outgoing velocity:

```python
gravity = abs(model.opt.gravity[2])
height_delta = max(target_apex_z - contact_ball_z, 0.01)
desired_vz = sqrt(2.0 * gravity * height_delta)
time_to_apex = desired_vz / gravity
desired_vxy = (target_xy - contact_ball_xy) / max(time_to_apex, 1e-6)
```

즉 문제 정의를 `공을 안쪽으로 보내라`가 아니라 `다음 strike zone으로 돌아오는 outgoing velocity를 만들라`로 바꿨다.

## 2. implementation added in this cycle

이번 턴에서 추가한 것은 reward 미세조정이 아니라 먼저 계측과 판정 루트였다.

- `src/pingpong_rl2/envs/keepup_env.py`
  - contact 시점 `desired_outgoing_velocity_x/y/z`
  - `actual_outgoing_velocity_x/y/z`
  - `outgoing_velocity_error_norm`
  - `outgoing_velocity_xy_error`
  - `outgoing_velocity_z_error`
  - `desired_time_to_apex`
  - `desired_outgoing_target_x/y`
  - `predicted_apex_x/y_from_actual_velocity`
  - `predicted_apex_xy_error`
- `scripts/run_ppo_rebound_analysis.py`
  - contact summary에 outgoing trajectory error 집계 추가
  - `2+ useful bounce episode` vs `zero useful bounce episode` contact mean error summary 추가
- `scripts/run_heuristic_keepup_diagnostic.py`
  - heuristic summary에 outgoing trajectory error 집계 추가
  - `three_or_more_useful_bounce_rate` 추가
- `scripts/run_ppo_learning.py`
  - 실험용 trajectory reward/observation plumbing 추가
  - `--trajectory-match-reward-weight`
  - `--include-desired-outgoing-velocity-observation`

중요한 점은, 이 CLI는 들어갔지만 이번 결과 기준으로 새 reward branch는 아직 승격하지 않았다.

## 3. metric-only result on existing best models

아래 4개 best model을 모두 50-episode로 다시 돌렸다.

- `clean_tnp_return_assist_v1_best_model.zip`
- `followup_strike_contract_v1_best_model.zip`
- `followup_strike_bootstrap_v1_best_model.zip`
- `followup_bootstrap_resume_contract_v1_best_model.zip`

핵심 비교는 아래다.

| model | mean useful bounces | two+ rate | all-contact mean outgoing error | useful-contact mean outgoing error | 2+ episode contact mean outgoing error | zero-bounce episode contact mean outgoing error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_tnp_return_assist_v1_best` | `0.40` | `0.02` | `1.5963` | `0.4583` | `0.6271` | `1.7790` |
| `followup_strike_contract_v1_best` | `0.28` | `0.04` | `1.7569` | `0.5321` | `0.7437` | `1.8891` |
| `followup_strike_bootstrap_v1_best` | `0.50` | `0.02` | `1.7980` | `0.7308` | `0.8781` | `1.9663` |
| `followup_bootstrap_resume_contract_v1_best` | `0.30` | `0.04` | `1.5681` | `0.5810` | `0.8510` | `1.6299` |

해석:

- 네 모델 모두에서 `2+ useful bounce episode`의 contact mean outgoing velocity error가 `zero-bounce episode`보다 낮았다.
- 즉 `desired outgoing velocity error`는 현재 keep-up 성공 여부와 실제로 상관이 있다.
- 이건 기존 `apex XY only`나 `vx only`보다 더 직접적인 신호다.

반대로 `predicted_apex_xy_error`는 일관성이 약했다.

- 몇 run에서는 `2+` episode가 오히려 더 큰 apex XY error를 보였다.
- 따라서 이번 단계의 결론은 `projected apex XY`를 reward로 바로 올리는 것이 아니라, `full outgoing velocity error`를 우선 objective로 봐야 한다는 쪽이다.

## 4. scripted controller feasibility

같은 direct trajectory metric으로 heuristic baseline도 다시 확인했다.

run:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name heuristic_followup_trajectory_100ep_v1 \
  --variant-name followup_trajectory \
  --episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --followup-strike-target-tilt -0.03 0.0
```

결과:

- `mean_useful_bounces=0.54`
- `two_or_more_useful_bounce_rate=0.05`
- `three_or_more_useful_bounce_rate=0.00`
- `all_contact_mean_outgoing_velocity_error_norm=1.7024`
- `useful_contact_mean_outgoing_velocity_error_norm=0.7077`
- `two_or_more_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm=0.6685`
- `zero_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm=1.9701`

판단:

- scripted controller도 direct trajectory target과 keep-up success 사이의 상관은 보여준다.
- 즉 target 정의 자체가 완전히 틀린 것은 아니다.
- 하지만 deterministic / narrow-reset 조건에서도 `3+`를 못 만들었다.
- 따라서 reward만 얹으면 해결된다고 보기 어렵고, 아직 control/physics ceiling이 남아 있다.

이번 단계의 중요한 결론은 아래 두 가지를 동시에 만족한다는 점이다.

1. direct trajectory metric은 분석용 objective로 유효하다.
2. 그러나 현재 heuristic/controller만으로도 stable `3+`는 아직 안 된다.

## 5. reward promotion experiment

metric이 유효하므로, 새 reward branch는 한 번만 직접 확인했다.

실험 설정:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset followup_strike_candidate \
  --run-name trajectory_match \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7 \
  --checkpoint-interval 5000 \
  --checkpoint-eval-episodes 10 \
  --eval-episodes 10 \
  --early-stop-patience-evals 2 \
  --trajectory-match-reward-weight 0.3 \
  --include-velocity-domain-observation \
  --include-desired-outgoing-velocity-observation
```

reward 형태:

```python
trajectory_match_term = weight * exp(-outgoing_velocity_error_norm)
```

적용 조건:

- `contact_event`
- `actual_outgoing_velocity_z > 0`
- useful contact 조건에는 묶지 않음

### 5.1 short checkpoint result

best short checkpoint는 `20k`였다.

- `mean_useful_bounces=0.3`
- `one+ rate=0.3`
- `two+ rate=0.0`

즉, early signal 단계부터 기존 best directions보다 강하지 않았다.

### 5.2 50-episode rebound result

best model 50-episode 결과:

- `mean_return=-3.0788`
- `mean_useful_bounces=0.12`
- `one_or_more_useful_bounce_rate=0.12`
- `two_or_more_useful_bounce_rate=0.00`
- `useful_contact_rate=5.66%`
- `ball_out_of_bounds=41/50`
- `all_contact_mean_outgoing_velocity_error_norm=1.5266`
- `useful_contact_mean_outgoing_velocity_error_norm=0.9292`

비교 해석:

- all-contact mean outgoing error는 일부 baseline보다 약간 좋아졌다.
- 하지만 useful-contact mean outgoing error는 오히려 나빠졌다.
- 실제 keep-up 목표 `mean useful bounces`, `one+`, `two+`, useful-contact rate는 모두 크게 후퇴했다.

즉 이 reward는 지금 단계에서 `좋은 strike`를 더 많이 만드는 것이 아니라, `non-useful contact`까지 포함한 contact distribution을 다른 쪽으로 밀었을 가능성이 크다.

## 6. decision: reward is not promoted

이번 direct trajectory objective cycle의 판단은 아래다.

- `desired outgoing velocity error`는 analysis metric으로 유지한다.
- scripted diagnostic에서도 이 metric은 유효했다.
- 하지만 현재 control/physics 조건에서는 heuristic조차 `3+`를 못 만들고 있다.
- 그 상태에서 바로 reward로 승격한 첫 PPO branch `trajectory_match_v1`은 50-episode 기준으로 명확한 regression이었다.

따라서 이번 단계의 결론은:

- `trajectory_match_reward_weight` branch는 현재 default로 승격하지 않는다.
- `followup_bootstrap_resume_contract_v1_best`를 main training baseline으로 유지한다.
- direct trajectory objective는 reward보다 먼저 analysis/control diagnostic objective로 사용한다.

## 7. what to continue and what to discard

### continue

- `outgoing_velocity_error_norm`을 기준으로 candidate를 읽는 것
- `followup_strike_candidate` control contract
- `followup_bootstrap_resume_contract_v1_best` staged training baseline
- heuristic / rebound analysis에서 `2+ episode error < zero-bounce episode error`를 gate로 쓰는 것

### discard for now

- `trajectory_match_v1` reward branch as current training direction
- `projected_apex_xy_error`를 단독 reward로 승격하는 것
- heuristic이 아직 `3+`도 못 만드는 상태에서 reward를 더 키우는 것

## 8. next minimal direction

다음 우선순위는 reward를 더 넣는 것이 아니다.

1. heuristic/controller가 deterministic 또는 narrow-reset에서 `3+`를 만들도록 control-side contact geometry를 먼저 높인다.
2. direct trajectory metric은 그 control change를 판정하는 gate로 계속 쓴다.
3. 그 다음에만 trajectory reward를 다시 시도한다.

지금 단계의 한 줄 결론:

> direct outgoing trajectory는 분석 objective로는 맞다. 하지만 현재 controller ceiling이 낮아서, 이걸 reward로 바로 올린 첫 PPO branch는 keep-up 목표를 오히려 망쳤다.
