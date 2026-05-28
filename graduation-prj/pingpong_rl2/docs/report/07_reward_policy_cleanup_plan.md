# pingpong_rl2 reward/policy cleanup plan

## 1. 목적

지금 필요한 것은 reward를 더 붙이는 것이 아니라, 기본 학습 경로를 하나로 고정하고 실험 해석이 가능하도록 만드는 것이다.

최종 목표는 아래 한 문장으로 고정한다.

> 공을 위로 올려친 뒤, 다시 칠 수 있는 strike zone으로 돌아오게 만든다.

이 문장에 직접 답하지 못하는 옵션 추가나 reward 추가는 기본 경로에 넣지 않는다.

## 2. 지금까지 뒤섞였던 부분

- `position`, `position_strike`, `position_tilt`가 run마다 섞였다.
- `strike_tilt_ramp_pitch`, `strike_tilt_assist_limit`, `include_velocity_domain_observation`, `outgoing_x_term`이 동시에 바뀐 run이 있었다.
- `pmk_tnpv_v2`처럼 이름과 달리 실제 config가 단순 `position` baseline인 run이 있었다.
- projected apex metric은 추가됐지만, target XY 정의가 아직 strike-zone-return 목표와 완전히 일치하지 않는다.

## 3. 기본 학습 경로

### 3.1 기본 baseline

- 기본 baseline action mode: `position_strike`
- `position`은 비교용 baseline으로만 유지한다.
- `position_tilt`는 당분간 실험용으로만 남긴다.

이유:

- 현재 bottleneck은 `공 아래로 들어가기 + 올라치기 timing`이다.
- 이 문제는 `position_strike`가 가장 직접적으로 다룬다.
- `position_tilt`는 아직 chatter/해석 불가 경향이 더 크다.

### 3.2 기본으로 고정할 학습 옵션

- `n_envs=4`
- `n_steps=256`
- `batch_size=256`
- `learning_rate=3e-4`
- `gamma=0.99`
- `ball_height=0.5`
- `max_episode_steps=600`
- `reset_xy_range=0.06`
- `reset_velocity_xy_range=0.01`
- `reset_velocity_z_range=(-0.02, 0.01)`
- `success_velocity_threshold=0.5`
- `tracking_during_contact_scale=0.0`

실험에서 매번 직접 바꿀 필수 옵션은 아래만 남긴다.

- `--run-name`
- `--run-version`
- `--reset-model` 또는 resume 여부
- `--total-timesteps`
- `--seed`

## 4. reward 정리

### 4.1 유지

- `tracking_term`
  - descending strike window에서 공 아래로 들어가도록 만드는 기본 dense signal
- `contact_bonus`
  - useful upward contact event를 강화
- `apex_match_term`
  - 위로 친 공의 품질을 높이 기반으로 보정
- `failure_penalty`
  - floor/body/out-of-bounds 종료 억제

### 4.2 기본값 0 유지

- `outgoing_x_term`
  - 평균 `vx`를 줄일 수는 있지만, 공이 실제로 다음 strike zone으로 돌아오는지는 보장하지 못했다.
  - baseline에서는 끄고, metric으로 strike-zone-return을 먼저 정의한 뒤에만 재검토한다.

### 4.3 실험용으로만 유지

- `tilt_angle_penalty`
- `tilt_action_delta_penalty`

이 둘은 `position_tilt` 실험에서만 의미가 있다.

## 5. CLI preset 정리

`run_ppo_learning.py`는 이제 아래 preset을 지원한다.

- `baseline_position`
  - `action_mode=position`
- `strike_position`
  - `action_mode=position_strike`
- `strike_velocity_obs`
  - `action_mode=position_strike`
  - `include_velocity_domain_observation=True`
- `tilt_experiment`
  - `action_mode=position_tilt`
  - `tilt_profile=auto`
- `final_candidate`
  - `action_mode=position_strike`
  - `strike_tilt_ramp_pitch=-0.03`
  - `strike_tilt_ramp_xy_tolerance=0.04`
  - `include_velocity_domain_observation=True`

권장 사용 방식:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset strike_position \
  --run-name clean_strike \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

preset과 충돌하는 수동 옵션을 같이 넘기면 에러로 막아서 실험 confound를 줄인다.

추가로 학습 hygiene를 위해 아래 checkpoint 옵션을 기본 지원한다.

- `--checkpoint-interval`
  - 주기적으로 checkpoint를 저장하고 interim evaluation을 남긴다.
- `--checkpoint-eval-episodes`
  - checkpoint 비교용 deterministic evaluation episode 수
- `--early-stop-patience-evals`
  - 개선 없는 checkpoint evaluation이 누적되면 조기 종료

기본 원칙:

- final model만 저장하지 않는다.
- `best_model.zip`을 함께 저장한다.
- zone-return 기준으로는 `two_or_more_useful_bounce_rate`를 최우선으로 본다.

## 6. projected apex metric 정리

`run_ppo_rebound_analysis.py`는 이제 아래를 지원한다.

- `--apex-target`
  - `controller_anchor`
  - `racket_home`
  - `racket_position`
  - `target_position`
- `--compare-apex-targets`
  - 모든 target 후보에 대한 `projected_apex_xy_error` 평균을 summary에 함께 기록

추가로 `--model-path`로 `*_best_model.zip` 또는 `checkpoints/*_step_*_model.zip`를 넘겨도,
원래 training run의 `training_summary.json`에서 env config를 복원하도록 수정했다.
이제 checkpoint model 분석이 기본 `position` env로 잘못 fallback되지 않는다.

추가로 summary에는 아래도 함께 기록한다.

- `episodes_with_two_or_more_useful_bounces`
- `two_or_more_useful_bounce_rate`
- target별 first-contact projected-apex error가 `2회 이상 useful bounce` episode와 그렇지 않은 episode를 얼마나 구분하는지

현재 해석 원칙:

- projected apex metric은 계속 metric으로만 둔다.
- 어떤 target이 useful contact / out-of-bounds / second useful contact와 가장 잘 맞는지 먼저 확인한다.
- 그 다음에만 `apex_xy_term` reward 승격을 검토한다.

## 7. 다음 clean ablation

다음 비교는 reward, reset, PPO hyperparameter, seed를 모두 고정하고 아래 네 개만 비교한다.

### 7.1 position baseline

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset baseline_position \
  --run-name clean_pos \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

### 7.2 strike baseline

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset strike_position \
  --run-name clean_strike \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

### 7.3 timed negative pitch

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --run-name clean_tnp \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7 \
  --action-mode position_strike \
  --strike-tilt-ramp-pitch -0.03 \
  --strike-tilt-ramp-xy-tolerance 0.04
```

### 7.4 velocity-domain observation

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset strike_velocity_obs \
  --run-name clean_vobs \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

각 run 뒤에는 반드시 아래 분석을 붙인다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name clean_strike \
  --run-version v1 \
  --episodes 50 \
  --compare-apex-targets
```

핵심 지표:

- `mean_useful_bounces`
- `max_useful_bounces`
- `failure_counts`
- `ball_out_of_bounds` 비율
- useful contact rate
- first/useful contact 이후 `ball_velocity_x/y/z`
- `projected_apex_xy_error`
- 두 번째 useful contact 발생 비율

## 8. 아직 1M을 돌리면 안 되는 이유

- `pmk_tnpv_v2`는 1M run이지만 구조 선택 문제에 답을 주지 못했다.
- `pmk_tnpv_v1` resume도 성능을 개선하지 못했다.
- reward를 더 붙이기 전에, 어떤 control/observation 조합이 실제로 반복 keep-up에 기여하는지 아직 clean하게 분리되지 않았다.

1M을 돌려도 되는 조건은 아래 둘을 동시에 만족할 때다.

- 100k 또는 300k clean ablation에서 한 후보가 baseline보다 `mean_useful_bounces`와 useful contact rate를 올린다.
- 같은 후보가 `ball_out_of_bounds`와 outward rebound 경향을 악화시키지 않는다.

## 9. 현재 결론

한 줄로 정리하면 아래다.

> `position_strike`를 기준 경로로 고정하고, `outgoing_x_term`은 접고, strike-zone-return metric을 먼저 제대로 만든 뒤 clean ablation으로 무엇이 실제로 keep-up을 개선하는지 다시 확인해야 한다.

## 10. checkpoint-selection 후속 결론

checkpoint hygiene를 넣고 다시 돌려본 결과, 현재는 `final model`보다 `best checkpoint`를 유지하는 것이 확실히 낫다.

- `clean_tnp_ckpt_v1`
  - `30k` checkpoint가 최적이었다.
  - 50-episode rebound analysis에서 `best_model.zip`은 `mean_useful_bounces=0.34`, `one_or_more_useful_bounce_rate=0.34`, useful-contact rate `14.9%`였다.
  - 같은 run의 final model은 `mean_useful_bounces=0.18`, `one_or_more_useful_bounce_rate=0.18`, useful-contact rate `7.6%`로 분명히 더 나빴다.
- 따라서 이 계열에서는 `더 오래 학습`보다 `중간 checkpoint 보존/선택`이 먼저다.

하지만 더 짧고 촘촘한 second cycle(`clean_tnp_ckpt_v2`, `40k`, `5k` interval)는 `v1 best`를 넘지 못했다.

- 장점:
  - `two_or_more_useful_bounce_rate=0.04`로 처음으로 2-bounce episode가 50-episode 평가에서 관측됐다.
  - `ball_out_of_bounds`는 `37 -> 34`로 소폭 줄었다.
- 단점:
  - `mean_useful_bounces`는 `0.34 -> 0.28`로 하락했다.
  - `one_or_more_useful_bounce_rate`도 `0.34 -> 0.24`로 하락했다.
  - 전체 `mean_ball_lateral_to_vertical_ratio`가 크게 악화돼 rebound 안정성은 오히려 불안정해졌다.

현재 실무 결론:

- 이 시점의 배포/재시작 기준 candidate는 `clean_tnp_ckpt_v1_best_model.zip`이었다.
- `clean_tnp_ckpt_v2_best_model.zip`은 `2회 useful bounce`가 드물게 나온다는 점에서 연구 가치는 있지만, baseline candidate로 교체할 정도는 아니다.
- 다음 변경은 학습 시간을 더 줄이는 튜닝이 아니라, `첫 useful bounce 이후 공이 다시 inward strike-zone으로 돌아오게 하는 구조`를 직접 겨냥해야 한다.

## 11. 첫 inward-return 구조 실험 결과

이 방향의 첫 직접 실험으로, env에 아래 optional term을 추가해 `clean_tnp_return_anchor_v1`를 실행했다.

- `useful_contact_return_target_xy_reward_weight`
- `return_target_xy_source`
- useful contact에서만 projected apex XY를 계산해 target XY와 가까우면 추가 reward를 준다.

첫 실험 설정은 아래였다.

- base policy: `position_strike + strike_tilt_ramp_pitch=-0.03`
- return target: `controller_anchor`
- reward weight: `1.0`
- checkpoint hygiene: `40k`, `5k interval`, `patience=2`

결과는 명확히 regression이었다.

- checkpoint 10-episode eval에서는 `15k` 시점에 `two_or_more_useful_bounce_rate=0.1`가 잠깐 보였다.
- 하지만 50-episode best-model rebound analysis에서는 유지되지 않았다.
- `clean_tnp_return_anchor_v1_best_model.zip`:
  - `mean_useful_bounces=0.08`
  - `one_or_more_useful_bounce_rate=0.08`
  - `two_or_more_useful_bounce_rate=0.0`
  - useful-contact rate `2.9%`
  - `ball_out_of_bounds=41`
- 기존 candidate `clean_tnp_ckpt_v1_best_model.zip`:
  - `mean_useful_bounces=0.34`
  - `one_or_more_useful_bounce_rate=0.34`
  - useful-contact rate `14.9%`
  - `ball_out_of_bounds=37`

해석:

- projected-apex XY reward를 useful-contact event에 직접 추가하는 것만으로는 문제를 풀지 못했다.
- 오히려 useful contact 자체가 크게 줄었고, 전체 rebound 안정성도 악화됐다.
- 따라서 `post-contact return`은 당분간 reward term보다 control/observation 구조 쪽에서 다시 접근해야 한다.

실무 결론:

- 이 reward term은 코드에는 optional 실험 기능으로 남겨두되, baseline에는 넣지 않는다.
- 이 결과는 다시 한 번 `10-episode checkpoint eval`만으로 방향을 결정하면 안 된다는 점을 확인했다. 최종 판단은 계속 `50-episode rebound analysis`로 한다.

## 12. post-contact return assist 제어 실험 결과

reward가 아니라 control 쪽에서 직접 접근하기 위해, `position_strike`에 아래 optional assist를 추가했다.

- `post_contact_return_assist_weight`
- `post_contact_return_max_intercept_time`

의도는 단순하다.

- useful bounce가 이미 한 번 나온 뒤
- 공이 다시 올라가는 동안
- target XY를 무조건 anchor로 되돌리는 대신
- 미래 strike-plane intercept 쪽으로 약하게 bias를 건다.

즉, `첫 useful bounce 이후 공이 다시 inward strike-zone으로 돌아오게 하는 구조`를 reward가 아니라 target generation에서 직접 건드렸다.

### 12.1 v1: assist weight 0.5

`clean_tnp_return_assist_v1`:

- base policy: `position_strike + strike_tilt_ramp_pitch=-0.03`
- `post_contact_return_assist_weight=0.5`
- `post_contact_return_max_intercept_time=0.6`
- checkpoint hygiene: `40k`, `5k interval`, `patience=2`

50-episode best-model 분석에서는 기존 baseline candidate보다 개선됐다.

- baseline `clean_tnp_ckpt_v1_best_model.zip`
  - `mean_useful_bounces=0.34`
  - `one_or_more_useful_bounce_rate=0.34`
  - `two_or_more_useful_bounce_rate=0.0`
  - useful-contact rate `14.9%`
  - `ball_out_of_bounds=37`
- `clean_tnp_return_assist_v1_best_model.zip`
  - `mean_useful_bounces=0.40`
  - `one_or_more_useful_bounce_rate=0.38`
  - `two_or_more_useful_bounce_rate=0.02`
  - useful-contact rate `16.3%`
  - `ball_out_of_bounds=38`

50-episode만 보면 `ball_out_of_bounds`가 1 episode 늘어서 애매했기 때문에, 같은 두 모델을 다시 100-episode로 비교했다.

100-episode 비교 결과:

- baseline `clean_tnp_ckpt_v1_best_model.zip`
  - `mean_useful_bounces=0.31`
  - `one_or_more_useful_bounce_rate=0.31`
  - `two_or_more_useful_bounce_rate=0.0`
  - useful-contact rate `12.97%`
  - `ball_out_of_bounds=78`
- `clean_tnp_return_assist_v1_best_model.zip`
  - `mean_useful_bounces=0.40`
  - `one_or_more_useful_bounce_rate=0.38`
  - `two_or_more_useful_bounce_rate=0.02`
  - useful-contact rate `16.0%`
  - `ball_out_of_bounds=77`

이 비교로 보면, `v1`은 noise가 아니라 실제 개선 방향으로 보는 것이 맞다.

### 12.2 v2: assist weight 0.35

더 약한 assist(`clean_tnp_return_assist_v2`)도 확인했다.

- checkpoint 10-episode eval은 좋아 보였지만
- 50-episode best-model 분석에서는 `two_or_more_useful_bounce_rate=0.04`가 보이는 대신
- useful-contact rate가 `12.4%`로 내려가고 `ball_out_of_bounds=42`로 악화됐다.

즉, `v2`는 short checkpoint eval 기준으로는 과대평가됐고, 실제 candidate로는 부적합했다.

### 12.3 현재 실무 결론

- reward-side inward-return term은 실패했다.
- control-side `post_contact_return_assist`는 유효했다.
- 현재 다음 cycle 기준 candidate는 `clean_tnp_return_assist_v1_best_model.zip`으로 갱신한다.
- `clean_tnp_ckpt_v1_best_model.zip`은 직전 baseline candidate로 유지하되, 기본 비교 기준점으로만 남긴다.
- `clean_tnp_return_assist_v2`는 추가 tuning 후보가 아니라 archive 대상이다.

보조 해석 메모:

- all-contact 기준 `mean_ball_lateral_to_vertical_ratio`는 일부 near-flat contact에 민감해서 흔들릴 수 있다.
- candidate 선택은 계속 `mean_useful_bounces`, `one_or_more/two_or_more useful bounce rate`, useful-contact rate, `ball_out_of_bounds`를 우선한다.
