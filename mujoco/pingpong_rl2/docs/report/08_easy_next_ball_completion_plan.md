# pingpong_rl2 easy-next-ball completion plan

## 1. 현재 active candidate 정합성 확인 결과

현재 active candidate는 `clean_tnp_return_assist_v1_best_model.zip`으로 유지하는 것이 맞다.

- `run_ppo_learning.py`의 `final_candidate` preset은 현재 아래 env config를 가리킨다.
  - `action_mode=position_strike`
  - `strike_tilt_ramp_pitch=-0.03`
  - `strike_tilt_ramp_xy_tolerance=0.04`
  - `post_contact_return_assist_weight=0.5`
  - `post_contact_return_max_intercept_time=0.6`
  - `include_velocity_domain_observation=False`
- `README.md`는 이제 `--preset final_candidate` 기준 실행 경로를 안내한다.
- `07_reward_policy_cleanup_plan.md`도 `clean_tnp_return_assist_v1_best_model.zip`을 현재 candidate로 기록한다.

즉, 코드 preset과 문서는 현재 후보 기준으로 일치한다.

## 2. stale run와 best-model 구분

`clean_final_v1`은 현재 `final_candidate` preset의 근거 run이 아니다. 이 run은 stale reference로 취급해야 한다.

근거:

- `clean_final_v1_training_summary.json`의 `config.preset`은 `final_candidate`지만, 실제 env config는 `include_velocity_domain_observation=true`이고 `post_contact_return_assist_*`가 없다.
- 이는 `final_candidate` preset이 return-assist winner로 재지정되기 전의 예전 config다.

따라서 아래를 구분해야 한다.

- 현재 preset 정의: 코드의 `final_candidate`
- stale historical run: `clean_final_v1`
- 현재 active model candidate: `clean_tnp_return_assist_v1_best_model.zip`

또한 이 프로젝트에서는 final model보다 best checkpoint가 더 좋은 경우가 이미 반복 확인됐다. 현재 기준 candidate도 final model이 아니라 `*_best_model.zip` 기준으로 다뤄야 한다.

## 3. env config 복원 경로 확인

`src/pingpong_rl2/utils/ppo_runs.py`의 아래 경로는 현재 올바르다.

- `infer_training_run_name_from_model_path()`
- `training_summary_candidates_for_model()`
- `load_env_config_for_model()`

이 로직은 `*_best_model.zip`과 `checkpoints/*_step_*_model.zip`에 대해 base run의 training summary에서 `env_config`를 복원한다. 따라서 rebound analysis에서 best/checkpoint model을 잘못된 default env로 여는 이전 문제는 현재 코드 기준으로 막혀 있다.

## 4. final_candidate 실제 env config

현재 `final_candidate` preset은 아래 실험 의도를 고정한다.

- base control: `position_strike`
- pre-contact control bias: timed negative pitch ramp
- post-contact control bias: return assist toward future strike-plane intercept
- observation: velocity-domain observation 기본 off
- reward: reward-side inward-return shaping 기본 off

이 구성은 지금까지의 결과 중 가장 덜 얽히고, 반복 keep-up 목표에 가장 직접적으로 맞는 control-side candidate다.

## 5. 새 metric 정의

이번 작업에서는 reward를 더 넣지 않고, analysis-only metric을 먼저 추가했다.

### 5.1 next-strike intercept metric

contact 이후 공이 다시 strike plane으로 내려오는 시점을 예측한다.

정의:

```text
target_z = controller_anchor_z + tracking_strike_plane_offset
solve z(t) = contact_ball_z + contact_ball_vz * t + 0.5 * g * t^2 = target_z
predicted_next_intercept_time = descending root
predicted_next_intercept_xy = contact_ball_xy + contact_ball_vxy * t
next_intercept_xy_error = ||predicted_next_intercept_xy - controller_anchor_xy||
next_intercept_reachable = next_intercept_xy_error <= strike_zone_xy_radius
```

주의:

- `target_z`는 env의 `_tracking_strike_plane_offset()`와 같은 정의를 쓴다.
- root는 contact 직후의 즉시 교차가 아니라 다음 descending strike를 보려고 positive root 중 큰 값을 쓴다.

### 5.2 easy-next-ball metric

reward가 아니라 summary용 heuristic score다.

```text
easy_next_ball_score =
  + intercept_xy_score
  + 0.75 * intercept_time_score
  + 0.5 * descending_speed_score
  - 0.5 * lateral_speed_penalty
  - 0.25 * excessive_speed_penalty
  - 0.5 * recovery_distance_penalty
```

의도는 단순하다.

- 다음 strike가 너무 멀지 않고
- 너무 급하지 않으며
- 내려오는 속도가 지나치게 크지 않고
- lateral drift가 과하지 않은 공을 더 높은 점수로 본다.

### 5.3 contact quality metric

contact CSV와 summary에 아래 analysis-only metric도 추가했다.

- `contact_relative_speed_norm`
- `contact_normal_relative_speed`
- `contact_tangential_relative_speed`
- `contact_tangential_relative_ratio`

이를 위해 contact-time `ball_position_z`와 `racket_face_normal`도 trace에 남기도록 했다.

## 6. metric-only 분석 결과

새 analysis는 아래 두 결과 파일로 생성했다.

- baseline: `clean_tnp_ckpt_v1_best_easy50_summary.json`
- active candidate: `clean_tnp_return_assist_v1_best_easy50_summary.json`

핵심 비교:

| model | mean useful bounces | useful contact rate | mean next-intercept XY error | useful-contact next-intercept XY error | next-intercept reachable rate | useful-contact reachable rate | mean easy-next-ball score | useful-contact easy-next-ball score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_tnp_ckpt_v1_best_model.zip` | `0.34` | `14.9%` | `0.202` | `0.215` | `26.7%` | `0.0%` | `-0.003` | `-0.620` |
| `clean_tnp_return_assist_v1_best_model.zip` | `0.40` | `16.3%` | `0.185` | `0.151` | `32.1%` | `25.0%` | `0.095` | `-0.267` |

해석:

- `return_assist_v1`는 baseline보다 전체 next-strike intercept quality가 좋아졌다.
- useful contact만 따로 봐도 `next_intercept_xy_error`가 `0.215 -> 0.151`로 내려갔다.
- useful-contact reachable rate가 baseline `0.0%`에서 `25.0%`로 올라갔다.
- easy-next-ball score도 평균 기준으로 `-0.003 -> 0.095`, useful-contact 기준으로 `-0.620 -> -0.267`로 개선됐다.

즉, 새 metric은 현재 control-side candidate가 baseline보다 “다음 공을 다시 칠 수 있는 쪽”으로 더 가고 있다는 점을 지지한다.

## 7. 아직 reward로 올리면 안 되는 이유

좋은 신호만 있는 것은 아니다.

episode first-contact 기준으로 보면, 아직 metric이 useful-bounce 지속성과 같은 방향으로 정렬되지 않는다.

| model | first-contact one+ error gap | first-contact one+ easy-score gap |
| --- | ---: | ---: |
| `clean_tnp_ckpt_v1_best_model.zip` | `+0.131` | `-1.073` |
| `clean_tnp_return_assist_v1_best_model.zip` | `+0.052` | `-0.814` |

여기서 gap은:

- `one+ useful bounce episode 평균` `-` `zero useful bounce episode 평균`

이다.

이 값이 뜻하는 바는 아래다.

- 아직 first contact만 놓고 보면, useful-bounce가 나온 episode의 next-intercept가 zero-bounce episode보다 더 좋은 쪽으로 분리되지 않는다.
- 다만 `return_assist_v1`에서는 이 역방향 gap이 baseline보다 작아졌다.

실무 해석:

- metric은 analysis signal로는 유효하다.
- 하지만 아직 reward로 바로 승격할 정도로 causal signal이 깨끗하지는 않다.
- 따라서 현재 단계에서는 control-side assist 보완이 reward 추가보다 우선이다.

## 8. contact quality에서 보이는 것

contact quality metric은 해석 가치가 있다.

- baseline 전체 `mean_contact_tangential_relative_ratio=0.366`
- baseline useful-contact `mean_contact_tangential_relative_ratio=0.091`
- return-assist 전체 `mean_contact_tangential_relative_ratio=0.257`
- return-assist useful-contact `mean_contact_tangential_relative_ratio=0.073`

즉 useful contact는 이미 “상대속도의 tangential 성분이 작은 contact”와 강하게 연결된다. 이건 contact quality logging이 헛 metric이 아니라는 뜻이다.

다만 이 값만으로 바로 reward를 만들지는 말고, next-intercept metric과 같이 보면서 control 변경이 contact quality를 어떻게 바꾸는지 먼저 보는 것이 맞다.

## 9. 다음 보완 방향

현재 최우선은 reward가 아니라 control-side assist 미세조정이다.

즉시 수행한 첫 follow-up도 이미 하나 있다.

- `clean_tnp_return_assist_w06_v1`
  - 변경점: `post_contact_return_assist_weight 0.5 -> 0.6`
  - 나머지 설정: 동일
  - 결과: keep-up aggregate 숫자는 일부 비슷하거나 소폭 좋아 보였지만, easy-next-ball metric은 명확히 악화됐다.

`0.5` 대 `0.6` 50-episode 비교:

| model | mean useful bounces | one+ rate | two+ rate | ball out of bounds | useful next-intercept error | useful reachable rate | useful easy score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| assist `0.5` | `0.40` | `0.38` | `0.02` | `38` | `0.151` | `25.0%` | `-0.267` |
| assist `0.6` | `0.40` | `0.40` | `0.00` | `36` | `0.359` | `0.0%` | `-0.711` |

해석:

- `0.6`은 `one_plus_rate`와 `ball_out_of_bounds`만 보면 조금 좋아 보일 수 있다.
- 하지만 useful contact가 만든 다음 공의 질은 오히려 크게 나빠졌다.
- 특히 `useful_contact_mean_next_intercept_xy_error`가 `0.151 -> 0.359`로 악화됐고, useful-contact reachable rate가 `25% -> 0%`로 떨어졌다.
- `two_or_more_useful_bounce_rate`도 `0.02 -> 0.00`으로 내려갔다.

따라서 현재 active candidate는 그대로 assist `0.5`를 유지하고, 다음 탐색은 `0.4` 또는 `max_intercept_time` 쪽이 우선이다.

우선순위:

1. `post_contact_return_assist_weight`를 `0.4`, `0.6`, `0.7` 중 하나씩만 바꿔 본다.
2. `post_contact_return_max_intercept_time`를 `0.5`, `0.7`로 하나씩만 바꿔 본다.
3. 새 metric으로 아래를 같이 본다.
   - `mean_next_intercept_xy_error`
   - `useful_contact_mean_next_intercept_xy_error`
   - `next_intercept_reachable_rate`
   - `useful_contact_next_intercept_reachable_rate`
   - `mean_easy_next_ball_score`
4. reward-side easy-next-ball term은 위 지표가 one+/two+ useful-bounce와 더 같은 방향으로 정렬될 때만 검토한다.

지금 당장 피할 것:

- global `vx` penalty 재튜닝
- `position_tilt` 재도입
- velocity-domain observation 재혼합
- reward 항 2개 이상 동시 추가

## 10. 다음 실험 명령어

학습:

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset final_candidate \
  --run-name <run_name> \
  --run-version <version> \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7
```

metric-only analysis:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/<run_name>_<version>/<run_name>_<version>_best_model.zip \
  --episodes 50 \
  --analysis-name <run_name>_<version>_easy50 \
  --compare-apex-targets
```

## 11. 1M 학습 가능 여부

현재는 아직 아니다.

이유:

- active candidate가 좋아지긴 했지만 `two_or_more_useful_bounce_rate`는 아직 `0.02` 수준이다.
- `ball_out_of_bounds`가 여전히 높다.
- easy-next-ball metric은 전체 방향성은 지지하지만, first-contact 기준으로는 아직 useful-bounce 지속성과 같은 방향으로 정렬되지 않는다.
- 따라서 reward 승격 없이 control-side 미세조정과 metric 재확인이 먼저다.

1M으로 올려도 되는 최소 조건은 아래다.

- 100k 또는 300k에서 current candidate보다 `mean_useful_bounces` 개선
- `one_or_more_useful_bounce_rate` 유지 또는 개선
- `two_or_more_useful_bounce_rate` 유지 또는 개선
- `ball_out_of_bounds` 악화 없음
- `useful_contact_mean_next_intercept_xy_error` 개선
- `useful_contact_next_intercept_reachable_rate` 개선
- easy-next-ball metric이 first-contact 기준으로도 덜 역방향이거나 정방향으로 바뀜
