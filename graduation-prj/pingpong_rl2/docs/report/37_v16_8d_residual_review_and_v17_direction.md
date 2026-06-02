# v16 8D residual review and v17 direction

작성일: 2026-06-03

## 배경

`pmk_cf_self_rally_v16`은 `position_contact_frame_velocity_residual` 8D action mode를 처음 적용한 run이다.

Action layout:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual]
```

목표는 v15에서 부족했던 방향/높이 제어를 policy가 일부 직접 보정하게 하는 것이었다.

## 분석 대상

학습:

- run: `pmk_cf_self_rally_v16`
- timesteps: `1,000,000`
- action mode: `position_contact_frame_velocity_residual`
- mode: 새 모델 학습

분석 파일:

- best model analysis:
  - `artifacts/ppo_runs/pmk_cf_self_rally_v16/analysis/pmk_cf_self_rally_v16_contact_diagnosis_summary.json`
- final model analysis:
  - `artifacts/ppo_runs/pmk_cf_self_rally_v16/analysis/pmk_cf_self_rally_v16_final_contact_diagnosis_summary.json`

최신 final model이 best model보다 약간 좋아서, 아래 판단은 final model 중심이다.

## v16 final 핵심 지표

100 episode 분석:

- `mean_useful_bounces = 0.60`
- `max_useful_bounces = 3`
- `one_or_more_useful_bounce_rate = 0.38`
- `two_or_more_useful_bounce_rate = 0.16`
- `three_or_more_useful_bounce_rate = 0.06`
- `failure_counts`
  - `ball_out_of_bounds = 75`
  - `low_apex_contact = 5`
  - `floor_contact = 8`
  - `ball_speed_limit = 9`
  - `time_limit = 3`

Contact quality:

- total contacts: `1099`
- useful contact rate: `0.0546`
- mean projected apex: `0.216m`
- median projected apex: `0.190m`
- target apex: `0.300m`
- upward contacts below target: `0.754`
- mean outgoing velocity XY error: `0.208`
- mean outgoing velocity Z error: `0.742`
- next intercept reachable rate: `0.449`
- mean easy-next-ball score: `0.476`

## v14/v15/v16 비교

요약:

| run | episodes | mean useful | max useful | one+ | two+ | three+ | main failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v14 | 100 | 0.85 | 6 | 0.49 | 0.19 | 0.09 | ball_out_of_bounds 61, low_apex 19 |
| v15 | 50 | 0.10 | 1 | 0.10 | 0.00 | 0.00 | ball_out_of_bounds 40 |
| v16 final | 100 | 0.60 | 3 | 0.38 | 0.16 | 0.06 | ball_out_of_bounds 75 |

해석:

- v16은 v15보다 확실히 좋아졌다.
- v16은 lateral speed/XY error가 v15보다 내려가서 viewer에서 부드러워 보이는 것이 지표와 맞다.
- 하지만 v14 최고치보다 낮고, 여전히 `ball_out_of_bounds`가 압도적이다.
- low-apex terminal은 줄었지만, 접촉별 apex는 아직 낮다. 종료 이유만 보면 괜찮아 보일 수 있으나, 실제 upward contact의 `75.4%`가 target apex 아래다.

## 8D action 사용 양상

v16 final contact rows 기준:

앞 5D:

- `radial`: mean `+0.0122`, limit `+-0.02`
- `tangent`: mean `-0.0200`, limit `+-0.02`, saturation rate `1.000`
- `z`: mean `-0.0033`, limit `+-0.03`
- `pitch`: mean `-0.0040`, limit `+-0.004`, saturation rate `1.000`
- `roll`: mean `+0.0040`, limit `+-0.004`, saturation rate `1.000`

새 3D:

- `vz_scale`: mean `+0.0128`
  - 실제 scale 약 `1.013x`
  - limit `+-0.35` 대비 매우 작다.
- `outgoing_x_residual`: mean `-0.0407 m/s`
  - limit `+-0.35 m/s` 대비 작다.
- `outgoing_y_residual`: mean `-0.0004 m/s`
  - 사실상 거의 안 쓴다.

해석:

- 기존 contact-point/tilt residual 일부는 이미 limit에 닿았다.
- 반대로 새 velocity residual 축은 열렸지만 충분히 쓰이지 않는다.
- 따라서 단순히 8D 축을 더 오래 학습시키는 것만으로는 부족할 수 있다.
- 다음 단계는 "새 축을 더 잘 탐색하게 하는 학습 설정"과 "정책이 실제 racket velocity/tilt primitive에 더 직접 개입하는 구조"가 필요하다.

## 타격 속도/타이밍 병목

v16 final contact 기준:

- desired outgoing z: mean `2.205 m/s`
- actual outgoing z: mean `1.644 m/s`
- outgoing z error: mean `0.742`
- controller-side desired z: mean `2.301`
- target velocity z: mean `0.948`
- actual racket velocity z: mean `0.454`
- planner intercept time: mean `0.036s`
- contact ball height above racket: mean `0.024m`

해석:

- desired outgoing z는 충분히 높게 설정되어 있다.
- 하지만 target velocity와 실제 racket velocity가 그만큼 올라가지 못한다.
- contact planner가 평균 `36ms` 전 접촉을 잡고 있어서, 라켓이 충분히 위로 가속할 시간이 부족하다.
- 그래서 height problem은 reward만의 문제가 아니라 "너무 늦은 타격 준비 / velocity execution bottleneck"도 포함한다.

## 현재 결론

v16은 좋은 방향의 첫 단계다.

- v15보다 훨씬 안정적이고 부드럽다.
- 새 8D 구조는 깨지지 않았고, final model은 best model보다 약간 낫다.
- 하지만 목표인 안정적 self-juggling에는 아직 부족하다.

핵심 병목:

1. `ball_out_of_bounds`가 여전히 주 실패다.
2. target apex보다 낮은 contact가 너무 많다.
3. tangent/pitch/roll action이 포화된다.
4. 새 velocity residual action은 거의 쓰이지 않는다.
5. contact planner가 너무 늦게 strike를 만들고 있다.

## v17 방향

아무 차원이나 늘리면 안 된다. 현재 지표에 맞춰 아래 순서가 타당하다.

### 1. contact timing / strike plane 개선

우선순위: 매우 높음.

현재 평균 planner intercept time이 `0.036s`라서 너무 늦다.

후보:

- strike plane offset을 configurable하게 만들기
- v17 preset에서 contact plane을 현재 `0.02m`보다 높게 설정
- preparation/readiness가 더 일찍 켜지도록 activation height 조정

목표:

- controller가 실제 racket upward velocity를 만들 시간을 확보한다.
- low apex를 reward로만 때우지 않는다.

### 2. direct racket velocity residual 추가

우선순위: 매우 높음.

`vz_scale`은 desired outgoing velocity를 살짝 조정하지만, 실제 target velocity / racket velocity까지 충분히 전달되지 않는다.

후보 action:

```text
racket_vz_residual
```

적용 위치:

- `_contact_frame_velocity_target()` 결과에 직접 더한다.

이유:

- 현재 actual racket z velocity mean은 `0.454m/s`로 낮다.
- desired outgoing z를 조정하는 간접 경로보다 직접 racket target velocity를 보정하는 편이 학습 신호가 명확하다.

### 3. tilt primitive scale 추가 또는 tilt residual 한계 조정

우선순위: 높음.

v16 final에서 pitch/roll residual이 각각 limit에 붙었다.

후보:

```text
trajectory_tilt_scale
centering_tilt_scale
```

또는:

- `tilt_action_limit`을 `0.004 -> 0.006~0.008` 범위로 완화

주의:

- raw tilt를 크게 열면 lateral instability가 다시 커질 수 있다.
- scale 방식이 더 안전하다.

### 4. PPO 탐색/업데이트 완화

우선순위: 높음.

v16 설정:

- `learning_rate = 1e-5`
- `n_epochs = 1`
- `clip_range = 0.05`
- `log_std_init = -3.0`
- `contact_frame_action_penalty_weight = 0.2`

이 설정은 안정적이지만 새 velocity 축 탐색에는 보수적이다.

후보:

- `learning_rate = 2e-5 ~ 3e-5`
- `n_epochs = 2`
- `clip_range = 0.08`
- `log_std_init = -2.5`
- `contact_frame_action_penalty_weight = 0.08 ~ 0.12`

목표:

- 새 action dimension을 실제로 쓰게 만든다.
- 단, 너무 aggressive하게 가면 v14처럼 lateral out-of-bounds가 커질 수 있으므로 1M gate로 확인한다.

## 다음 작업 전 검증 기준

v17에서 확인할 지표:

- `mean_useful_bounces >= 0.8`
- `two_or_more_useful_bounce_rate >= 0.20`
- `three_or_more_useful_bounce_rate >= 0.08`
- `ball_out_of_bounds <= 60/100`
- `upward_contact_projected_apex_below_target_rate <= 0.65`
- `actual_outgoing_velocity_z`가 desired z에 가까워지는지
- `racket_velocity_z` mean이 올라가는지
- 새 velocity residual/action scale이 0 근처에 묶이지 않는지

## 판단

다음은 v16을 더 오래 학습시키기보다 v17 구조 수정이 맞다.

이유:

- final model에서도 action 포화와 새 축 미사용이 동시에 보인다.
- 이 상태로 2M을 더 돌리면 tangent/pitch/roll 포화 정책이 더 굳을 가능성이 있다.
- 높이 문제는 desired velocity 부족이 아니라 execution/timing 부족이다.

따라서 v17은 contact timing + direct racket velocity residual + 더 유연한 tilt/탐색 설정을 함께 적용해야 한다.
