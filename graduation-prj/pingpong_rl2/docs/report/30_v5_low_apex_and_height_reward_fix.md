# 30. v5 낮은 apex 병목과 height reward 보강

작성일: 2026-06-01

## v5 결과 요약

`pmk_cf_self_rally_v5`는 v4보다 확실히 좋아졌다.

- `completed_timesteps = 2,000,000`
- `mean_useful_bounces = 1.45`
- `max_useful_bounces = 7`
- `two_or_more_useful_bounce_rate = 0.42`
- `three_or_more_useful_bounce_rate = 0.22`
- `time_limit = 49 / 100`
- `ball_out_of_bounds = 43 / 100`
- `robot_body_contact = 0 / 100`

viewer에서도 600 step까지 가는 episode가 생겼다. 다만 contact 수에 비해 useful bounce가 적고, 공이 낮게 반복 접촉되는 문제가 남았다.

## contact 분석 결론

100 episode rebound 분석:

- total contacts: `2615`
- useful contacts: `111`
- useful contact rate: `0.042`
- mean projected apex height: `0.187m`
- useful projected apex height: `0.339m`
- mean next intercept xy error: `0.062m`
- useful next intercept xy error: `0.019m`
- next intercept reachable rate: `0.474`
- useful next intercept reachable rate: `1.000`
- mean actual outgoing x: `+0.031`
- useful actual outgoing x: `-0.022`
- mean actual outgoing z: `1.512`
- useful actual outgoing z: `2.360`

즉 v5의 핵심 병목은 "공을 로봇팔 밖으로 계속 밀어내는 tilt 실패"보다는 "많은 contact가 목표 apex까지 못 올라가는 약한 타격"이다.

조건별 실패를 순서대로 나누면 `apex >= 0.30m`에서 가장 많이 탈락했다.

- upward velocity 부족: `119`
- racket z velocity 부족: `8`
- contact xy error 초과: `175`
- racket lateral speed 초과: `118`
- apex가 target보다 낮음: `1861`
- apex window 초과: `115`
- next intercept unreachable: `97`
- all pass: 약 `118`

따라서 useful_bounces가 낮게 보이는 것은 카운터 버그라기보다 strict success gate가 낮은 타격을 의도대로 걸러내는 결과다.

## tilt 판단

tilt는 v5에서 사용 중이다.

- `target_tilt_0` 평균: `-0.051`
- `target_tilt_0` 범위: `-0.146 ~ +0.134`
- `target_tilt_1` 범위: `-0.070 ~ +0.119`

v4와 비교하면 lateral drift도 크게 줄었다.

- v4 actual outgoing x 평균: `+0.330`
- v5 actual outgoing x 평균: `+0.031`

그래서 지금 tilt를 더 크게 흔드는 방향으로 바로 가면, 낮은 타격/속도 제한/불안정 접촉이 다시 악화될 가능성이 크다. 다음 학습의 1차 목표는 vertical impulse와 target apex matching을 더 명확히 만드는 것이다.

## 구현 변경

### 1. 낮은 upward contact에 직접 penalty 추가

`contact_apex_under_target_penalty_weight`를 추가했다.

동작:

- contact가 있고
- outgoing z가 success threshold보다 위이며
- projected apex가 target apex보다 낮으면
- `height_tolerance` 기준으로 부족한 만큼 penalty를 준다.

이 보상은 useful 판정을 느슨하게 하지 않는다. 낮게 톡톡 맞는 contact를 성공처럼 보이게 만들지 않고, 정책이 목표 높이로 올려치는 쪽을 더 선호하게 만든다.

self-rally preset:

- `contact_apex_under_target_penalty_weight = 0.60`

### 2. 기본 vertical primitive 소폭 강화

v5에서 speed-limit failure는 낮고, 낮은 apex 실패가 압도적이었기 때문에 기본 타격 높이를 약간 보강했다.

self-rally preset:

- `vertical_action_limit = 0.030`
- `contact_frame_apex_lift_max = 0.055`
- `contact_frame_velocity_target_gain = 1.00`
- `contact_frame_velocity_target_max = 2.0`

이 변경은 무작정 reward 숫자를 키우는 것이 아니라, planner/primitive가 계산한 desired outgoing velocity를 실제 policy가 더 잘 낼 수 있게 하는 쪽이다.

## 다음 학습

다음 run은 v6로 분리한다.

```bash
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v6 \
  --reset-model \
  --total-timesteps 2000000
```

평가는 기존처럼 한다.

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v6/pmk_cf_self_rally_v6_model.zip \
  --episodes 100
```

정량 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v6/pmk_cf_self_rally_v6_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v6_contact_diagnosis
```

## v6에서 볼 지표

- mean projected apex height가 `0.187m`에서 `0.25m+`로 올라가는지
- useful projected apex height가 `0.30 ~ 0.40m` window에 유지되는지
- useful contact rate가 `0.042`보다 올라가는지
- actual outgoing x 평균이 다시 크게 `+`로 밀리지 않는지
- next intercept xy error가 `0.06m` 이하로 유지되는지
- ball_speed_limit failure가 크게 늘지 않는지

## 검증

로컬 검증:

- `python -m py_compile src/pingpong_rl2/envs/keepup_env.py scripts/run_ppo_learning.py`
- `python -m unittest tests/test_keepup_env.py`
- `python -m unittest tests/test_ppo_runs.py tests/test_vector_env.py tests/test_keepup_contract_features.py`

모두 통과했다.

`total_timesteps=0` preset smoke:

- `mean_useful_bounces = 0.960`
- `max_useful_bounces = 4`
- `two_or_more_rate = 0.280`
- `three_or_more_rate = 0.130`

이 smoke 수치는 최종 성능이 아니라 preset/env 연결 확인이다.
