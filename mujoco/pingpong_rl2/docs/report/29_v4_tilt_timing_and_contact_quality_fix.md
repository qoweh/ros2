# 29. v4 tilt timing / contact-quality reward 보강

작성일: 2026-06-01

## v4 결과 요약

`pmk_cf_self_rally_v4`는 2M timesteps 학습 후에도 useful bounce가 0이었다.

학습 summary:

- `completed_timesteps = 2,000,000`
- `mean_useful_bounces = 0.0`
- `max_useful_bounces = 0`
- failure:
  - `ball_out_of_bounds = 0.77`
  - `ball_speed_limit = 0.16`
  - `floor_contact = 0.07`
  - `robot_body_contact = 0.0`

라켓 길이/손잡이 변경은 팔 충돌에는 도움이 됐다. 하지만 self-rally 목표에는 아직 실패했다.

## rebound 분석 결론

100 episode contact 분석 결과:

- total contacts: `259`
- useful contact rate: `0.0`
- desired outgoing x 평균: `-0.121`
- actual outgoing x 평균: `+0.330`
- desired outgoing z 평균: `2.221`
- actual outgoing z 평균: `1.710`
- projected apex height 평균: `0.241m`
- target apex height: `0.300m`
- next intercept xy error 평균: `0.179m`
- next intercept reachable rate: `0.089`

즉 공을 맞히긴 하지만,

- 바깥쪽 `+x`로 계속 밀고
- 높이는 부족하고
- 다음 intercept가 라켓 근처로 돌아오지 않는다.

## tilt 해석

v4에서 tilt가 아예 안 쓰인 것은 아니다.

- `target_tilt_0` 평균: `-0.047`
- `target_tilt_0` 최소: `-0.092`
- `target_tilt_1` 평균: `0.0069`

문제는 target tilt가 contact 직전에는 커지지만 실제 라켓 면이 그만큼 빨리 따라가지 못한다는 점이다.

별도 자세 추적 확인:

- `target_tilt=(-0.1, 0)`은 충분히 오래 두면 face normal x를 `+` 방향으로 만든다.
- 그런데 v4 contact 시점 actual face normal x는 대체로 아직 `-` 방향 근처에 남아 있었다.

따라서 "tilt 미사용"이 아니라 "tilt ramp가 늦고 orientation controller가 느림"이 핵심이다.

## 구현 변경

### 1. tilt를 더 일찍 켜기

`contact_frame_tilt_ramp_time`을 추가했다.

- 기본값: `0.16`
- self-rally preset: `0.35`

기존에는 intercept time `0.16s` 안쪽에서만 tilt ramp가 커졌다. 이제 self-rally에서는 더 일찍 라켓 면을 준비한다.

### 2. 라켓 자세 추종 강화

self-rally preset:

- `controller_orientation_gain = 1.10`
- `controller_max_orientation_step = 0.24`

목표는 target tilt가 생겨도 실제 라켓 면이 늦게 따라오는 문제를 줄이는 것이다.

### 3. z 방향 타격 강화

self-rally preset:

- `controller_max_velocity_step = 0.04`
- `contact_frame_apex_lift_max = 0.040`
- `contact_frame_velocity_target_gain = 0.90`
- `contact_frame_velocity_target_max = 1.8`

v4는 평균 apex가 목표 `0.30m`보다 낮았기 때문에, 기본 primitive가 더 명확히 위로 치도록 했다.

### 4. 상태 기반 tilt 범위 확대

self-rally preset:

- `target_tilt_limit = (0.16, 0.16)`
- `tilt_action_limit = 0.006`
- `contact_frame_trajectory_tilt_limit = (0.08, 0.08)`
- `contact_frame_centering_tilt_limit = (0.06, 0.06)`
- `contact_frame_centering_tilt_radius = 0.10`

RL residual은 작게 유지하고, planner/primitive의 상태 기반 tilt가 더 큰 역할을 하게 했다.

### 5. strict success 전에 학습 신호 주기

`reward_contact_quality_on_any_upward_contact`를 추가했다.

self-rally preset:

- `reward_contact_quality_on_any_upward_contact = True`
- `nonuseful_contact_penalty_weight = 0.75`
- `trajectory_match_reward_weight = 0.50`

v4는 strict success를 하나도 못 얻었기 때문에, PPO가 좋은 방향으로 이동할 양의 신호가 부족했다. 이제 strict success가 아니더라도 upward contact가 apex/easy-next-ball 면에서 좋아지면 shaping reward를 받는다.

## 짧은 검증

학습 없이 preset/env만 확인:

- `total_timesteps = 0`
- `mean_useful_bounces = 1.000`
- `max_useful_bounces = 6`
- `two_or_more_rate = 0.270`
- `three_or_more_rate = 0.100`

짧은 2048-step PPO 확인:

- `mean_useful_bounces = 1.030`
- `max_useful_bounces = 4`
- `two_or_more_rate = 0.290`
- `three_or_more_rate = 0.100`

이 수치는 최종 성능이 아니라 smoke 검증이다. 다만 v4의 학습 후 `0.0`보다 초기 조건이 훨씬 낫기 때문에, 수정 방향은 v4보다 타당하다.

## 다음 학습 권장

새 run으로 분리해서 돌린다.

```bash
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v5 \
  --reset-model \
  --total-timesteps 2000000
```

학습 후 확인:

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v5/pmk_cf_self_rally_v5_model.zip \
  --episodes 100
```

정량 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v5/pmk_cf_self_rally_v5_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v5_contact_diagnosis
```

확인할 지표:

- useful contact rate
- mean projected apex height가 `0.30m` 근처로 올라가는지
- actual outgoing x가 `+`로 계속 밀리지 않고 desired x에 가까워지는지
- next intercept xy error가 `0.179m`에서 줄어드는지
- target tilt와 actual face normal이 contact 전에 같은 방향으로 충분히 따라가는지
