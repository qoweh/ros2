# 32. v7 low-apex 종료 병목과 recovery reward 보강

작성일: 2026-06-02

## v7 평가 요약

사용자가 `pmk_cf_self_rally_v7`을 2M timesteps 학습한 뒤 남긴 50 episode 분석을 확인했다.

핵심 결과:

- episodes: `50`
- contacts: `241`
- failure:
  - `low_apex_contact = 49`
  - `ball_out_of_bounds = 1`
- mean useful bounces: `0.320`
- max useful bounces: `2`
- useful contact rate: `0.066`
- next intercept reachable rate: `0.739`
- useful next intercept reachable rate: `1.000`

v6 대비 lateral/return 품질은 좋아졌다. v6의 next intercept reachable rate는 `0.345`였고, v7은 `0.739`다. 따라서 이번 병목은 라켓 tilt나 XY return이 아니라 vertical outgoing quality다.

## contact 높이 진단

`projected_contact_apex_height_above_racket`:

- all contacts mean: `0.217m`
- all contacts median: `0.183m`
- useful contacts mean: `0.353m`
- non-useful contacts mean: `0.208m`

위로 튄 contact 중 threshold 미달 비율:

- apex `< 0.16m`: `0.420`
- apex `< 0.20m`: `0.541`
- apex `< 0.30m`: `0.766`

episode 마지막 contact:

- mean terminal projected apex: `0.123m`
- mean terminal actual outgoing z: `1.179m/s`
- mean terminal desired outgoing z: `2.229m/s`
- mean terminal z error: `1.050m/s`

해석:

- v7은 낮은 접촉 반복을 오래 허용하지 않는 데는 성공했다.
- 그러나 low-apex termination이 너무 빨라서, 낮은 접촉에서 `조금 더 높게 치는` 중간 학습 신호가 약해졌다.
- 현재 낮은 접촉에 대한 `apex_match_term`은 target 0.30, tolerance 0.10 기준으로 apex 0.20 미만에서 0이 된다.
- 따라서 0.08m, 0.12m, 0.18m 접촉을 구분하는 직접적인 positive progress reward가 필요하다.

## 구현 변경

### 1. apex progress reward 추가

새 env 옵션:

- `contact_apex_progress_reward_weight`

동작:

- upward contact의 projected apex를 target height로 나눈 값을 `[0, 1]`로 clip한다.
- 낮은 contact라도 더 높은 apex를 만들면 더 큰 양의 보상을 받는다.
- target 이상은 cap되어, 너무 높게 쏘는 정책을 추가로 장려하지 않는다.

reward term:

- `contact_apex_progress_term`

기본값은 `0.0`이라 기존 실험 계약은 유지된다.

### 2. self-rally preset v8 방향 조정

`contact_frame_self_rally_candidate`를 다음 방향으로 조정했다.

- low-apex termination:
  - threshold `0.20 -> 0.16`
  - grace `1 -> 2`
- vertical primitive:
  - `contact_frame_apex_lift_gain 0.05 -> 0.08`
  - `contact_frame_apex_lift_max 0.055 -> 0.075`
  - `contact_frame_velocity_target_max 2.0 -> 2.4`
  - `controller_max_velocity_step 0.04 -> 0.055`
  - `contact_frame_followthrough_max 0.055` 명시
- reward:
  - `contact_apex_under_target_penalty_weight 0.60 -> 0.55`
  - `contact_apex_progress_reward_weight = 0.80`

의도:

- 0.16m 미만의 명백히 낮은 반복 contact는 여전히 끊는다.
- 0.16~0.30m의 회복 가능한 contact는 더 많이 관찰하게 두고, 더 높게 칠수록 보상이 증가하게 한다.
- vertical primitive의 상한을 조금 열어 actual outgoing z가 desired z에 더 가까워지게 한다.

## 다음 학습 명령

v7과 섞지 말고 새 run version으로 학습한다.

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v8 \
  --reset-model \
  --total-timesteps 2000000
```

분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v8/pmk_cf_self_rally_v8_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v8_contact_diagnosis
```

## v8 판단 기준

우선 볼 지표:

- `mean_projected_contact_apex_height_above_racket >= 0.25m`
- `upward_contact_projected_apex_below_0_20_rate`가 v7의 `0.541`보다 감소
- `terminal_contact_summary.mean_terminal_projected_contact_apex_height_above_racket`가 v7의 `0.123m`보다 상승
- useful contact rate가 v7의 `0.066`보다 상승
- next intercept reachable rate가 `0.70+` 근처를 유지
- `ball_speed_limit` 증가가 크지 않을 것

v8도 low-apex가 계속 많으면 reward를 더 키우기보다 controller velocity tracking과 actual racket z velocity를 먼저 본다.
