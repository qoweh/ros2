# v21 Low-Apex Review And v22 Low-Stable Window

## 한줄 결론

v21은 viewer에서 보이는 것처럼 "낮지만 안정적인 루프"를 꽤 만들고 있다. 다만 기존 판정이 `target_ball_height=0.30m`보다 낮은 apex를 useful로 인정하지 않아, 실제 체감보다 `low_apex_contact`와 낮은 useful count가 과하게 보였다. v22에서는 useful height window를 실제 양방향 window로 고치고, 진짜 잔진동만 잡도록 low-apex 종료 기준을 낮췄다.

## 현재 `low_apex_contact` 기준

v21 config:

- `target_ball_height = 0.30m`
- `height_tolerance = 0.10m`
- `low_apex_contact_height_threshold = 0.20m`
- `low_apex_contact_grace_count = 2`

코드 기준:

1. 접촉이 있어야 한다.
2. 이미 useful contact로 인정된 접촉이면 low-apex가 아니다.
3. 공이 위로 나가야 한다. `actual_outgoing_velocity_z > success_velocity_threshold`
4. projected apex height above racket이 threshold보다 낮아야 한다.
5. 이런 contact가 grace count를 넘으면 `failure_reason = low_apex_contact`

즉 v21에서는 라켓 기준 projected apex가 `0.20m` 미만인 비-useful upward contact가 3번 연속 나오면 종료된다.

## 왜 viewer와 수치가 다르게 느껴졌나

기존 `_success_reason`은 `require_apex_height_window_for_success=True`에서도 아래쪽 tolerance를 사실상 쓰지 않았다.

기존 effective useful window:

- `0.30m <= projected_apex <= 0.40m`

사람이 자연스럽게 기대한 window:

- `0.20m <= projected_apex <= 0.40m`

그래서 `0.20~0.30m`의 낮지만 받을 수 있는 공은 failure는 아니더라도 useful/stable로도 인정되지 않았다.

v21 offline re-score:

| Useful height window | mean useful-like contacts | max | episodes >= 1 | episodes >= 2 | episodes >= 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| current `0.30~0.40m` | 1.45 | 6 | 67 | 44 | 21 |
| relaxed `0.24~0.40m` | 2.79 | 9 | 78 | 62 | 52 |
| relaxed `0.20~0.40m` | 3.80 | 14 | 85 | 69 | 58 |
| relaxed `0.16~0.40m` | 5.06 | 17 | 89 | 77 | 65 |

이 말은 v21이 완전히 못 배운 게 아니라, 낮은 안정 루프를 기존 metric이 과소평가했다는 뜻이다.

## v21 자체는 나아졌나

v21은 v20 대비 일부는 개선, 일부는 악화다.

좋아진 점:

- mean projected apex: v20 `0.269m` -> v21 `0.282m`
- terminal mean apex: v20 `0.164m` -> v21 `0.184m`
- `low_apex_contact`: v20 `77/100` -> v21 `68/100`
- upward below target rate: v20 `0.625` -> v21 `0.605`
- 새 action 축을 실제로 사용함
  - `target_apex_z_residual` 평균 `-0.011m`
  - 정책이 목표 apex를 약 1cm 낮추는 쪽을 선택했다.

나빠진 점:

- mean useful bounces: v20 `2.00` -> v21 `1.45`
- max useful bounces: v20 `10` -> v21 `6`
- next-intercept reachable: v20 `0.801` -> v21 `0.679`
- projected apex xy error: v20 `0.024` -> v21 `0.033`

해석: v21은 더 낮고 안정적인 높이로 가려는 방향은 보였지만, 현재 useful/next-intercept 기준과 완전히 맞지는 않았다. 발표/시연 체감과 metric 사이의 gap이 커졌다.

## v22 변경

코드 변경:

- `require_apex_height_window_for_success=True`일 때 success height window를 진짜 양방향으로 사용한다.
  - 기존: `projected_apex >= target` 그리고 `abs(projected-target) <= tolerance`
  - 변경: `target - tolerance <= projected_apex <= target + tolerance`
- stable-cycle height 판정도 같은 window를 사용한다.
- under-target penalty와 low-apex recovery memory는 `target`이 아니라 `minimum useful apex` 아래에서만 강하게 작동하게 했다.

새 preset:

`contact_frame_self_rally_v22_low_stable_window`

주요 값:

- v21의 15D action mode 유지
- useful height window: `0.20~0.40m`
- `low_apex_contact_height_threshold = 0.14m`
- `low_apex_contact_grace_count = 3`
- `contact_apex_under_target_penalty_weight = 0.65`
- `contact_lateral_stability_min_apex_ratio = 0.70`
- `stable_contact_min_apex_ratio = 0.70`

의도:

- `0.20m` 이상은 낮지만 받을 수 있는 공으로 인정
- `0.14m` 미만은 잔진동/너무 낮은 공으로 종료 후보
- 낮은 안정 루프를 metric과 reward가 더 정직하게 인정

## 변경 파일

- `src/pingpong_rl2/envs/keepup_env.py`
  - `_minimum_useful_apex_height`
  - `_maximum_useful_apex_height`
  - `_apex_height_in_success_window`
  - success/stable-cycle/under-target/recovery 기준 정렬
- `scripts/run_ppo_learning.py`
  - v22 preset 추가
- `tests/test_keepup_env.py`
  - 낮은 쪽 success window와 stable-cycle 테스트 추가

## 검증

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_env
```

결과: `Ran 122 tests ... OK`

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v22_low_stable_window \
  --run-name tmp_v22_env_check \
  --run-version codex \
  --reset-model \
  --total-timesteps 64 \
  --smoke \
  --bootstrap-heuristic-episodes 0 \
  --bootstrap-followup-epochs 0 \
  --output-dir artifacts/tmp/tmp_v22_env_check_codex
```

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v22_low_stable_window \
  --run-name tmp_v22_resume_check \
  --run-version codex \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v21/pmk_cf_self_rally_v21_model.zip \
  --total-timesteps 64 \
  --smoke \
  --bootstrap-heuristic-episodes 0 \
  --bootstrap-followup-epochs 0 \
  --output-dir artifacts/tmp/tmp_v22_resume_check_codex
```

결과: v21 모델에서 v22 기준으로 resume 가능. smoke evaluation에서 mean useful bounces `4.0`.

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/tmp/tmp_v22_env_check_codex/tmp_v22_env_check_codex_model.zip \
  --episodes 2 \
  --seed 221 \
  --output-dir artifacts/tmp/tmp_v22_env_check_codex/analysis \
  --analysis-name tmp_v22_analysis_check
```

## 학습 명령

v22는 reward/success 기준이 바뀌었지만 action dimension은 v21과 동일한 15D다. viewer상 v21이 이미 낮은 안정 루프를 어느 정도 만들고 있으므로, 우선 v21 모델에서 이어서 fine-tune하는 쪽을 추천한다.

500k fine-tune:

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v22_low_stable_window \
  --run-name pmk_cf_self_rally \
  --run-version v22 \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v21/pmk_cf_self_rally_v21_model.zip \
  --total-timesteps 500000
```

학습 후 분석:

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name pmk_cf_self_rally \
  --run-version v22 \
  --episodes 100 \
  --seed 221 \
  --analysis-name pmk_cf_self_rally_v22_final_contact_diagnosis
```

v22 분석에서 볼 것:

- `low_apex_contact`가 줄었는지
- terminal projected apex가 `0.14m` 아래에 몰리는지
- useful/stable count가 viewer 느낌과 맞아졌는지
- `time_limit` episode가 v21보다 늘었는지
- 낮은 안정 루프가 `ball_out_of_bounds`를 늘리지 않는지
