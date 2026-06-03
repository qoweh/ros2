# v20 Review And v21 Apex/Timing Residual

## 한줄 결론

v20은 일부 긴 episode와 `max_useful_bounces=10`을 만들었지만, v19 대비 전체 안정성이 좋아졌다고 보기는 어렵다. 특히 terminal low-apex가 더 악화되어서, 다음 실험은 v20 brake를 더 키우기보다 v19 기반에 `desired apex height`와 `strike plane/contact timing`을 정책이 직접 조절하는 15D action mode를 추가하는 쪽이 타당하다.

## v20 결과 검토

분석 대상:

- Run: `artifacts/ppo_runs/pmk_cf_self_rally_v20`
- Model: `pmk_cf_self_rally_v20_model.zip`
- Analysis: `analysis/pmk_cf_self_rally_v20_final_contact_diagnosis_summary.json`
- Preset: `contact_frame_self_rally_v20_boundary_brake`
- Action mode: 기존 13D `position_contact_frame_velocity_tilt_lateral_residual`

v19/v20 비교:

| Metric | v19 | v20 |
| --- | ---: | ---: |
| mean useful bounces | 2.21 | 2.00 |
| max useful bounces | 9 | 10 |
| 2+ useful rate | 0.53 | 0.49 |
| 3+ useful rate | 0.32 | 0.39 |
| `time_limit` | 22/100 | 14/100 |
| `low_apex_contact` | 69/100 | 77/100 |
| `ball_out_of_bounds` | 8/100 | 8/100 |
| upward contact below 0.20m | 0.380 | 0.408 |
| next-intercept reachable rate | 0.863 | 0.801 |
| mean projected apex xy error | 0.0197 | 0.0243 |
| mean outgoing z error | 0.695 | 0.717 |
| terminal mean apex above racket | 0.204m | 0.164m |
| terminal below 0.20m rate | 0.753 | 0.909 |
| terminal outgoing z error | 0.841m/s | 0.971m/s |

v20의 checkpoint eval에서는 600k checkpoint가 가장 좋았다.

- mean useful bounces: `2.025`
- max useful bounces: `8`
- 2+ useful rate: `0.50`
- 3+ useful rate: `0.325`
- `ball_out_of_bounds`: `4/40`

즉 v20 학습 후반이 완전히 망가진 것은 아니지만, final 100-episode 분석 기준으로는 v19보다 명확히 우세하지 않다.

## v20 실패 모드

`time_limit`은 좋은 bucket이다.

- `time_limit`: 14 episodes, mean useful `4.86`, mean steps `600`
- terminal apex mean `0.305m`

반대로 `low_apex_contact`와 `ball_out_of_bounds`는 마지막 접촉에서 z 속도 실행이 부족하다.

- `low_apex_contact` terminal apex mean: `0.139m`
- `low_apex_contact` terminal desired z: `2.126m/s`
- `low_apex_contact` terminal actual z: `1.129m/s`
- `ball_out_of_bounds` terminal apex mean: `0.153m`
- `ball_out_of_bounds` terminal desired z: `2.221m/s`
- `ball_out_of_bounds` terminal actual z: `1.212m/s`
- `ball_out_of_bounds` terminal next xy error mean: `0.232m`

v20의 lateral brake는 `ball_out_of_bounds` terminal에서 자주 최대치 `0.45m/s`까지 켜졌지만, 그래도 실제 outgoing x가 원하는 방향으로 충분히 바뀌지 않았다. 그래서 이 문제는 단순히 brake gain을 더 올리는 문제라기보다, 접촉 순간의 높이/타이밍/라켓 속도 실행 문제와 섞여 있다.

## Action Saturation 재확인

v20 접촉 시 action saturation은 거의 없었다.

- 13개 축 모두 `0.95 * action_limit` 이상 포화 비율 `0.0`
- trained log_std도 v17처럼 큰 mismatch 상태가 아니다.
- 앞 5축 std는 action limit에 맞게 낮아졌고, 큰 velocity residual 축은 약 `0.08` 수준이다.

따라서 현재 병목은 "action bound가 작아서 못 한다"보다는 "정책이 원하는 apex 높이와 접촉 높이를 직접 선택하지 못하고, z/racket velocity residual로 간접 제어한다"에 가깝다.

## v21 구현 방향

report 42의 action dimension 확장 후보 중 우선순위가 높은 두 축을 추가했다.

새 action mode:

`position_contact_frame_velocity_tilt_lateral_apex_residual`

15D action:

1. radial contact residual
2. tangent contact residual
3. z contact residual
4. pitch residual
5. roll residual
6. desired outgoing z scale residual
7. desired outgoing x residual
8. desired outgoing y residual
9. racket vz residual
10. trajectory tilt scale residual
11. centering tilt scale residual
12. racket vx residual
13. racket vy residual
14. target apex z residual
15. strike plane z residual

새 축의 limit:

- `contact_frame_target_apex_z_action_limit = 0.08m`
- `contact_frame_strike_plane_z_action_limit = 0.025m`

의도:

- `target apex z residual`: 낮게 통통거리는 루프를 직접 겨냥한다. 정책이 "이번 공은 더 높게 보내야 한다"를 z velocity scale보다 명확하게 표현할 수 있다.
- `strike plane z residual`: 너무 낮거나 늦은 접촉을 직접 조절한다. planner/intercept/desired target z가 같은 residual을 보게 해서 분석과 controller 목표가 엇갈리지 않게 했다.

v21 preset:

`contact_frame_self_rally_v21_apex_timing_residual`

주의: v21은 v20을 그대로 상속하지 않고 v19 기반으로 만들었다. v20 brake가 low-apex를 악화시킨 흔적이 있어서, 이번 실험은 원인 분리를 위해 `v19 + apex/timing residual` 중심으로 둔다.

## 변경 파일

- `src/pingpong_rl2/envs/keepup_env.py`
  - 15D action mode 추가
  - apex z residual / strike plane z residual 상태, bound, validation, info 기록 추가
  - planner desired velocity와 contact trajectory metrics가 resolved apex target을 사용하도록 정렬
- `src/pingpong_rl2/controllers/heuristic_keepup.py`
  - heuristic bootstrap이 새 action mode에서 뒤 10개 residual을 0으로 채우도록 추가
- `scripts/run_ppo_learning.py`
  - v21 preset 추가
  - 새 action mode, action limit CLI 인자, bootstrap allowlist 추가
- `scripts/run_ppo_rebound_analysis.py`
  - 새 action limit override와 CSV 컬럼 추가
- `src/pingpong_rl2/defaults.py`
  - 기본 run name `pmk_cfvtlar` 추가
- `src/pingpong_rl2/utils/ppo_runs.py`
  - 새 action mode 기본 run-name 매핑 추가
- `tests/test_keepup_env.py`
  - 15D action space, residual 효과, heuristic 호환성 테스트 추가

## 검증

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_env
```

결과: `Ran 120 tests ... OK`

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_ppo_runs
```

결과: `Ran 4 tests ... OK`

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v21_apex_timing_residual \
  --run-name tmp_v21_env_check \
  --run-version codex \
  --reset-model \
  --total-timesteps 64 \
  --smoke \
  --bootstrap-heuristic-episodes 0 \
  --bootstrap-followup-epochs 0 \
  --output-dir artifacts/tmp/tmp_v21_env_check_codex
```

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/tmp/tmp_v21_env_check_codex/tmp_v21_env_check_codex_model.zip \
  --episodes 2 \
  --seed 221 \
  --output-dir artifacts/tmp/tmp_v21_env_check_codex/analysis \
  --analysis-name tmp_v21_analysis_check
```

참고: `mujoco_env`에는 `pytest`가 없어서 `python -m unittest`로 검증했다.

## 학습 명령

v21은 action dimension이 13D에서 15D로 바뀌었기 때문에 v20 zip을 그대로 resume할 수 없다. 새 정책으로 시작해야 한다.

1M 먼저 학습:

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v21_apex_timing_residual \
  --run-name pmk_cf_self_rally \
  --run-version v21 \
  --reset-model \
  --total-timesteps 1000000
```

학습 후 분석:

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name pmk_cf_self_rally \
  --run-version v21 \
  --episodes 100 \
  --seed 221 \
  --analysis-name pmk_cf_self_rally_v21_final_contact_diagnosis
```

분석에서 특히 볼 것:

- `applied_action_13_target_apex_z_residual`
- `applied_action_14_strike_plane_z_residual`
- `contact_frame_planner_resolved_target_apex_z`
- terminal `actual_outgoing_velocity_z` vs `desired_outgoing_velocity_z`
- upward contact below `0.20m`
- `ball_out_of_bounds` terminal의 `next_intercept_xy_error`
