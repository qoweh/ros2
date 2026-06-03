# v22 Review And v23 Outward Timing Guard

## 한줄 결론

`pmk_cf_self_rally_v22`는 v21보다 훨씬 발표 가능한 쪽으로 올라왔다. 평균 useful bounce는 `7.24`, `time_limit`은 `61/100`이다. 다만 남은 핵심 실패는 `ball_out_of_bounds=21/100`이고, 원인은 정책이 +x로 보내려는 것보다 접촉 순간 라켓이 +x, 즉 목표점 바깥 방향으로 움직이는 타이밍 문제가 더 크다. v23은 action 차원을 늘리지 않고, 15D 정책이 이미 가진 `racket_vx/racket_vy` residual을 더 잘 쓰도록 outward racket velocity penalty와 약한 lateral brake를 추가했다.

## v22 결과

분석 파일:

- `artifacts/ppo_runs/pmk_cf_self_rally_v22/analysis/pmk_cf_self_rally_v22_final_contact_diagnosis_summary.json`
- `artifacts/ppo_runs/pmk_cf_self_rally_v22/analysis/pmk_cf_self_rally_v22_final_contact_diagnosis_contacts.csv`
- `artifacts/ppo_runs/pmk_cf_self_rally_v22/analysis/pmk_cf_self_rally_v22_final_contact_diagnosis_episodes.csv`

주요 수치:

- mean return: `57.08`
- mean useful bounces: `7.24`
- max useful bounces: `16`
- `1+ useful`: `94%`
- `2+ useful`: `85%`
- `3+ useful`: `82%`
- failure counts:
  - `time_limit`: `61`
  - `ball_out_of_bounds`: `21`
  - `low_apex_contact`: `12`
  - `floor_contact`: `4`
  - `ball_speed_limit`: `2`

해석:

- v22는 v21의 낮은 안정 루프 과소평가 문제를 잘 고쳤다.
- `low_apex_contact`는 v21의 주 실패였지만 v22에서는 주 실패가 아니다.
- 이제 가장 큰 병목은 lateral out, 특히 +x 방향으로 공이 로봇팔에서 멀어지는 문제다.

## viewer 관찰 검증

사용자가 본 "초반 episode에서 ball out이 눈에 띈다"는 관찰은 맞다.

첫 12 episode:

| episode | failure | useful |
| ---: | --- | ---: |
| 1 | `floor_contact` | 2 |
| 2 | `low_apex_contact` | 1 |
| 3 | `ball_out_of_bounds` | 1 |
| 4 | `time_limit` | 12 |
| 5 | `low_apex_contact` | 5 |
| 6 | `time_limit` | 13 |
| 7 | `low_apex_contact` | 7 |
| 8 | `time_limit` | 10 |
| 9 | `low_apex_contact` | 0 |
| 10 | `ball_out_of_bounds` | 4 |
| 11 | `time_limit` | 9 |
| 12 | `time_limit` | 11 |

초반 12개만 봐도 `ball_out_of_bounds`가 2개라 viewer 체감상 눈에 띌 수 있다.

## 15D action 사용량

v22의 trained PPO `log_std`:

```text
std = [0.0069, 0.0069, 0.0105, 0.0028, 0.0028,
       0.0796, 0.0802, 0.0797, 0.0798, 0.0800,
       0.0800, 0.0802, 0.0799, 0.0280, 0.0087]
```

v17 때처럼 작은 물리 action bound와 큰 std가 맞지 않아 앞 축이 전부 clipping되는 문제는 v22에서 대부분 해결됐다.

접촉 전체 action 사용:

| axis | mean | abs mean | max abs | sat >=95% |
| --- | ---: | ---: | ---: | ---: |
| radial | `0.0005` | `0.0016` | `0.0080` | `0.0%` |
| tangent | `0.0018` | `0.0025` | `0.0073` | `0.0%` |
| z | `0.0089` | `0.0089` | `0.0176` | `0.0%` |
| pitch | `-0.0027` | `0.0028` | `0.0045` | `0.0%` |
| roll | `-0.0049` | `0.0049` | `0.0080` | `8.6%` |
| vz_scale | `0.0035` | `0.0054` | `0.0229` | `0.0%` |
| outgoing_x_residual | `-0.1552` | `0.1552` | `0.1811` | `0.0%` |
| outgoing_y_residual | `-0.0144` | `0.0144` | `0.0269` | `0.0%` |
| racket_vz_residual | `0.0258` | `0.0258` | `0.0473` | `0.0%` |
| trajectory_tilt_scale | `-0.0038` | `0.0067` | `0.0284` | `0.0%` |
| centering_tilt_scale | `-0.0089` | `0.0120` | `0.0425` | `0.0%` |
| racket_vx_residual | `-0.0164` | `0.0165` | `0.0389` | `0.0%` |
| racket_vy_residual | `-0.0081` | `0.0090` | `0.0210` | `0.0%` |
| target_apex_z_residual | `-0.0072` | `0.0090` | `0.0337` | `0.0%` |
| strike_plane_z_residual | `0.0038` | `0.0039` | `0.0114` | `0.0%` |

해석:

- 축들이 죽어 있지는 않다.
- `outgoing_x_residual`은 강하게 -x 보정을 배웠다.
- 하지만 `racket_vx_residual`은 bound 대비 약하게 쓰인다.
- 따라서 문제는 "dimension이 없어서"보다는, 라켓 횡속도 타이밍을 정책이 더 강하게 학습할 reward 신호가 부족한 쪽에 가깝다.

## ball_out_of_bounds 원인

terminal contact 평균:

| failure | desired vx | actual vx | racket vx | next intercept xy error | projected apex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ball_out_of_bounds` | `-0.351` | `+0.596` | `+0.380` | `0.285` | `0.179` |
| `time_limit` | `-0.067` | `+0.040` | `+0.005` | `0.051` | `0.303` |
| `low_apex_contact` | `-0.152` | `+0.148` | `+0.131` | `0.095` | `0.098` |

중요한 점:

- `ball_out_of_bounds`에서도 정책의 desired outgoing x는 평균 `-0.351m/s`다.
- 하지만 실제 outgoing x는 평균 `+0.596m/s`로 반대로 나간다.
- 같은 순간 라켓 x velocity가 평균 `+0.380m/s`다.

즉 정책이 의도적으로 +x로 보내는 것이 주 원인이라기보다, 접촉 타이밍에서 라켓이 바깥쪽으로 움직이며 공을 밀어내는 현상이 크다.

상관관계도 이 해석과 맞다.

- `actual_outgoing_velocity_x` vs `next_intercept_xy_error`: `+0.619`
- `actual_outgoing_velocity_x` vs `racket_velocity_x`: `+0.452`
- `actual_outgoing_velocity_x` vs `racket_face_normal_x`: `-0.341`
- `actual_outgoing_velocity_x` vs `applied_action_6_outgoing_x_residual`: `+0.306`
- `actual_outgoing_velocity_x` vs `applied_action_11_racket_vx_residual`: `+0.212`

## 낮은 높이 이슈

v22에서 `low_apex_contact=12/100`이므로, v21처럼 metric이 과도하게 낮은 안정 루프를 죽이는 상태는 아니다.

다만 `ball_out_of_bounds` terminal의 projected apex가 평균 `0.179m`이고, actual outgoing z도 desired z보다 낮다.

- desired z: `2.190m/s`
- actual z: `1.224m/s`

이것도 단순히 힘 부족이라기보다, 라켓이 접촉 순간 lateral로 스윕하면서 정상적인 vertical impulse를 못 주는 timing/impact quality 문제와 같이 묶여 있다.

## v23 변경

새 preset:

`contact_frame_self_rally_v23_outward_timing_guard`

v22에서 이어받고 다음을 추가했다.

- `contact_racket_outward_velocity_penalty_weight = 0.85`
- `contact_racket_outward_velocity_tolerance = 0.04`
- `contact_frame_lateral_brake_gain = 0.65`
- `contact_frame_lateral_brake_max = 0.25`
- `contact_frame_lateral_brake_radius = 0.12`
- `contact_racket_lateral_velocity_penalty_weight = 0.55`

새 reward term:

- `contact_racket_outward_velocity_penalty`

정의:

1. contact ball xy가 keepup target xy에서 바깥쪽으로 얼마나 벗어났는지 방향을 만든다.
2. contact racket velocity xy를 그 outward direction에 projection한다.
3. positive outward speed만 penalty로 쓴다.
4. inward sweep은 penalty가 0이다.

의도:

- 전체 lateral velocity를 무작정 줄이지 않는다.
- 로봇팔 쪽으로 되돌리는 inward correction은 살린다.
- `ball_out_of_bounds`를 만드는 outward sweep timing만 명확히 벌한다.
- 기존 15D의 `racket_vx/racket_vy` residual이 더 실제로 쓰이게 만든다.

## cleanup

삭제:

- `src/pingpong_rl2/controllers/joint_controller.py`
- `JointPositionController` export
- `PingPongSim.home_gripper_target`
- `PingPongSim.racket_grip_position`
- `PingPongSim.reset_if_failed`

유지:

- `SB3AsyncVectorEnvAdapter.env_method`
- `SB3AsyncVectorEnvAdapter.env_is_wrapped`
- `SB3AsyncVectorEnvAdapter.get_images`

이 세 메서드는 텍스트 참조가 없어도 SB3 `VecEnv` interface contract라 삭제하면 안 된다.

## 검증 예정 지표

v23 분석에서 우선 볼 것:

- `ball_out_of_bounds`가 v22의 `21/100`보다 줄었는지
- mean useful bounces가 `7` 근처 이상을 유지하는지
- `time_limit`이 `61/100` 근처 또는 그 이상인지
- `low_apex_contact`가 다시 크게 늘지 않는지
- `mean_contact_racket_outward_speed`가 v22 대비 내려가는지
- `mean_contact_frame_lateral_brake_speed`가 너무 커지지 않는지

## 구현 검증

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  src/pingpong_rl2/envs/pingpong_sim.py \
  src/pingpong_rl2/controllers/__init__.py \
  scripts/run_ppo_learning.py \
  scripts/run_ppo_rebound_analysis.py \
  tests/test_keepup_env.py
```

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest \
  tests.test_keepup_env tests.test_keepup_contract_features tests.test_vector_env
```

결과: `Ran 134 tests ... OK`

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v23_outward_timing_guard \
  --run-name tmp_v23_outward_guard_check \
  --run-version codex \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v22/pmk_cf_self_rally_v22_model.zip \
  --total-timesteps 64 \
  --smoke \
  --bootstrap-heuristic-episodes 0 \
  --bootstrap-followup-epochs 0 \
  --output-dir artifacts/tmp/tmp_v23_outward_guard_check_codex
```

결과: v22 checkpoint에서 v23 env로 resume 가능. smoke eval은 mean useful bounces `8.5`, max useful bounces `10`.

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v23_outward_timing_guard \
  --run-name tmp_v23_fresh_check \
  --run-version codex \
  --reset-model \
  --total-timesteps 64 \
  --smoke \
  --bootstrap-heuristic-episodes 0 \
  --bootstrap-followup-epochs 0 \
  --output-dir artifacts/tmp/tmp_v23_fresh_check_codex
```

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/tmp/tmp_v23_outward_guard_check_codex/tmp_v23_outward_guard_check_codex_model.zip \
  --episodes 3 \
  --seed 232 \
  --output-dir artifacts/tmp/tmp_v23_outward_guard_check_codex/analysis \
  --analysis-name tmp_v23_outward_guard_analysis_check
```

결과: 3 episode 모두 `time_limit`, mean useful bounces `10.0`. `mean_contact_racket_outward_speed`, `max_contact_racket_outward_speed`, `mean_contact_racket_outward_velocity_penalty`가 summary에 기록됨.

## 학습 명령

v23은 v22와 같은 15D action mode라서 v22 model에서 resume한다.

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v23_outward_timing_guard \
  --run-name pmk_cf_self_rally \
  --run-version v23 \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v22/pmk_cf_self_rally_v22_model.zip \
  --total-timesteps 500000
```

학습 후 분석:

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name pmk_cf_self_rally \
  --run-version v23 \
  --episodes 100 \
  --seed 231 \
  --analysis-name pmk_cf_self_rally_v23_final_contact_diagnosis
```

간단 확인:

```bash
jq '{mean_useful_bounces,max_useful_bounces,failure_counts,contact_summary:{mean_contact_racket_outward_speed,max_contact_racket_outward_speed,mean_contact_frame_lateral_brake_speed,mean_contact_racket_outward_velocity_penalty}}' \
  artifacts/ppo_runs/pmk_cf_self_rally_v23/analysis/pmk_cf_self_rally_v23_final_contact_diagnosis_summary.json
```
