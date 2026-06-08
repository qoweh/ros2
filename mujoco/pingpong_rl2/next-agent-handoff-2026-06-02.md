# Next Agent Handoff - 2026-06-02

이 파일은 새 세션/다른 에이전트가 이전 대화와 작업을 반복하지 않도록 남기는 인수인계 문서다.

## 프로젝트 목표

`pingpong_rl2`의 최종 목표는 MuJoCo에서 Franka/Panda 로봇팔이 탁구채를 들고 탁구공을 계속 위로 튕기는 강화학습 self-rally를 완성하는 것이다.

핵심 성공 조건은 단순히 공을 많이 맞히는 것이 아니다.

- 공을 적절한 높이로 띄워야 한다.
- 다음 공이 로봇팔/라켓이 다시 치기 쉬운 XY 위치로 돌아와야 한다.
- 너무 낮게 톡톡 치며 오래 버티는 정책은 실패로 봐야 한다.
- 라켓 tilt는 lateral recovery에 필요하지만, 최근 병목은 tilt 미작동보다 낮은 apex 반복 접촉이었다.

## 환경

작업 디렉토리:

```bash
/Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
```

사용 환경:

```bash
conda activate mujoco_env
```

학습은 보통:

```bash
python scripts/run_ppo_learning.py ...
```

viewer는 보통:

```bash
mjpython scripts/run_viewer.py ...
```

사용자가 말하기를 MuJoCo 관련 패키지는 `mujoco_env`에 설치되어 있다.

## 가장 최근 실험 상태

### v5

`pmk_cf_self_rally_v5`는 v4보다 좋아졌고, 600 step까지 가는 episode도 생겼다.

분석 결과:

- total contacts: `2615`
- useful contact rate: `0.042`
- mean projected apex height: `0.187m`
- useful projected apex height: `0.339m`
- next intercept reachable rate: `0.474`
- useful next intercept reachable rate: `1.000`
- actual outgoing x mean: `+0.031`
- useful actual outgoing x mean: `-0.022`

해석:

- tilt는 작동하고 있었다.
- v4의 actual outgoing x mean `+0.330`에서 v5 `+0.031`로 lateral drift는 크게 개선됐다.
- 그러나 대부분 contact가 너무 낮게 튀어서 useful 판정을 못 받았다.

관련 문서:

- `docs/report/30_v5_low_apex_and_height_reward_fix.md`

### v6

사용자가 약 4M timesteps까지 학습시킨 결과:

- viewer에서는 contact가 많지만 useful_bounces는 낮음.
- 예시 episode 대부분 `useful_bounces=0~1`.
- 학습 로그에 `training_mode=resume`이 찍혔다.
- 즉 v6는 기존 v6 checkpoint에서 이어 학습된 상태라, 나쁜 저높이 정책이 이미 굳은 채 더 학습됐을 가능성이 있다.

100 episode rebound 분석:

- total contacts: `1075`
- useful contacts: `46`
- useful contact rate: `0.043`
- mean projected apex height: `0.199m`
- median projected apex height: `0.165m`
- useful projected apex height: `0.347m`
- mean actual outgoing z: `1.56`
- useful actual outgoing z: `2.40`
- desired outgoing z: `2.24`
- next intercept reachable rate: `0.345`
- useful next intercept reachable rate: `1.0`

첫 실패 조건 기준:

- `apex >= 0.30m` 미달: `590 / 1075`

해석:

- reward weight만 조금 바꾸는 문제로 보지 않는다.
- 정책이 낮게라도 계속 맞히는 rollout을 오래 유지할 수 있었고, PPO가 그 상태를 많이 보면서 저높이 반복 접촉 쪽으로 굳었다.
- 따라서 v7에서는 reward 숫자 조정보다 task 구조를 바꿨다.

관련 문서:

- `docs/report/31_v6_low_bounce_loop_and_strict_cycle_fix.md`

## 최근 코드 변경 요약

### `src/pingpong_rl2/envs/keepup_env.py`

추가된 주요 옵션:

- `contact_apex_under_target_penalty_weight`
- `terminate_on_low_apex_contact`
- `low_apex_contact_height_threshold`
- `low_apex_contact_grace_count`

동작:

- upward contact가 발생했지만 projected apex가 threshold보다 낮으면 low-apex contact로 본다.
- self-rally preset에서는 1회 grace 후, 낮은 apex contact가 반복되면 episode를 `failure_reason=low_apex_contact`로 종료한다.
- 전체 `terminate_on_nonuseful_contact=True`는 쓰지 않았다. lateral/next-intercept만 살짝 부족한 exploratory hit까지 모두 죽이면 학습이 너무 sparse해질 수 있기 때문이다.

### `scripts/run_ppo_learning.py`

`contact_frame_self_rally_candidate` preset의 중요한 현재 값:

```python
"scene_path": "assets/scene.xml",
"ball_height": 0.34,
"target_ball_height": 0.30,
"reset_ball_height_range": 0.02,
"reset_xy_range": 0.028,
"reset_velocity_xy_range": 0.0,
"reset_velocity_z_range": (-0.01, 0.01),
"terminate_on_low_apex_contact": True,
"low_apex_contact_height_threshold": 0.20,
"low_apex_contact_grace_count": 1,
"terminate_on_nonuseful_contact": False,
"vertical_action_limit": 0.030,
"contact_frame_apex_lift_max": 0.055,
"contact_frame_velocity_target_gain": 1.00,
"contact_frame_velocity_target_max": 2.0,
"target_tilt_limit": (0.16, 0.16),
"tilt_action_limit": 0.006,
"checkpoint_interval": 0,
```

CLI로 추가된 옵션:

- `--reset-ball-height-range`
- `--contact-apex-under-target-penalty-weight`
- `--terminate-on-low-apex-contact`
- `--low-apex-contact-height-threshold`
- `--low-apex-contact-grace-count`

### `tests/test_keepup_env.py`

추가/수정된 테스트:

- 낮은 apex contact penalty 확인
- 낮은 apex contact 판정이 목표보다 낮은 upward contact만 잡는지 확인

### 문서

최근 보고서:

- `docs/report/29_v4_tilt_timing_and_contact_quality_fix.md`
- `docs/report/30_v5_low_apex_and_height_reward_fix.md`
- `docs/report/31_v6_low_bounce_loop_and_strict_cycle_fix.md`

index:

- `docs/report/00_index.md`

## 검증된 명령

최근 검증:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  scripts/run_ppo_learning.py
```

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest \
  tests/test_keepup_env.py \
  tests/test_keepup_contract_features.py \
  tests/test_ppo_runs.py \
  tests/test_vector_env.py
```

결과:

- 100 tests 통과

Smoke:

- `total_timesteps=0`: `mean_useful_bounces=0.100`, `max_useful_bounces=2`
- `total_timesteps=2048`: `mean_useful_bounces=0.150`, `max_useful_bounces=2`

Smoke 수치는 성능 판단이 아니다. 새 termination/reset 구조에서 학습 루프가 깨지지 않는지만 확인한 것이다.

## 다음에 해야 할 일

v6와 섞지 말고 v7로 새로 학습한다.

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v7 \
  --reset-model \
  --total-timesteps 2000000
```

중요:

- `--run-version v6`를 다시 쓰지 말 것.
- 같은 version을 재사용하면 기존 checkpoint에서 resume될 수 있다.
- 새 비교 실험은 항상 새 version과 `--reset-model`을 쓸 것.

viewer:

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v7/pmk_cf_self_rally_v7_model.zip \
  --episodes 100
```

정량 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v7/pmk_cf_self_rally_v7_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v7_contact_diagnosis
```

## v7 평가 기준

초반에는 `failure_reason=low_apex_contact`가 많이 나오는 것이 정상일 수 있다. 학습이 진행되면 이것이 줄어야 한다.

볼 지표:

- mean projected apex height가 v6의 `0.199m`에서 최소 `0.25m+`로 올라가는지
- useful projected apex가 `0.30~0.40m` window에 남는지
- useful contact rate가 `0.043`보다 올라가는지
- next intercept reachable rate가 `0.345`보다 올라가는지
- `ball_speed_limit`이 크게 늘지 않는지
- contact 수만 많고 useful가 낮으면 실패다.

## 실패 시 다음 판단

v7도 실패하면 보상 weight만 더 키우지 말 것.

우선 분석:

1. `run_ppo_rebound_analysis.py`로 contact CSV를 만든다.
2. `projected_contact_apex_height_above_racket`, `actual_outgoing_velocity_z`, `desired_outgoing_velocity_z`, `next_intercept_xy_error`, `next_intercept_reachable`, `failure_reason`를 본다.
3. `low_apex_contact`가 너무 많으면 vertical primitive/controller 추종 문제를 본다.
4. `ball_speed_limit`이 많으면 vertical primitive가 과한 것이다.
5. apex는 맞는데 next intercept가 멀면 tilt/trajectory target 문제를 본다.

가능한 다음 수정 방향:

- 저높이 종료 threshold/grace 조정
- vertical primitive가 desired z를 더 잘 따라가도록 controller velocity/position tracking 분석
- reset curriculum을 더 단계적으로 구성
- heuristic/bootstrap을 다시 쓰되, 낮은 contact가 아니라 useful cycle 이후 샘플만 쓰기

## 메모리 공유 관련

새 세션이나 다른 에이전트가 이 세션의 내부 기억을 자동으로 공유한다고 가정하지 말 것.

공유된다고 믿을 수 있는 것은 파일 시스템에 남은 코드, artifacts, report, 이 handoff 파일이다. 다음 에이전트는 먼저 이 파일과 `docs/report/31_v6_low_bounce_loop_and_strict_cycle_fix.md`를 읽고 이어가면 된다.
