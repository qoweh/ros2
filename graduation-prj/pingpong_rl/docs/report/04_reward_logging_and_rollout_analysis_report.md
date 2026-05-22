# Reward Logging과 Rollout Analysis 4차 작업 보고

## 1. 작업 목표

이번 단계의 목표는 reward/termination 의미를 바꾸지 않고, PPO 이전에 env signal을 분석할 수 있는 logging/export 경로를 추가하는 것이었다.

구체적으로는 아래를 목표로 잡았다.

- reward decomposition logging 추가
- episode 종료 원인 logging 추가
- contact 순간 velocity distribution 수집 경로 추가
- CSV/JSON export 지원 추가

## 2. 이번 작업에서 구현한 내용

### 2.1 env `info` 확장
`PingPongEEDeltaEnv.step()`과 `reset()`에서 아래 logging 필드를 추가했다.

- reward 분해:
  - `reward_total`
  - `reward_height`
  - `reward_distance`
  - `reward_contact`
  - `reward_success`
  - `reward_failure`
- episode 경계:
  - `terminated`
  - `truncated`
  - `success_reason`
  - `failure_reason`
  - `time_limit_reached`
  - `episode_steps`
- velocity 관측:
  - `ball_velocity_x`
  - `ball_velocity_y`
  - `ball_velocity_z`
  - `ball_speed_norm`

중요:
- reward 식 자체는 바꾸지 않았다.
- `reward_distance`, `reward_success`는 현재 reward semantics를 유지하기 위해 `0.0` placeholder로만 기록한다.

### 2.2 substep contact trace 추가
`PingPongSim`에 `step_with_contact_trace()`를 추가했다.

이 함수는 한 env step 내부 substep 중에서 처음 관측된 `ball_geom`-`racket_head` contact를 잡아 아래 값을 기록한다.

- `contact_observed_during_step`
- `contact_substep`
- `contact_ball_velocity_x`
- `contact_ball_velocity_y`
- `contact_ball_velocity_z`
- `contact_ball_speed_norm`

이 변경은 logging용이며, reward/success 판정 의미는 그대로 유지했다.

### 2.3 rollout export 스크립트 추가
새 스크립트:

- `pingpong_rl/scripts/run_ee_rollout_analysis.py`

산출물:

- episode summary CSV
- step-wise reward/logging CSV
- contact event CSV
- contact velocity summary JSON

## 3. 검증 결과

### 3.1 focused unittest
검증 명령:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
python -m unittest discover -s pingpong_rl/tests -p 'test_scene_load.py'
```

결과:
- 14개 테스트 통과

### 3.2 rollout export smoke test
검증 명령:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
python pingpong_rl/scripts/run_ee_rollout_analysis.py \
  --episodes 1 \
  --max-episode-steps 80 \
  --output-dir /tmp/pingpong_rollout_analysis \
  --output-prefix smoke
```

실제 결과:
- episode 수: `1`
- success: `0`
- failure: `1`
- contact event 수: `7`
- episode failure reason: `ball_out_of_bounds`

contact velocity 요약:
- `ball_velocity_z`
  - `p50 = 0.4687`
  - `p75 = 0.5757`
  - `p90 = 1.0833`
  - `max = 1.7415`

episode CSV 기준 누적 reward:
- `reward_total_sum = 3.0837`
- `reward_height_sum = 3.3337`
- `reward_contact_sum = 0.0`
- `reward_success_sum = 0.0`
- `reward_failure_sum = -0.25`

즉 smoke run만 봐도 현재는 성공 없이 height term이 reward 대부분을 차지하고 있다는 점이 바로 드러난다.

## 4. 이번 단계에서 의도적으로 하지 않은 것

이번 단계에서는 아래를 의도적으로 하지 않았다.

- reward coefficient 조정
- reward 식 변경
- success threshold 자동 수정
- scene physics/contact parameter 수정

즉 이번 작업은 튜닝이 아니라, 이후 튜닝 판단 근거를 만드는 관측 계층 추가다.

## 5. 다음 작업 제안

### 5.1 바로 이어서 할 것
- multi-episode rollout으로 `ball_velocity_z` 분포를 더 모으기
- success/failure/time-limit 비율을 먼저 보기
- reward decomposition 누적값이 실제로 어떤 항에 치우치는지 확인하기

### 5.2 그 다음
- 분포를 본 뒤에만 `success_velocity_threshold` 보정 검토
- reward에서 contact/success 항을 강화할지 판단
- PPO baseline 학습 로그와 현재 CSV/JSON log를 연결