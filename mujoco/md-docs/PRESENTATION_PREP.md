# 발표 준비 통합 정리

작성일: 2026-06-06  
대상 저장소: `/Users/pilt/project-collection/ros2/mujoco`

이 문서는 발표 준비를 위해 저장소 전체를 한 번 훑은 결과를 모은 것이다. 핵심 목적은 세 가지다.

1. git 기록 기준으로 프로젝트가 어떤 흐름으로 발전했는지 설명한다.
2. 코드 기준으로 강화학습이 어떻게 진행되고, 환경이 어떻게 구성되는지 답변할 수 있게 정리한다.
3. docs/md 파일들이 무엇을 다루는지 색인화하고, 발표용 시각화 후보를 정리한다.

현재 작업 트리에는 기존 발표팩 파일 수정분과 새 파일이 이미 있었다.

- `pingpong_rl2/docs/rl_presentation_pack/07_v35_training_review_and_next_plan.md`
- `pingpong_rl2/docs/rl_presentation_pack/README.md`
- `pingpong_rl2/docs/rl_presentation_pack/08_v36_wider_domain_review.md`

이 문서는 위 파일들을 덮어쓰지 않고 새로 작성했다.

발표 범위 기준:

- `pingpong_rl3`는 제외한다.
- 최종 모델은 `pingpong_rl2/artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed`로 본다.
- v40 이후 또는 2공 확장은 발표 본문/질문 대비 범위에서 제외한다.

## 1. 프로젝트 한 줄 정의

이 프로젝트는 MuJoCo에서 Franka Panda 로봇팔과 라켓, 탁구공을 시뮬레이션하고, Stable-Baselines3 PPO로 탁구공을 계속 올려치는 keep-up 정책을 학습시키는 강화학습 프로젝트다.

ROS2 폴더 아래에 있지만, 현재 학습 및 시뮬레이션 코어는 ROS2 노드가 아니라 Python 기반 `MuJoCo + Gymnasium + Stable-Baselines3` 구조다. `pyproject.toml`의 핵심 의존성도 `gymnasium`, `mujoco`, `numpy`, `stable-baselines3`, `tensorboard`다.

## 2. git 기준 개발 흐름

### 2.1 날짜별 커밋 밀도

| 날짜 | 커밋 수 | 의미 |
| --- | ---: | --- |
| 2026-05-08 | 1 | 저장소 시작 |
| 2026-05-11 | 1 | `.gitignore` 등 기본 정리 |
| 2026-05-12 | 6 | SO-101/초기 MuJoCo 장면, 라켓/공 배치, viewer 모드 |
| 2026-05-13 | 6 | EE 제어, flat observation, reward/logging, PPO 호환성 |
| 2026-05-14 | 3 | 논문/스크립트 정리, `car_rl` 토이 PPO 시작 |
| 2026-05-18 | 6 | `car_rl`와 `pingpong_rl` 비교, keep-up baseline 및 보상/제어 갱신 |
| 2026-05-19 | 1 | 문서와 curriculum 정리 |
| 2026-05-21 | 4 | table tennis 방향 문서, 안정성 리뷰, EE delta/rebound shaping |
| 2026-05-22 | 10 | `position_tilt`, assist/curriculum, Franka asset, `PingPongSim` 개선 |
| 2026-05-26 | 4 | `pingpong_rl2` 시작, utilities/report/resume/reset 정리 |
| 2026-05-27 | 5 | tilt profile, strike readiness, predicted intercept |
| 2026-05-28 | 9 | velocity-domain observation, contact trace, presets, rebound analysis |
| 2026-05-29 | 5 | phase contract, heuristic, follow-up strike, feasibility map |
| 2026-06-01 | 24 | contact-frame primitive, residual RL, self-rally planner, sweet spot, outgoing objective |
| 2026-06-02 | 6 | low-apex termination/recovery, stable cycle, lateral stability |
| 2026-06-03 | 12 | v17-v25 residual action 확장, height-qualified reward, 30-bounce horizon |
| 2026-06-04 | 10 | v26 이후 broad reset, spin/tracking residual, staged distribution |
| 2026-06-05 | 1 | 발표팩 데이터와 시각화 스크립트 추가 |

### 2.2 단계별 서사

| 단계 | 기간 | 핵심 변화 | 발표 메시지 |
| --- | --- | --- | --- |
| 초기 구상 | 2026-05-08~05-12 | SO-101/Gazebo 후보를 검토하다가 MuJoCo Python 중심으로 이동 | 물리 시뮬레이션과 RL 반복 속도를 우선했다 |
| 1차 탁구 환경 | 2026-05-12~05-13 | 탁구공, 라켓, 로봇팔 장면과 EE delta 환경 구축 | 문제를 Gymnasium 스타일 RL task로 바꿨다 |
| PPO 감 잡기 | 2026-05-14~05-18 | `car_rl` 2D 차량 토이 프로젝트 추가 | PPO, reward shaping, monitor/eval 흐름을 작은 문제에서 검증했다 |
| `pingpong_rl` v1 | 2026-05-18~05-22 | 29D 관측, 3D/5D action, active hit, tilt assist, curriculum | 단순 위치 제어만으로는 반복 keep-up이 어렵다는 병목을 찾았다 |
| `pingpong_rl2` 재설계 | 2026-05-26~05-29 | reset, contact trace, rebound analysis, heuristic gate, phase contract | reward 튜닝보다 contact/control feasibility 확인이 먼저라는 결론 |
| contact-frame/self-rally | 2026-06-01~06-03 | 접촉 좌표계 action, desired outgoing velocity, low-apex recovery, stable cycle | 정책이 직접 라켓 목표와 반발 궤적을 조정하도록 action ownership을 넓혔다 |
| 장기 랠리와 robust reset | 2026-06-03~06-04 | v25 이후 long horizon, broad XYZ reset, 17D action, spin, staged distribution | 단발 성공에서 넓은 초기조건의 긴 랠리로 목표가 바뀌었다 |
| 최종 모델 정리와 발표 준비 | 2026-06-04~06-05 및 artifacts | v36 기반 resume, v39 mid curriculum fixed, 발표팩 CSV/PNG/outline | 최종 발표 기준 모델은 `keep1_v39_17d_mid_curriculum_fixed`다 |

## 3. 전체 시스템 구조

### 3.1 실행 구조

```text
MJCF scene/assets
  -> PingPongSim
  -> RL Env: PingPongEEDeltaEnv / PingPongKeepUpEnv
  -> Gymnasium wrapper
  -> SB3 VecEnv + VecMonitor
  -> PPO policy
  -> Monitor CSV, contacts CSV, training_summary.json, evaluation summary
  -> viewer / rebound analysis / presentation visuals
```

### 3.2 제어 구조

정책이 로봇 관절 토크를 직접 내는 구조는 아니다. 정책 action은 라켓의 목표 위치, 기울기, 속도, 목표 반발 궤적에 대한 residual이다.

```text
Observation
  -> PPO policy
  -> continuous residual action
  -> target position / tilt / velocity / desired outgoing velocity
  -> RacketCartesianController
  -> damped Jacobian IK
  -> joint targets
  -> MuJoCo substeps
  -> contact trace + reward + termination
```

이 구조의 장점은 로봇팔의 저수준 역기구학은 controller가 맡고, RL은 탁구 과제에서 중요한 "어디서, 어떤 방향으로, 얼마나 강하게 칠 것인가"를 학습한다는 점이다.

## 4. 강화학습 진행 방식

### 4.1 공통 학습 루프

1. `reset`: 공의 위치, 높이, 속도, 스핀을 샘플링하고 로봇/라켓을 초기화한다.
2. `observation`: 관절 상태, 라켓 상태, 공 상태, 예측 intercept, phase/context 등을 flat vector로 만든다.
3. `policy`: PPO actor가 continuous action을 출력한다.
4. `action clipping`: 환경별 action limit으로 안전하게 자른다.
5. `controller`: action을 라켓 목표 위치/속도/tilt로 해석하고 IK로 joint target을 만든다.
6. `simulation`: MuJoCo substep을 진행하며 접촉 여부와 접촉 순간 속도/위치/법선 등을 기록한다.
7. `reward`: tracking, contact quality, apex, next intercept, action penalty, failure penalty 등을 합산한다.
8. `done`: floor/body/out-of-bounds/speed/low-apex 등 실패 또는 time limit으로 종료한다.
9. `logging`: monitor CSV, contact CSV, summary JSON, TensorBoard에 기록한다.

### 4.2 PPO 선택 이유

PPO는 continuous control에 잘 맞고, Stable-Baselines3에서 Gymnasium/MuJoCo 환경과 바로 연결된다. 이 프로젝트의 action은 이산 action이 아니라 라켓 위치/속도/tilt residual 같은 연속값이므로 DQN식 Q-table이나 epsilon-greedy보다 PPO actor-critic이 자연스럽다.

중요한 발표 포인트: PPO에는 DQN의 `epsilon`이 없다. 탐험성 변화는 epsilon 그래프가 아니라 `entropy`, `log_std`, `action std`, action usage/saturation 같은 지표로 설명해야 한다.

## 5. 코드별 환경 정리

### 5.1 `car_rl`

`car_rl`은 탁구 본 실험 전에 PPO 흐름을 익히기 위한 2D 차량 토이 프로젝트다.

| 항목 | 내용 |
| --- | --- |
| 환경 | `car_rl/car_env.py`의 `CarEnv` |
| 물리 | MuJoCo XML을 쓰지만 차량 이동은 custom kinematic bicycle integration 성격이 강함 |
| action | 2D: throttle, steering |
| observation | 10D: local goal, distance, yaw error, speed, steering, previous actions, remaining time |
| reward | 목표 접근 progress, yaw alignment, 거리 penalty, action penalty, success/out-of-bounds 보상/벌점 |
| 학습 | `car_rl/train.py`, SB3 PPO, EvalCallback, TensorBoard |
| 평가 | `car_rl/test.py`, deterministic rollout, success rate/mean reward/mean distance |

발표에서의 역할은 "PPO와 reward shaping을 작은 문제에서 먼저 검증했다"는 배경 설명이다.

### 5.2 `pingpong_rl` v1

초기 1공 탁구 환경이다.

| 항목 | 내용 |
| --- | --- |
| 시뮬레이터 | `pingpong_rl/src/pingpong_rl/envs/pingpong_env.py`의 `PingPongSim` |
| 환경 | `PingPongEEDeltaEnv` |
| action mode | `position` 3D, `position_tilt` 5D |
| observation | 29D: joint pos/vel, racket pos/vel, target pos, ball pos/vel |
| 성공 판정 | 접촉 이벤트, active hit, 충분한 upward ball velocity, projected apex |
| reward | contact bonus, height/lift, tracking/centering, lateral rebound, active hit, orientation/joint/action penalty, failure penalty |
| 학습 | `scripts/run_ppo_baseline.py`, curriculum callback, PPO logging callback |
| 분석 | episodes/steps/contacts CSV와 training summary JSON |

핵심 학습: 단순 EE 위치 제어와 reward shaping만으로는 반복 keep-up이 안정적으로 이어지지 않았다. 접촉 순간의 품질, 다음 공의 치기 쉬움, 라켓의 속도/tilt ownership이 중요해졌다.

### 5.3 `pingpong_rl2` v2

현재 발표의 중심이 되는 1공 keep-up 환경이다.

| 항목 | 내용 |
| --- | --- |
| 핵심 환경 | `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`의 `PingPongKeepUpEnv` |
| 시뮬레이터 | `pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py` |
| controller | `controllers/ee_pose_controller.py` |
| heuristic | `controllers/heuristic_keepup.py` |
| PPO entrypoint | `scripts/run_ppo_learning.py` |
| 평가 | `scripts/run_ppo_evaluation.py`, `scripts/run_ppo_rebound_analysis.py` |
| 실행 프리셋 | `configs/*.json`, `run_ppo_learning.py` 내부 preset |
| 주요 산출물 | `artifacts/ppo_runs/*/monitor_*.csv`, `*_training_summary.json`, `analysis/*_summary.json`, contacts/episodes CSV |

#### 5.3.1 observation 구성

기본 observation은 35D다.

| 구성 | 차원 | 의미 |
| --- | ---: | --- |
| joint_positions | 7 | Franka 관절 위치 |
| joint_velocities | 7 | Franka 관절 속도 |
| racket_position | 3 | 라켓 위치 |
| racket_velocity | 3 | 라켓 속도 |
| target_position | 3 | controller target |
| ball_position | 3 | 공 위치 |
| ball_velocity | 3 | 공 속도 |
| ball_relative_position | 3 | 공과 라켓/target 기준 상대 위치 |
| predicted_intercept_relative_xy | 2 | 예측 접촉점 XY |
| predicted_intercept_time | 1 | 예측 접촉 시간 |

최종 v39 17D 모델 계열은 여기에 task phase, contact context, next intercept, desired outgoing velocity, racket face normal, target tilt가 붙어 발표팩 기준 55D observation으로 설명된다.

#### 5.3.2 action mode 발전

| action mode | 차원 | 의미 |
| --- | ---: | --- |
| `position` | 3 | 라켓 목표 위치 residual |
| `position_tilt` | 5 | 위치 3D + pitch/roll |
| `position_strike` | 3 | strike target 기반 위치 residual |
| `position_strike_tilt` | 5 | strike + tilt |
| `position_strike_tilt_lift` | 6 | strike/tilt + follow-up lift |
| `position_contact_frame` | 5 | 접촉 좌표계 위치 + tilt |
| `position_contact_frame_velocity_residual` | 8 | contact-frame + outgoing velocity residual |
| `position_contact_frame_velocity_tilt_residual` | 11 | 라켓 vz/tilt scale residual 추가 |
| `position_contact_frame_velocity_tilt_lateral_residual` | 13 | 라켓 vx/vy residual 추가 |
| `position_contact_frame_velocity_tilt_lateral_apex_residual` | 15 | 목표 apex z/strike plane z 추가 |
| `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual` | 17 | tracking vx/vy residual까지 포함한 최종 v39 계열 |

최종 17D action 이름은 발표팩 데이터 기준 다음과 같다.

`radial`, `tangent`, `z`, `tilt_pitch`, `tilt_roll`, `vz_scale`, `outgoing_x`, `outgoing_y`, `racket_vz`, `trajectory_tilt_scale`, `centering_tilt_scale`, `racket_vx`, `racket_vy`, `target_apex_z`, `strike_plane_z`, `tracking_vx`, `tracking_vy`.

#### 5.3.3 성공과 실패 기준

`PingPongKeepUpEnv`의 useful bounce 성공은 단순 접촉이 아니다. 핵심 조건은 다음과 같다.

- 새 contact event가 있어야 한다.
- 접촉 후 공의 z 속도가 threshold보다 커야 한다.
- 접촉 순간 라켓 z 속도가 충분히 위쪽이어야 한다.
- 라켓 중심과 공 접촉 XY alignment가 너무 벗어나면 안 된다.
- projected contact apex가 성공 높이 window 안에 있어야 한다.
- 옵션에 따라 next intercept가 reachable이어야 하고 easy-next-ball score가 threshold를 넘어야 한다.

실패는 floor contact, robot body contact, ball out-of-bounds, speed limit, low-apex contact, nonuseful contact 등으로 나뉜다.

#### 5.3.4 reward 구조

reward는 하나의 단순 식보다 "성공으로 이어지는 중간 신호"를 여러 항으로 나눈 구조다.

- 추적: ball/racket/target alignment
- 접촉: contact bonus, stale/nonuseful penalty
- 높이: apex match, apex under-target penalty, low-apex recovery progress
- 다음 공: next intercept reachable, easy-next-ball score, next intercept XY error
- 궤적: desired outgoing velocity match/error
- 안정성: lateral stability, stable contact, stable cycle
- 제어 비용: action norm, tilt angle, tilt delta
- 실패 비용: floor, body, out-of-bounds, speed, low apex

발표에서는 전체 reward를 다 외우기보다 "최종 목표가 긴 랠리이므로 보상도 접촉 성공만이 아니라 다음 공이 다시 칠 수 있는 상태인지까지 본다"라고 말하는 것이 좋다.

## 6. 주요 성능 산출물

### 6.1 발표팩 기본 지표: v25~v35

`pingpong_rl2/docs/rl_presentation_pack/data/version_metrics.csv`와 `long_horizon_metrics.csv`에 v25~v35 비교가 정리되어 있다.

| 모델 | train eval mean useful | max useful | 30+ rate | long eval mean useful | long max | long 30+ rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v25 | 30.4 | 48 | 0.65 | - | - | - |
| v26 | 49.9 | 82 | 0.733 | - | - | - |
| v30 | 59.2 | 88 | 0.787 | - | - | - |
| v31 | 35.5 | 95 | 0.47 | - | - | - |
| v32 17D | 49.7 | 96 | 0.66 | 79.7 | 159 | 0.80 |
| v33 17D | 47.0 | 47 | 1.00 | 91.2 | 170 | 0.75 |
| v34 17D | 102 | 175 | 0.79 | 116 | 170 | 0.90 |
| v35 17D | 105 | 171 | 0.82 | 102 | 169 | 0.80 |

v34/v35 사이의 메시지는 "v35는 train eval에서 안정성이 좋아 보이지만 long-horizon target hit에서는 v34가 더 강한 면이 있다"다.

### 6.2 최종 모델 기준: `keep1_v39_17d_mid_curriculum_fixed`

최종 발표 기준 모델은 아래 artifacts에 있다.

- 모델: `pingpong_rl2/artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip`
- 학습 summary: `pingpong_rl2/artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json`
- long eval summary: `pingpong_rl2/artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/analysis/keep1_v39_oldbase_long7200_eval20_summary.json`
- 시작 모델: `keep1_v36_17d_balanced_xyz012`
- 학습 방식: resume, 700,000 timesteps
- action mode: `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual`
- reset: XY disk radius 0.13m, height [0.22, 0.52], XY velocity 0.045, Z velocity [-0.14, 0.04]
- eval step limit: 7200
- stable cycle reward cap: 30
- low-apex contact grace count: 6

| 모델 | train eval mean useful | train max | train 30+ rate | long eval mean useful | long max | long 30+ rate | 메모 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v36 | 107 | 180 | 0.87 | 94.5 | 165 | 0.80 | balanced XYZ, train eval은 좋지만 long은 v34보다 낮음 |
| v37 | 78.3 | 172 | 0.70 | 72.8 | 152 | 0.70 | wide curriculum에서 성능 하락 |
| v38 | 50.7 | 162 | 0.55 | 41.4 | 154 | 0.50 | low-apex가 다시 커짐 |
| v39 final | 119.52 | 181 | 0.83 | 130.95 | 182 | 0.90 | 최종 발표 모델 |

v39 train eval 100 episode 실패 분포:

- time_limit: 63
- ball_out_of_bounds: 22
- robot_body_contact: 9
- floor_contact: 3
- ball_speed_limit: 2
- low_apex_contact: 1

v39 long eval 20 episode 실패 분포:

- time_limit: 14
- ball_out_of_bounds: 3
- robot_body_contact: 2
- ball_speed_limit: 1

v39 coverage/stress 평가도 이미 있다. 발표 본문에는 최종 모델 성능으로 long eval을 쓰고, 질문이 들어오면 아래 stress 결과로 "분포가 더 넓어지면 성능이 떨어진다"는 한계를 설명할 수 있다.

| 평가 파일 | mean useful | max useful | 30+ rate | 의미 |
| --- | ---: | ---: | ---: | --- |
| `cov_v39_all015_vel006_z018_eval12_3600_summary.json` | 45.5 | 86 | 0.583 | XY/속도/높이를 모두 넓힌 coverage |
| `cov_v39_height018056_mid_eval12_3600_summary.json` | 58.9 | 92 | 0.75 | 높이 범위를 넓힘 |
| `cov_v39_height020054_mid_eval12_3600_summary.json` | 52.8 | 89 | 0.667 | 다른 높이 범위 |
| `cov_v39_square_xy013_mid_eval12_3600_summary.json` | 49.8 | 91 | 0.667 | disk 대신 square XY |
| `cov_v39_vel006_vz018_xy012_eval12_3600_summary.json` | 50.7 | 85 | 0.667 | 속도 범위를 넓힘 |
| `cov_v39_xy015_basevel_eval12_3600_summary.json` | 33.1 | 91 | 0.417 | XY 0.15m |
| `cov_v39_xy016_basevel_eval12_3600_summary.json` | 22.4 | 84 | 0.25 | XY 0.16m |
| `stress_v39_xy013_vel045_z014_eval12_3600_summary.json` | 36.2 | 85 | 0.583 | stress 조건 1 |
| `stress_v39_xy014_vel05_z016_eval12_3600_summary.json` | 42.0 | 85 | 0.583 | stress 조건 2 |

발표 메시지: v39는 기본/oldbase long eval에서는 긴 랠리가 가능하지만, reset 분포를 더 넓히면 성능이 떨어진다. 즉 "최종 모델은 성공했지만, 완전한 일반화까지는 추가 curriculum/domain randomization이 필요하다"가 한계다.

## 7. 문서 및 Markdown 파일 카탈로그

### 7.1 루트와 초기 제안/핸드오프

| 파일 | 다루는 내용 |
| --- | --- |
| `NOTICE.md` | 프로젝트 notice, 외부 자산 관련 고지 |
| `RULE.md` | 코딩/작업 원칙 |
| `TODO.md` | 초기 세팅 할 일 |
| `TODO2.md` | 짧은 추가 TODO |
| `agent-answer.md` | 이전 에이전트 답변/작업 정리로 보이는 참고 문서 |
| `todo-agent.md` | 에이전트용 TODO와 작업 지시 메모 |
| `참고.md` | 짧은 참고 메모 |
| `web-service.md` | 웹 데모/서비스 구상, 프론트엔드/백엔드/배포/문서 페이지 계획 |
| `mujoco/md-docs/init-mds/claude-proposal.md` | SO-101, Gazebo, ros_gz, MuJoCo 후보 비교 |
| `mujoco/md-docs/init-mds/claude-proposal2.md` | 로봇팔 강화학습 시뮬레이션 개발 방향 제안 |
| `mujoco/md-docs/init-mds/gemini-proposa.md` | SO-101 Gazebo 컨트롤 개발 후보군 |
| `mujoco/md-docs/init-mds/gemini-proposal2.md` | 로봇팔 RL 프로젝트 제안서 v2 |
| `next-agent-master-instructions.md` | 다음 에이전트가 복잡도, CLI, reward, 정책 구조를 정리하라는 종합 지시 |
| `next-agent-rethink-keepup-prompt.md` | keep-up 완성을 위해 MDP/reward/controller를 재검토하라는 프롬프트 |
| `next-agent-keepup-completion-plan.md` | easy-next-ball metric, reward/policy 개선, 실험 프로토콜 지시 |
| `next-agent-contact-control-feasibility-prompt.md` | reward 튜닝보다 contact/control feasibility를 먼저 검증하라는 지시 |
| `next-agent-5.5-keepup-gate-and-primitive.md` | contact-frame primitive gate와 scripted 검증 기준 |
| `next-agent-contact-primitive-handoff-2026-06-01.md` | 2026-06-01 기준 contact primitive 진행상황과 다음 선택지 |
| `next-agent-direct-trajectory-prompt.md` | 직접 궤적 목표, outgoing trajectory metric/reward 전환 지시 |
| `next-agent-spark-keepup-completion-plan-2026-06-01.md` | self-rally 완성 플랜과 PPO 재개 조건 |

### 7.2 `car_rl`

| 파일 | 다루는 내용 |
| --- | --- |
| `car_rl/PROJECT_SUMMARY.md` | `car_rl` 토이 프로젝트 전체 요약 |
| `car_rl/report/00_README.md` | 보고서 읽는 순서와 파일 안내 |
| `car_rl/report/01_execution_flow.md` | 실행 흐름, train/test/viewer 경로 |
| `car_rl/report/02_car_env_mujoco_gymnasium.md` | `CarEnv`, MuJoCo, Gymnasium wrapping 설명 |
| `car_rl/report/03_viewer_loop_and_time.md` | viewer loop와 시간 흐름 |
| `car_rl/report/04_reward_and_task_design.md` | 차량 목표 도달 task와 reward 설계 |
| `car_rl/report/05_ppo_core.md` | PPO 알고리즘 핵심 개념 |
| `car_rl/report/06_training_testing_models.md` | 학습/평가/모델 파일 관리 |
| `car_rl/report/07_faq_debugging.md` | FAQ와 디버깅 메모 |

### 7.3 `pingpong_rl` v1 문서

| 파일 | 다루는 내용 |
| --- | --- |
| `pingpong_rl/README.md` | v1 빠른 안내, 읽기 순서, 실행 파일, viewer와 학습 환경 차이 |
| `pingpong_rl/assets/franka/README.md` | Franka Panda MJCF asset 설명 |
| `pingpong_rl/assets/franka/CHANGELOG.md` | Franka asset changelog |
| `pingpong_rl/docs/log/00_index.md` | log 문서 인덱스 |
| `pingpong_rl/docs/log/01_pingpong_rl_debug_log.md` | v1 디버그 로그 |
| `pingpong_rl/docs/log/02_curriculum_and_reward_plan.md` | curriculum 및 reward 설계 메모 |
| `pingpong_rl/docs/log/help/01_mujoco_scene_setup_help.md` | MuJoCo scene 구성 도움말 |
| `pingpong_rl/docs/log/help/02_ee_viewer_demo_help.md` | EE viewer demo 사용법 |
| `pingpong_rl/docs/log/help/03_ball_bounce_and_ee_delta_env_help.md` | ball bounce와 EE delta env 도움말 |
| `pingpong_rl/docs/log/help/04_reward_logging_and_rollout_analysis_help.md` | reward logging과 rollout analysis |
| `pingpong_rl/docs/log/help/05_ppo_logging_bridge_help.md` | PPO logging bridge |
| `pingpong_rl/docs/log/help/06_robot_table_tennis_project_direction_help.md` | 로봇 탁구 졸업작품 방향 정리 |
| `pingpong_rl/docs/log/help/07_upward_bounce_rl_end_to_end_help.md` | upward bounce RL 전체 가이드 |
| `pingpong_rl/docs/log/step/01_project_pipeline_unified.md` | MuJoCo 환경 구축과 RL pipeline 기준 |
| `pingpong_rl/docs/log/step/02_ee_task_space_next_steps.md` | EE task-space 다음 단계 |
| `pingpong_rl/docs/log/step/03_ee_delta_env_contract.md` | EE delta env의 observation/action/reward 계약 |
| `pingpong_rl/docs/report/00_index.md` | report 인덱스 |
| `pingpong_rl/docs/report/01_mujoco_scene_setup_report.md` | MuJoCo 환경 구축 1차 작업 |
| `pingpong_rl/docs/report/02_ee_viewer_demo_report.md` | EE viewer demo 작업 |
| `pingpong_rl/docs/report/03_ball_bounce_and_ee_delta_env_report.md` | ball bounce 튜닝과 EE delta env |
| `pingpong_rl/docs/report/04_reward_logging_and_rollout_analysis_report.md` | reward logging과 rollout analysis |
| `pingpong_rl/docs/report/05_preppo_distribution_and_ppo_logging_report.md` | pre-PPO 분포 분석과 PPO logging 연결 |
| `pingpong_rl/docs/report/06_competitive_robot_table_tennis_paper_report.md` | 인간 수준 경쟁 탁구 로봇 논문 정리 |
| `pingpong_rl/docs/report/07_takeaways_from_competitive_table_tennis_for_bounce_rl.md` | 논문에서 upward bounce RL로 가져갈 점 |
| `pingpong_rl/docs/report/08_car_rl_vs_pingpong_rl_structure_report.md` | `car_rl`와 `pingpong_rl` 구조 비교 |
| `pingpong_rl/docs/report/09_pingpong_pending_tasks_audit.md` | v1 미완료 작업 audit |
| `pingpong_rl/docs/report/10_pingpong_rl_status_report.md` | v1 현재 상태 보고 |
| `pingpong_rl/docs/report/11_keepup_stability_review.md` | stable keep-up 기준 재검토 |
| `pingpong_rl/docs/report/12_keepup_v3_followup_and_control_assist_report.md` | `ppo_keepup_v3` 후속 실험과 control assist |
| `pingpong_rl/docs/report/13_v7_check_position_tilt_and_curriculum_report.md` | `ppo_keepup_v7`, `position_tilt`, curriculum 검토 |

### 7.4 `pingpong_rl2` 분석/보고서

| 파일 | 다루는 내용 |
| --- | --- |
| `pingpong_rl2/README.md` | v2 구조와 실행 안내 |
| `pingpong_rl2/archive/README.md` | archive notes |
| `pingpong_rl2/assets/franka/README.md` | Franka Panda MJCF asset 설명 |
| `pingpong_rl2/assets/franka/CHANGELOG.md` | Franka asset changelog |
| `pingpong_rl2/next-agent-handoff-2026-06-02.md` | 2026-06-02 기준 다음 에이전트 handoff |
| `pingpong_rl2/docs/analysis/control_structure_analysis.md` | control 구조 분석 |
| `pingpong_rl2/docs/analysis/learning_failure_analysis.md` | 학습 실패 원인 분석 |
| `pingpong_rl2/docs/analysis/pending_decisions.md` | 남은 의사결정 |
| `pingpong_rl2/docs/analysis/reset_distribution_analysis.md` | reset distribution 분석 |
| `pingpong_rl2/docs/analysis/reward_dependency_analysis.md` | reward term 의존성 분석 |
| `pingpong_rl2/docs/report/00_index.md` | v2 report 인덱스 |
| `pingpong_rl2/docs/report/01_pingpong_rl2_minimal_keepup_status_report.md` | minimal keep-up 상태 |
| `pingpong_rl2/docs/report/02_pingpong_rl2_position_tilt_and_rebound_report.md` | `position_tilt` branch와 rebound analysis |
| `pingpong_rl2/docs/report/03_pingpong_rl2_position_tilt_chatter_fix_report.md` | tilt chatter fix와 staged tilt profile |
| `pingpong_rl2/docs/report/04_pingpong_rl2_inward_tilt_direction_report.md` | inward tilt 방향 A/B |
| `pingpong_rl2/docs/report/05_project_completion_plan.md` | v2 완성 계획 |
| `pingpong_rl2/docs/report/06_learning_design_checklist.md` | 학습 설계 체크리스트 |
| `pingpong_rl2/docs/report/07_reward_policy_cleanup_plan.md` | reward/policy cleanup과 clean ablation |
| `pingpong_rl2/docs/report/08_easy_next_ball_completion_plan.md` | easy-next-ball completion plan |
| `pingpong_rl2/docs/report/09_keepup_task_rethink_plan.md` | keep-up task 재정의 |
| `pingpong_rl2/docs/report/10_keepup_phase_contract_implementation_report.md` | phase contract 구현 |
| `pingpong_rl2/docs/report/11_keepup_heuristic_variant_gate_report.md` | heuristic variant gate |
| `pingpong_rl2/docs/report/12_followup_strike_bootstrap_report.md` | follow-up strike PPO와 bootstrap |
| `pingpong_rl2/docs/report/13_direct_trajectory_objective_report.md` | direct trajectory objective |
| `pingpong_rl2/docs/report/14_contact_trace_sanity_report.md` | contact trace sanity |
| `pingpong_rl2/docs/report/15_contact_feasibility_map_report.md` | contact feasibility map |
| `pingpong_rl2/docs/report/16_contact_upper_bound_report.md` | contact upper bound |
| `pingpong_rl2/docs/report/17_contact_primitive_training_plan.md` | contact primitive 학습 계획 |
| `pingpong_rl2/docs/report/18_tilt_primitive_scripted_feasibility_report.md` | tilt primitive scripted feasibility |
| `pingpong_rl2/docs/report/19_non_tilt_contact_residual_report.md` | non-tilt contact residual |
| `pingpong_rl2/docs/report/20_state_dependent_contact_point_and_timing_report.md` | state-dependent contact point와 timing |
| `pingpong_rl2/docs/report/21_contact_frame_primitive_report.md` | contact-frame primitive 본문, 2026-06-01 후속 실험들이 가장 많이 누적된 핵심 보고서 |
| `pingpong_rl2/docs/report/22_sweet_spot_start_range_and_next_work.md` | sweet spot 시작 범위와 다음 작업 |
| `pingpong_rl2/docs/report/23_sweet_spot_completion_status_report.md` | sweet spot 완료 상태 |
| `pingpong_rl2/docs/report/24_self_rally_planner_primitive_report.md` | self-rally planner/primitive 구현 |
| `pingpong_rl2/docs/report/25_cleanup_and_self_rally_status.md` | artifact cleanup과 self-rally 상태 |
| `pingpong_rl2/docs/report/26_learning_runtime_parallel_and_v2_diagnosis.md` | parallel PPO runtime과 v2 diagnosis |
| `pingpong_rl2/docs/report/27_self_rally_execution_stabilization_report.md` | self-rally 실행 안정화 |
| `pingpong_rl2/docs/report/28_outward_racket_scene_and_state_tilt_report.md` | outward racket scene과 state-dependent tilt |
| `pingpong_rl2/docs/report/29_v4_tilt_timing_and_contact_quality_fix.md` | v4 tilt timing/contact quality reward |
| `pingpong_rl2/docs/report/30_v5_low_apex_and_height_reward_fix.md` | v5 low apex와 height reward |
| `pingpong_rl2/docs/report/31_v6_low_bounce_loop_and_strict_cycle_fix.md` | v6 low bounce loop와 strict cycle |
| `pingpong_rl2/docs/report/32_v7_low_apex_recovery_reward_fix.md` | v7 low-apex recovery reward |
| `pingpong_rl2/docs/report/33_v12_stable_cycle_material_and_training_review.md` | v12 이후 stable-cycle, 물성, 학습 전 검토 |
| `pingpong_rl2/docs/report/34_v13_fast_episode_low_apex_recovery_fix.md` | v13 fast episode와 low-apex recovery |
| `pingpong_rl2/docs/report/35_v14_lateral_out_of_bounds_and_v15_stability_gate.md` | v14 lateral out-of-bounds와 v15 stability gate |
| `pingpong_rl2/docs/report/36_rl_action_ownership_and_8d_residual_plan.md` | RL action ownership와 8D residual |
| `pingpong_rl2/docs/report/37_v16_8d_residual_review_and_v17_direction.md` | v16 8D residual 검토와 v17 방향 |
| `pingpong_rl2/docs/report/38_v17_contact_timing_velocity_tilt_residual.md` | v17 contact timing/velocity/tilt residual |
| `pingpong_rl2/docs/report/39_v17_action_scale_and_v18_lateral_residual.md` | v17 action scale과 v18 lateral residual |
| `pingpong_rl2/docs/report/40_v18_low_loop_and_v19_height_qualified_reward.md` | v18 low loop와 v19 height-qualified reward |
| `pingpong_rl2/docs/report/41_v19_boundary_out_and_v20_brake.md` | v19 boundary-out와 v20 brake |
| `pingpong_rl2/docs/report/42_v19_presentation_readiness_and_future_roadmap.md` | v19 발표 준비도와 미래 로드맵 |
| `pingpong_rl2/docs/report/43_v20_review_and_v21_apex_timing_residual.md` | v20 review와 v21 apex/timing residual |
| `pingpong_rl2/docs/report/44_v21_low_apex_review_and_v22_low_stable_window.md` | v21 low-apex review와 v22 low-stable window |
| `pingpong_rl2/docs/report/45_v22_review_and_v23_outward_timing_guard.md` | v22 review와 v23 outward timing guard |
| `pingpong_rl2/docs/report/46_v23_v24_review_and_v25_30_bounce_horizon.md` | v23/v24 review와 v25 30-bounce horizon |
| `pingpong_rl2/docs/report/47_run_ppo_learning_preset_config_reference.md` | `run_ppo_learning` preset/config reference |
| `pingpong_rl2/docs/report/48_v26_unlimited_broad_xyz_reset.md` | v26 unlimited horizon과 broad XYZ reset |
| `pingpong_rl2/docs/report/49_racket_center_tracking_spin_and_two_ball_plan.md` | racket-centered tracking residual, spin reset, 이후 확장 계획 메모 |
| `pingpong_rl2/docs/report/50_v28_tracking_spin_analysis_and_v29_staged_distribution.md` | v28 tracking/spin 분석과 v29 staged distribution |
| `pingpong_rl2/docs/report/51_v29_review_max_step_and_two_ball_mode.md` | v29 검토, max step, keep-up 모드 판단 |
| `pingpong_rl2/docs/report/52_v30_review_and_short_model_names.md` | v30 검토와 짧은 model name |
| `pingpong_rl2/docs/report/53_keep1_v31_wider_xy_and_keep2_model_plan.md` | v31 wider XY와 후속 모델 계획 |
| `pingpong_rl2/docs/report/54_v32_17d_transfer_finetune_report.md` | v32 17D transfer/fine-tune |

### 7.5 `pingpong_rl2/docs/rl_presentation_pack`

| 파일 | 다루는 내용 |
| --- | --- |
| `README.md` | 발표 정리 패키지 개요, 파일 구성, 생성된 자료 목록 |
| `00_pre_v25_trial_history.md` | v25 이전 시행착오를 git 기록 중심으로 정리 |
| `01_experiment_story.md` | v25 이후 실험 스토리라인 |
| `02_training_setup_and_troubleshooting.md` | 학습 세팅, 로그, 명령어, 문제 해결 |
| `03_action_observation_validation.md` | 17D action과 55D observation 검증 |
| `04_visualization_catalog.md` | 생성된 시각화 자료별 발표 포인트 |
| `05_slide_outline.md` | 실제 발표 슬라이드 구성 후보 |
| `06_source_data.md` | 수치 원본 report/artifacts 색인 |
| `07_v35_training_review_and_next_plan.md` | v35 학습 완료 검토와 v36 개선 방향 |
| `08_v36_wider_domain_review.md` | v36 학습 완료 검토와 넓은 영역 개선안 |
| `scripts/generate_visuals.py` | artifacts에서 CSV/PNG를 재생성하는 스크립트 |

생성된 주요 그림:

- `assets/01_version_timeline_metrics.png`
- `assets/02_failure_modes_by_version.png`
- `assets/03_long_horizon_target_hits.png`
- `assets/04_apex_height_distribution.png`
- `assets/05_action_usage_17d.png`
- `assets/06_action_ablation_mean_useful.png`
- `assets/07_observation_action_diagram.png`
- `assets/08_monitor_training_curves.png`

생성된 주요 데이터:

- `data/version_metrics.csv`
- `data/long_horizon_metrics.csv`
- `data/action_usage_v34.csv`
- `data/action_ablation_v34.csv`

## 8. 시각화 후보

### 8.1 바로 사용 가능한 그림

`pingpong_rl2/docs/rl_presentation_pack/assets`에 이미 8개 그림이 있다.

| 그림 | 발표에서 쓰는 메시지 |
| --- | --- |
| `01_version_timeline_metrics.png` | v25~v35로 갈수록 horizon, reset, action dimension이 바뀌며 mean useful bounce가 개선됨 |
| `02_failure_modes_by_version.png` | 성능 개선은 실패 모드의 이동과 함께 봐야 함 |
| `03_long_horizon_target_hits.png` | v34/v35 long-horizon 목표 비교 |
| `04_apex_height_distribution.png` | low-apex termination과 useful height 기준 설명 |
| `05_action_usage_17d.png` | 17D action 중 어떤 축을 많이 쓰는지 |
| `06_action_ablation_mean_useful.png` | 작게 보이는 action 축도 제거하면 성능이 떨어질 수 있음 |
| `07_observation_action_diagram.png` | 55D observation과 17D action의 관계 |
| `08_monitor_training_curves.png` | terminal log가 조용해도 monitor log로 학습 진행을 확인 가능 |

### 8.2 요청한 항목별 가능성

| 항목 | 가능 여부 | 후보 시각화 | 근거 데이터/코드 |
| --- | --- | --- | --- |
| 문제 정의 | 바로 가능 | "로봇팔이 공을 계속 올려쳐 다음 공도 칠 수 있게 만드는 문제" 한 장 그림 | `README`, `05_slide_outline.md`, env 설명 |
| 환경 구조도 | 바로 가능 | MuJoCo scene -> Sim -> Env -> PPO -> logs diagram | `pingpong_rl2/src`, `pyproject.toml` |
| 상태 공간 시각화 | 바로 가능 | 55D observation block chart | `keepup_env.py` observation components, `07_observation_action_diagram.png` |
| 행동 공간 시각화 | 바로 가능 | 17D action grouped bar/diagram | `action_usage_v34.csv`, `generate_visuals.py` |
| 보상 함수 설명 | 바로 가능 | reward term flow 또는 contact 전/후 reward stack | `keepup_env.py` `_reward_terms`, contacts CSV |
| 학습 곡선 Reward | 바로 가능 | monitor reward moving average | `monitor_*.monitor.csv`, `08_monitor_training_curves.png` |
| 성공률 추이 | 바로 가능 | 1+/10+/30+ useful bounce rate timeline | `version_metrics.csv`, training summaries |
| 에이전트 행동 시각화 | 가능 | MuJoCo viewer screenshot/GIF, racket/ball trajectory trail | viewer scripts, rollout contacts/episodes CSV |
| 정책 변화 비교 | 가능 | v25~v39 action mode evolution, action usage/ablation | git log, `action_usage_v34.csv`, `action_ablation_v34.csv`, v39 contacts CSV |
| Q-value/Value heatmap | 추가 구현 필요 | PPO critic value heatmap over ball XY/intercept time | SB3 policy value function, env state grid sampling |
| Epsilon 변화 | PPO라 직접 불가 | 대체: entropy/log_std/action std 변화 | TensorBoard, action usage/std |
| 하이퍼파라미터 비교 | 가능 | config table + 성능 bar chart | `configs/*.json`, `*_training_summary.json` |
| baseline 대비 성능 | 가능 | zero/heuristic/v1/v2/v34/v39 비교 | heuristic scripts, analysis summaries |
| 실패 사례 분석 | 바로 가능 | failure mode stacked bars, low-apex/out-of-bounds cases | `failure_counts`, `02_failure_modes_by_version.png` |
| 실제 데모/GIF | 가능 | viewer 녹화, long-rally best episode GIF | `run_ppo_viewer.py`, `viewer.py`, MuJoCo |
| 시스템 아키텍처 | 바로 가능 | package/code architecture diagram | source tree, training scripts |

### 8.3 추가로 만들면 좋은 그림 후보

1. `v25 -> v39 final` 확장 timeline  
   현재 발표팩은 v35 중심이다. v36~v39 summary를 추가하면 최종 모델까지의 성능 변화가 한 장으로 정리된다.

2. `v34/v35 vs v39 final` long-horizon 비교  
   v39 long eval mean useful 130.95, max 182이므로 최종 모델 성과를 보여주는 핵심 그림 후보다.

3. reward term contribution stacked bar  
   contacts CSV 또는 step log에서 reward term을 episode별로 합산하면 "왜 reward가 많은가"를 설득하기 좋다.

4. failure case gallery  
   low apex, ball out, robot body contact, speed limit을 viewer screenshot/GIF로 1장씩 보여주면 질문 대응이 쉬워진다.

5. value heatmap  
   PPO는 Q-value가 아니라 value function이다. 공의 상대 XY, 높이, intercept time을 grid로 바꿔 critic value를 그리면 "정책이 어떤 상태를 좋게 보는가"를 보여줄 수 있다.

6. action saturation heatmap  
   `action_usage_v34.csv`의 `mean_abs_over_limit`, `sat90_rate`를 v39 contacts CSV 기준으로 다시 계산하면 어떤 축이 최종 정책에서 병목인지 설명하기 좋다.

7. reset distribution coverage  
   v25~v39의 `reset_xy_range`, height, velocity range를 타임라인으로 보여주면 "과제를 점점 넓혔다"는 서사가 명확해진다.

8. v39 coverage/stress robustness chart  
   `cov_v39_*_summary.json`과 `stress_v39_*_summary.json`을 bar chart로 그리면 최종 모델의 일반화 한계를 설득력 있게 보여줄 수 있다.

## 9. 발표에서 강조할 핵심 답변

### 왜 MuJoCo인가?

로봇팔, 라켓, 공 접촉을 빠르게 반복 실험해야 했기 때문이다. ROS2/Gazebo 후보도 검토했지만, RL 학습에서는 빠른 reset, substep contact trace, Python Gymnasium 연동이 중요했다.

### 정책은 무엇을 직접 제어하나?

관절 토크를 직접 제어하지 않는다. 정책은 라켓 목표 위치/tilt/속도/반발 궤적 residual을 출력하고, controller가 IK로 관절 target을 만든다.

### 성공은 단순 접촉인가?

아니다. useful bounce는 접촉 후 공이 충분히 위로 올라가고, 라켓 중심에 가깝게 맞고, apex window를 만족하며, 옵션에 따라 다음 intercept가 reachable해야 한다.

### reward가 왜 복잡한가?

목표가 "한 번 맞히기"가 아니라 "계속 칠 수 있는 다음 상태 만들기"이기 때문이다. 그래서 contact, apex, next intercept, lateral stability, low-apex recovery, action penalty를 분리했다.

### 왜 epsilon 그래프가 없나?

PPO는 epsilon-greedy가 아니다. continuous Gaussian policy를 학습하므로 탐험성은 entropy/log_std/action std로 봐야 한다.

### 가장 큰 시행착오는?

초기에는 reward를 더 붙이면 해결될 것처럼 보였지만, 실제 병목은 contact/control feasibility와 action ownership이었다. 그래서 contact trace, feasibility map, heuristic gate, contact-frame primitive, residual action 확장으로 방향이 바뀌었다.

### 현재 최고 성과는?

최종 모델은 `keep1_v39_17d_mid_curriculum_fixed`다. train eval 100 episode 기준 mean useful bounce 119.52, max 181, 30+ rate 0.83이고, 7200-step long eval 20 episode 기준 mean useful bounce 130.95, max 182, 30+ rate 0.90이다.

### 최종 모델의 한계는?

기본/oldbase long eval에서는 긴 랠리가 가능하지만, coverage/stress 평가에서 reset XY, 높이, 속도를 더 넓히면 mean useful bounce가 20~60대까지 떨어진다. 즉 v39는 발표 기준 최종 성과로 충분하지만, 넓은 분포 일반화는 남은 과제다.

## 10. 발표 자료 구성 추천

1. 문제 정의  
   "로봇팔이 탁구공을 계속 올려치는 keep-up 정책 학습"

2. 왜 어려운가  
   접촉 순간이 짧고, 라켓 위치/속도/tilt/다음 공 상태가 모두 중요하다.

3. 시스템 구조  
   MuJoCo scene, sim wrapper, RL env, controller, PPO, logs.

4. MDP 정의  
   state 55D, action 17D, reward, termination.

5. 개발 흐름  
   v1 단순 EE -> v2 contact-frame -> 17D residual -> broad reset/long horizon.

6. 실험 결과  
   version metrics, long horizon, failure modes.

7. 행동/정책 분석  
   action usage, action ablation, reward/failure 분석.

8. 데모  
   MuJoCo viewer 또는 GIF.

9. 한계와 다음 작업  
   v39 coverage/stress 분석, failure case 개선, value heatmap/웹 데모.

## 11. 발표 전 할 일

1. `pingpong_rl2/docs/rl_presentation_pack`의 기존 그림을 발표 슬라이드에 넣는다.
2. 최종 결과 기준으로 발표하려면 v36~v39, 특히 `keep1_v39_17d_mid_curriculum_fixed`를 `generate_visuals.py`에 추가한다.
3. MuJoCo viewer로 v39 모델 데모 영상을 녹화한다.
4. failure case를 2~3개 골라 "왜 실패했는가"를 시각적으로 설명한다.
5. PPO epsilon 질문에 대비해 "PPO는 epsilon이 없고 entropy/log_std/action std를 본다"를 한 문장으로 준비한다.
6. v39 coverage/stress 결과를 한계 슬라이드에 넣을지 결정한다.
