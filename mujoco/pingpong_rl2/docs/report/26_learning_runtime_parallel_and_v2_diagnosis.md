# 26. Learning Runtime, Parallel PPO, and Self-Rally v2 Diagnosis

작성일: 2026-06-01

후속 구현 결과는 `27_self_rally_execution_stabilization_report.md`에 정리했다. 이 문서는 `v2` 실패 원인과 runtime/checkpoint 진단 기록이다.

## 이번 작업 기록

이번 작업에서는 `contact_frame_self_rally_candidate` 학습 실행 중 보였던 병렬 환경 문제와 checkpoint 비용을 정리했다.

- `src/pingpong_rl2/envs/gym_env.py`
  - Gymnasium vector env가 `info` dict를 합칠 때 `None` 값과 숫자 값이 섞여 터지던 문제를 막기 위해, reset/step info에서 `None` 값을 제거하도록 했다.
  - 사용자가 본 `Calling close while waiting for a pending call to step` 경고는 근본 원인이 아니라, worker step 중 예외가 난 뒤 close가 호출되면서 따라 나온 증상이다.

- `scripts/run_ppo_learning.py`
  - `contact_frame_self_rally_candidate` preset의 기본 checkpoint를 껐다.
  - 이제 이 preset은 기본적으로 중간 checkpoint 평가/저장을 하지 않고, 마지막에 `*_model.zip`과 final evaluation만 저장한다.
  - 중간 best model이 필요하면 명시적으로 `--checkpoint-interval 50000` 같은 값을 넘기면 된다.

검증한 내용:

- `n_envs=1/2/4` vector step smoke에서 `n_envs=4`가 정상 동작했다.
- 짧은 PPO smoke 학습이 `n_envs=4`로 완료됐다.
- `--checkpoint-interval 0` 최소 학습에서 checkpoint directory가 생기지 않고 `checkpoint_count=0`, `best_model_path=None`으로 기록되는 것을 확인했다.
- `py_compile` 및 주요 unittest가 통과했다.

## 병렬 학습이 어떻게 도는가

현재 `run_ppo_learning.py`는 `make_sb3_async_vector_env()`를 통해 여러 개의 `PingPongKeepUpGymEnv`를 만든다.

- `n_envs=1`: `SyncVectorEnv`
- `n_envs>=2`: `AsyncVectorEnv`
- `contact_frame_self_rally_candidate` 기본값: `n_envs=4`, `n_steps=512`, `batch_size=512`

즉 PPO 한 iteration에서 모으는 rollout 크기는 아래와 같다.

```text
rollout_size = n_envs * n_steps = 4 * 512 = 2048 transitions
```

각 env는 같은 정책으로 action을 받지만, MuJoCo 상태와 episode 진행은 독립적이다. Stable-Baselines3는 네 env에서 모은 observation/action/reward/done을 하나의 rollout buffer로 합친 뒤 PPO update를 수행한다.

MuJoCo 물리엔진은 실제로 매 step마다 돌고 있다. 다만 viewer/render를 켜지 않고, 장면이 작고, timestep이 짧으며, CPU에서 순수 수치 시뮬레이션만 하기 때문에 초당 수천 env-step이 가능하다. `fps`는 화면 프레임 수가 아니라 "학습용 환경 transition 처리량"이다. `mjpython scripts/run_viewer.py`처럼 렌더링을 켜면 훨씬 느려지는 것이 정상이다.

## Learning 로그 읽는 법

학습 중 터미널에 보이는 항목은 Stable-Baselines3 PPO 로그다.

`rollout/`

- 최근 rollout/episode 통계다.
- `ep_len_mean`: 최근 episode 평균 길이.
- `ep_rew_mean`: 최근 episode 평균 reward.
- 이 값은 성공률 자체가 아니다. 현재 task에서는 `mean_useful_bounces`, `two_or_more_rate`, `robot_body_contact_rate`, rebound analysis의 contact quality가 더 중요하다.

`time/`

- 학습 진행 속도와 누적량이다.
- `fps`: 초당 처리한 environment transition 수. `n_envs=4`면 네 env에서 나온 transition이 합쳐져 계산된다.
- `iterations`: PPO rollout/update 반복 횟수.
- `total_timesteps`: 누적 transition 수.
- `time_elapsed`: wall-clock 시간.

`train/`

- PPO가 rollout을 모은 뒤 policy/value network를 업데이트한 통계다.
- `approx_kl`: 이전 정책과 새 정책의 차이. 너무 크면 정책이 급하게 변하는 것이다.
- `clip_fraction`: PPO clipping이 걸린 비율.
- `entropy_loss`: action distribution의 탐색성 관련 값.
- `explained_variance`: value function이 return을 얼마나 설명하는지. 1에 가까울수록 좋고, 음수면 value 예측이 불안정하다.
- `policy_gradient_loss`, `value_loss`, `loss`: policy/value update 손실.
- `std`: Gaussian action policy의 표준편차. residual action 정책에서는 너무 커지면 불필요한 흔들림이 생길 수 있다.

## Checkpoint 판단

checkpoint는 세 가지 목적으로만 필요하다.

- 긴 학습 중간 결과로 되돌아가기
- 중간 checkpoint 평가로 best model을 고르기
- 학습 붕괴 시 어느 시점부터 망가졌는지 추적하기

하지만 self-rally 실험에서는 checkpoint 평가가 학습 시간을 잡아먹고 파일도 많이 만든다. 사용자가 실행한 `pmk_cf_self_rally_v2`는 checkpoint 41개를 만들었고, run 전체 약 12MB 중 checkpoint zip만 약 9MB였다.

이번 정리에서 `pmk_cf_self_rally_v2/checkpoints` 디렉토리는 삭제했다. `pmk_cf_self_rally_v2_model.zip`, `pmk_cf_self_rally_v2_best_model.zip`, training summary, rebound analysis 결과는 남겼다.

따라서 현재 기본 방침은 아래와 같다.

- 기본 self-rally 학습: checkpoint 끔
- 발표/시연용: `*_model.zip`만으로 viewer 실행
- 중간 best 모델 비교가 꼭 필요할 때만 `--checkpoint-interval`을 명시

주의: checkpoint를 끄면 `*_best_model.zip`은 자동 생성되지 않는다. 새 기본 실행에서는 마지막 모델인 `*_model.zip`을 기준으로 평가한다.

## pmk_cf_self_rally_v2 상태

사용자가 2M으로 학습한 최신 모델은 다음 run이다.

```text
artifacts/ppo_runs/pmk_cf_self_rally_v2
```

학습 요약:

- completed_timesteps: 2,000,000
- checkpoint_count: 41
- best checkpoint: 350,000 steps
- final evaluation mean_useful_bounces: 0.25
- final evaluation max_useful_bounces: 2
- final evaluation two_or_more_rate: 0.02
- final evaluation failure:
  - ball_out_of_bounds: 0.61
  - robot_body_contact: 0.25
  - ball_speed_limit: 0.09
  - floor_contact: 0.05

`pmk_cf_self_rally_v2_best_model.zip`에 대한 100 episode rebound analysis:

- mean_useful_bounces: 0.29
- max_useful_bounces: 2
- one_or_more_useful_bounce_rate: 0.21
- two_or_more_useful_bounce_rate: 0.08
- failure_counts:
  - ball_out_of_bounds: 69
  - robot_body_contact: 13
  - ball_speed_limit: 6
  - floor_contact: 12
- total_contacts: 589
- useful_contact_rate: 0.049
- all-contact mean_next_intercept_xy_error: 0.084m
- all-contact next_intercept_reachable_rate: 0.409
- all-contact mean_outgoing_velocity_z_error: 0.792
- useful-contact mean_next_intercept_xy_error: 0.020m
- useful-contact mean_ball_lateral_speed: 0.030m/s
- useful-contact next_intercept_reachable_rate: 1.0
- useful-contact mean_easy_next_ball_score: 1.105

## 현재 문제가 무엇인가

`v2`는 reward만 조금 부족한 상태가 아니다. 좋은 contact가 드물게 나오면 그 contact의 품질은 괜찮다. 문제는 대부분의 contact가 그 상태까지 가지 못한다는 것이다.

핵심 병목:

- useful contact rate가 4.9%로 너무 낮다.
- 전체 contact의 다음 intercept reachable rate가 40.9%에 그친다.
- robot body contact가 많다. 특히 rebound analysis에서 `link5`가 12회, `link6`이 1회였다.
- strict success contract 때문에 낮거나 먼 공은 useful로 인정되지 않는다. 사용자가 viewer에서 보는 "어거지로 맞음"은 대부분 성공으로 세지 않는 contact일 가능성이 크다.
- pitch/roll/tilt는 이미 primitive에 들어가 있지만, 라켓이 타격 위치/시간에 제대로 도착하지 못하면 tilt가 해결책이 되지 않는다.

따라서 지금의 실패는 "공을 어느 방향으로 보상할까"만의 문제가 아니라, 아래 두 계약이 부족한 문제다.

1. 공이 내려올 때 라켓이 먼저 안전한 타격 자세로 들어가야 한다.
2. 타격 순간에는 라켓 중심/속도/자세가 안정되어야 하고, 팔 몸체가 공 경로를 막으면 안 된다.

## 왜 안정적인 공 띄우기가 아직 안 되는가

사용자가 지적한 세 현상은 모두 같은 구조적 원인으로 묶인다.

1. 공을 너무 낮게 띄우는 문제

- strict apex window와 target apex 보상은 들어갔지만, 실제 contact 순간의 라켓 속도/normal이 안정적으로 맞지 않으면 목표 apex까지 못 올린다.
- useful contact의 z velocity error는 낮지만, 전체 contact의 z velocity error는 매우 크다. 즉 "잘 맞은 일부"가 아니라 "대부분의 타격 품질"을 끌어올려야 한다.

2. 가까운 공에 팔을 빨리 오므리지 못하는 문제

- 현재 self-rally preset의 body clearance/nullspace 보정은 꺼진 상태다.
- `v2` env config에서 `controller_body_clearance_gain=0.0`, `controller_nullspace_posture_gain=0.0`이었다.
- 그래서 공이 로봇팔 안쪽으로 들어오면 라켓만 맞추려다가 link5/link6가 공을 막는 실패가 생긴다.

3. 먼 공을 tilt로 회복하지 못하는 문제

- tilt action과 trajectory/centering tilt는 존재한다.
- 하지만 tilt는 "라켓이 이미 contact 위치에 있고, normal/속도 제어가 맞는 상태"에서 lateral correction을 만든다.
- 현재는 contact timing과 arm clearance가 먼저 깨지므로, tilt를 더 키우면 회복보다 chatter/불안정 접촉이 늘 가능성이 높다.

## 다음 작업 방향

다음 단계는 값 sweep이 아니라 구조 보강이어야 한다.

1. self-rally preset에 body-clearance/nullspace recovery를 다시 넣는다.

- `controller_body_clearance_gain`
- `controller_body_clearance_margin`
- `controller_body_clearance_max_step`
- `controller_body_clearance_body_names=("link5", "link6")`
- `controller_nullspace_posture_gain`
- `controller_nullspace_posture_max_step`

목표는 공이 팔 안쪽으로 들어올 때 link5/link6가 공 경로를 막지 않고, 라켓만 contact plane에 남도록 하는 것이다.

2. strike window를 안정화한다.

- 타격 직전 짧은 구간에서는 target XY를 계속 크게 chase하지 말고, contact point 주변에 hold/stabilize해야 한다.
- contact 순간의 racket lateral velocity, target jump, contact_xy_error를 분석/penalty에 더 직접적으로 넣어야 한다.
- "움직이면서 우연히 맞는 접촉"은 성공으로 보지 않는 쪽이 맞다.

3. recovery phase를 별도 계약으로 만든다.

- 공이 상승 중일 때는 휘적거리지 않고 다음 intercept를 준비한다.
- 공이 하강으로 바뀌면 빠르게 intercept 자세로 들어간다.
- close-ball과 far-ball에서 다른 recovery target을 쓰는 것이 좋다.

4. tilt는 마지막 보정으로 유지한다.

- pitch/roll은 필요하다.
- 다만 tilt 자유도를 먼저 키우는 순서가 아니라, intercept/clearance/strike-window가 안정된 뒤 lateral return correction으로 써야 한다.

5. 다음 실험은 checkpoint 없이 돌린다.

권장 명령:

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v3 \
  --reset-model \
  --total-timesteps 2000000
```

단, 위 명령은 checkpoint가 꺼진 기본 preset 기준이다. 중간 best model을 남기고 싶을 때만 아래 옵션을 추가한다.

```bash
--checkpoint-interval 50000 --checkpoint-eval-episodes 20
```

## 결론

`v2`는 "다음 공을 치기 쉬운 위치로 보내는 reward"만으로는 부족하다는 것을 보여준다. 지금 병목은 공의 목표점보다 한 단계 아래인 physical execution이다. 라켓이 제때 안전한 자세로 들어가지 못하고, 팔 몸체가 공 경로를 막고, 타격 순간이 안정되지 않는다.

따라서 다음 구현은 reward 숫자 조절보다 `body clearance + recovery posture + strike-window stabilization`을 우선해야 한다. 그 뒤에 tilt를 lateral return 보정으로 키우는 것이 목표 self-rally에 맞는 순서다.
