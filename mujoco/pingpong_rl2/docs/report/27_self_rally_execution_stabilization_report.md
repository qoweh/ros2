# 27. Self-Rally Execution Stabilization Report

작성일: 2026-06-01

## 목적

이번 작업은 reward 숫자 조정이 아니라, 사용자가 반복해서 지적한 실제 실패 모드를 직접 줄이기 위한 구조 보강이다.

문제:

1. 공을 너무 낮게 띄운다.
2. 공이 로봇팔 가까이 들어올 때 팔을 충분히 접거나 비키지 못해 라켓을 제 위치에 두지 못한다.
3. 공이 로봇팔에서 멀어질 때 pitch/roll/tilt로 회복해야 하는데, 실제로는 타격 자세가 안정되기 전에 실패한다.
4. 라켓이 타격 지점에 먼저 도착해서 치는 것이 아니라, 움직이는 중에 우연히 공이 맞는 접촉이 생긴다.

`pmk_cf_self_rally_v2` 분석에서 좋은 contact가 드물게 나오면 next-intercept 품질은 괜찮았다. 따라서 이번 작업은 "다음 목표 XY를 reward로 더 강하게 준다"보다 아래 실행 안정화에 집중했다.

## 구현한 보강

### 1. Body clearance/nullspace를 self-rally 기본 preset에 포함

`v2`의 env config에서는 아래 값들이 꺼져 있었다.

- `controller_body_clearance_gain=0.0`
- `controller_nullspace_posture_gain=0.0`

그래서 공이 팔 안쪽으로 들어오면 라켓을 맞추려다가 `link5`, `link6`가 공을 막는 실패가 많았다.

새 `contact_frame_self_rally_candidate` 기본값:

- `controller_nullspace_posture_gain=0.20`
- `controller_nullspace_posture_max_step=0.010`
- `controller_body_clearance_gain=0.75`
- `controller_body_clearance_margin=0.14`
- `controller_body_clearance_vertical_margin=0.32`
- `controller_body_clearance_max_step=0.018`
- `controller_body_clearance_body_names=("link5", "link6")`

그리고 clearance 활성 조건도 완화했다. 이전에는 상승 중 clearance가 `successful_bounce_count > 0`일 때만 켜졌기 때문에, strict success로 인정되지 않은 접촉 이후에는 팔 회피가 꺼졌다. 이제는 최근 contact가 있고 공이 아직 가까운 높이에 있으면 useful 여부와 무관하게 clearance가 켜진다.

의도:

- 가까운 공에서 link5/link6가 공 경로를 막는 일을 줄인다.
- 라켓 target은 contact에 남기고, 팔 몸체는 nullspace로 비키게 한다.

### 2. Strike hold window 추가

기존 contact-frame planner는 하강 중 target XY/apex는 고정했지만, contact position은 매 step 새로 계산했다. 이 때문에 타격 직전에도 라켓 target이 계속 움직이고, "도착해서 치기"보다 "추적하다가 맞기"가 생길 수 있었다.

새 옵션:

- `contact_frame_strike_hold_time`
- `contact_frame_strike_hold_min_readiness`

동작:

- 공이 하강 중이고 intercept time이 hold window 안으로 들어온다.
- 라켓이 충분히 준비된 상태면 그 순간의 planned contact position을 고정한다.
- 이후 contact 전까지는 target contact position을 계속 따라 바꾸지 않는다.
- hold가 켜지면 lateral intercept velocity도 0으로 눌러서, 타격 직전에는 sideways chase보다 안정된 contact를 우선한다.

새 self-rally 기본값:

- `contact_frame_strike_hold_time=0.05`
- `contact_frame_strike_hold_min_readiness=0.60`

의도:

- 타격 순간의 라켓 중심/자세를 안정화한다.
- 움직이면서 공을 맞히는 접촉을 줄인다.

### 3. 낮은 타격 보정: 목표 높이를 바꾸지 않고 velocity primitive를 더 쓰게 함

`target_ball_height=0.25`는 이론적으로 약 0.45초 정도의 다음 하강 시간을 만들 수 있는 높이라, 우선 목표 높이 자체를 바꾸지 않았다. 문제는 목표가 낮아서가 아니라, 실제 contact 순간의 라켓 속도/normal이 목표 outgoing velocity를 충분히 만들지 못하는 쪽으로 봤다.

새 self-rally 기본값:

- `contact_frame_velocity_target_gain=0.65`
- `contact_frame_velocity_target_max=1.6`
- `controller_velocity_feedback_gain=0.25`
- `controller_max_velocity_step=0.03`
- `contact_frame_apex_lift_gain=0.05`
- `contact_frame_apex_lift_max=0.025`
- `contact_frame_velocity_lead_gain=0.04`
- `contact_frame_velocity_lead_max=0.025`

의도:

- planner가 계산한 required racket velocity를 더 실제 제어에 반영한다.
- 너무 낮게 튀는 contact를 줄인다.
- RL이 vertical residual을 처음부터 어렵게 배우지 않게 한다.

### 4. Tilt는 유지하되 "마지막 보정"으로 확장

사용자가 말한 것처럼 pitch/roll/tilt는 필요하다. 다만 `v2`의 문제는 tilt가 없어서라기보다, tilt가 작동할 접촉 상태까지 못 가는 것이었다.

이번 변경:

- `target_tilt_limit=(0.09, 0.09)`
- `contact_frame_trajectory_tilt_limit=(0.05, 0.05)`
- `contact_frame_centering_tilt_limit=(0.035, 0.045)`

의도:

- far-ball에서 다음 intercept를 anchor 쪽으로 돌리는 lateral correction headroom을 조금 늘린다.
- 하지만 tilt를 먼저 과하게 키워 chatter가 생기지 않도록, strike hold와 lateral sweep penalty를 함께 둔다.

### 5. Side-sweep contact를 useful로 세지 않도록 제한

움직이면서 맞는 contact가 성공으로 들어가면 RL이 잘못된 행동을 배운다.

새 옵션:

- `contact_racket_lateral_velocity_penalty_weight`
- `contact_racket_lateral_velocity_tolerance`
- `max_contact_racket_lateral_speed_for_success`

새 self-rally 기본값:

- `contact_racket_lateral_velocity_penalty_weight=0.25`
- `contact_racket_lateral_velocity_tolerance=0.18`
- `max_contact_racket_lateral_speed_for_success=0.45`

의도:

- 라켓 lateral sweep으로 우연히 맞은 contact를 덜 보상한다.
- useful contact는 "라켓 중심이 들어와 있고, 위로 쳤고, 다음 공이 쉬운 위치로 오며, 라켓이 과하게 옆으로 쓸지 않은" 경우로 더 좁힌다.

## 수정 파일

- `src/pingpong_rl2/envs/keepup_env.py`
  - strike hold state/logic 추가
  - body clearance 활성 조건 완화
  - lateral racket speed success gate/penalty 추가
  - 관련 info/training_config 진단값 추가

- `scripts/run_ppo_learning.py`
  - `contact_frame_self_rally_candidate` preset을 execution-stabilized 설정으로 갱신
  - 새 옵션을 CLI/env_kwargs/preset managed defaults에 연결

- `scripts/run_ppo_rebound_analysis.py`
  - contact CSV에 `racket_lateral_speed`, `contact_frame_strike_hold_active`, `controller_body_clearance_active` 기록 추가

- `tests/test_keepup_env.py`
  - strike hold contact position freeze
  - hold 중 lateral intercept velocity 억제
  - recent contact 후 body clearance 활성
  - side-sweep contact success rejection
  - lateral racket velocity penalty 검증

## 검증

통과한 검증:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  scripts/run_ppo_learning.py \
  scripts/run_ppo_rebound_analysis.py
```

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest \
  tests/test_vector_env.py \
  tests/test_ppo_runs.py \
  tests/test_keepup_env.py \
  tests/test_keepup_contract_features.py
```

결과:

- `97 tests OK`

짧은 zero-action rollout 확인:

- self-rally preset에 새 설정이 실제 적용됨
- episode 중 `contact_frame_strike_hold_active=True`가 관측됨
- episode 중 `controller_body_clearance_active=True`가 관측됨
- primitive target velocity z가 양수로 올라옴

zero-action rollout은 정책 성능 평가가 아니라, 새 구조가 런타임에서 실제로 켜지는지 확인한 smoke check다.

짧은 PPO 실행 확인:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name debug_self_rally_v3 \
  --run-version smoke \
  --output-dir artifacts/debug_self_rally_v3_smoke \
  --reset-model \
  --total-timesteps 2048
```

결과:

- `completed_timesteps=2048`
- `checkpoint_count=0`
- `best_model_path=None`
- preset/env config에 `body_clearance_gain=0.75`, `strike_hold_time=0.05`, `max_contact_racket_lateral_speed_for_success=0.45`가 반영됨
- 디버그 산출물은 확인 후 삭제함

## 다음 학습 명령

이제 이전 `pmk_cf_self_rally_v2`를 이어서 학습하지 말고, 새 구조로 fresh run을 시작한다.

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v3 \
  --reset-model \
  --total-timesteps 2000000
```

checkpoint는 기본적으로 꺼져 있으므로, 학습 후 viewer는 final model을 본다.

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v3/pmk_cf_self_rally_v3_model.zip \
  --episodes 100
```

분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v3/pmk_cf_self_rally_v3_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v3_quality100
```

## 판단 기준

`v3`가 올바른 방향인지 볼 때는 아래를 먼저 본다.

- `robot_body_contact_rate`가 `v2`보다 줄어야 한다.
- `useful_contact_rate`가 `v2`의 0.049보다 올라야 한다.
- `all-contact next_intercept_reachable_rate`가 `v2`의 0.409보다 올라야 한다.
- `racket_lateral_speed`가 큰 contact가 useful로 많이 잡히면 안 된다.
- `useful_contact_mean_next_intercept_xy_error`는 계속 0.02m 근처를 유지해야 한다.
- viewer에서 공이 낮게 맞고 바로 짧은 주기로 떨어지는 패턴이 줄어야 한다.

만약 `v3`에서도 useful contact rate가 오르지 않으면, 다음 문제는 reward가 아니라 IK/controller 속도 한계 또는 contact-frame target z/strike plane 자체일 가능성이 크다. 그 경우에는 관절 속도/토크 제한, racket site 위치, paddle collision geometry를 보는 단계로 넘어가야 한다.
