# 24. Self-Rally Planner/Primitive Implementation Report

작성일: 2026-06-01

## 목표 재정의

이 프로젝트의 목표는 상대 코트로 공을 보내는 탁구가 아니라, 로봇팔에 붙인 탁구채로 공을 계속 위로 튕기는 self-rally/keep-up이다. 따라서 정책이 배워야 하는 핵심은 "이번 타격 후 공이 다음에도 라켓 근처, 적절한 높이, 적절한 시간 뒤에 다시 내려오게 만드는 것"이다.

기존 방향은 접촉 보상, 위쪽 속도 보상, 여러 보조 옵션을 조금씩 더하는 방식에 가까웠다. 이 방식은 공을 맞히고 위로 보내는 행동은 강화하지만, 공이 XY 평면에서 라켓/로봇팔 중심에서 점점 멀어지는 타격을 충분히 금지하지 못한다.

## 연구 보고서에서 참고한 점

- DeepMind/Google의 `Achieving Human Level Competitive Robot Table Tennis`는 단일 end-to-end 정책 하나에 모든 것을 맡기기보다, 저수준 skill/controller와 고수준 선택기를 나누는 계층적/모듈식 구조를 사용한다. 이번 작업에서는 이를 keep-up 문제에 맞게 축소해서, planner/primitive가 기본 타격 목표를 만들고 RL은 residual만 고치도록 했다.  
  참고: https://arxiv.org/abs/2408.03906

- 같은 논문은 low-level skill descriptor로 에이전트가 어떤 공을 어떤 방식으로 처리할 수 있는지 표현한다. 이 프로젝트에서는 descriptor를 `next_intercept_xy`, `target_apex_z`, `desired_outgoing_velocity`, `required_racket_velocity`, `racket_tilt`로 바꿔 적용했다.

- `Sample-efficient Reinforcement Learning in Robotic Table Tennis`는 타격 시점의 공 상태와 라켓의 자세/속도를 action 또는 skill 변수로 두는 접근을 사용한다. 이번 구현은 관절 raw action을 바로 학습시키지 않고, 타격 시점의 contact-frame position/velocity/tilt primitive 위에 residual action을 얹는다.  
  참고: https://arxiv.org/abs/2011.03275

- 로봇 탁구 시스템 연구들은 공 예측, 기준 궤적 생성, 빠른 추종 제어를 분리하는 경향이 있다. 이번 구현도 ballistic intercept 예측, self-rally 목표 생성, controller 추종, PPO residual을 분리했다.  
  참고: https://sites.google.com/view/robotictabletennis/

## 이번 구현 방향

1. Planner를 고정했다.

- `contact_frame_planner_enabled` 옵션을 추가했다.
- 공이 내려오는 동안 다음 목표 XY는 `keepup_target_xy`로 고정한다.
- 목표 apex는 `controller_anchor_z + target_ball_height + contact_frame_planner_target_apex_z_offset`로 고정한다.
- 접촉 위치와 시간은 현재 공 상태에서 계속 재계산하지만, "다음 공을 어디로 보낼지"는 하강 구간 동안 흔들리지 않게 했다.

2. Primitive가 기본 타격을 계산한다.

- planner가 만든 contact position/target apex/target XY에서 `desired_outgoing_velocity`를 계산한다.
- 라켓 face normal 기준으로 필요한 racket velocity를 계산한다.
- trajectory tilt와 centering tilt가 planner의 contact point와 desired outgoing velocity를 기준으로 동작한다.
- contact-frame target position도 planner contact position을 기본값으로 사용한다.

3. RL은 residual만 결정한다.

- 새 preset `contact_frame_self_rally_candidate`를 추가했다.
- 이 preset은 `position_contact_frame` action을 작은 residual로 제한한다.
- 기본값:
  - `lateral_action_limit=0.02`
  - `vertical_action_limit=0.025`
  - `tilt_action_limit=0.01`
- 즉 PPO는 "어디로 크게 칠지"를 처음부터 다시 배우는 것이 아니라, primitive의 기본 타격을 조금 보정한다.

4. Reward/Success를 엄격하게 했다.

- useful contact는 다음 descending intercept가 reachable zone 안에 있어야 성공으로 인정한다.
- bad hit에 대한 dense penalty를 추가했다.
  - `next_intercept_xy_error_penalty_weight`
  - `post_contact_lateral_velocity_penalty_weight`
  - `contact_xy_error_penalty_weight`
  - `nonuseful_contact_penalty_weight`
- 이제 공이 위로 뜨더라도 다음에 치기 어려운 위치로 가면 성공으로 세지 않고, 보상도 깎인다.

## 주요 수정 파일

- `src/pingpong_rl2/envs/keepup_env.py`
  - self-rally planner 상태/업데이트 추가
  - planner 기반 desired outgoing velocity, required racket velocity, contact-frame target/tilt 연결
  - strict contact quality penalty 추가
  - training config/info에 planner 진단값 추가

- `scripts/run_ppo_learning.py`
  - `contact_frame_self_rally_candidate` preset 추가
  - planner/reward penalty CLI 옵션 추가

- `scripts/run_heuristic_keepup_diagnostic.py`
  - planner/reward penalty 옵션과 CSV 진단값 추가

- `scripts/run_ppo_rebound_analysis.py`
  - planner/reward penalty override와 contact CSV 진단값 추가

- `scripts/run_viewer.py`
  - planner/reward penalty override 추가

- `tests/test_keepup_env.py`
  - planner target 고정
  - planner contact target 사용
  - bad upward contact penalty 검증 테스트 추가

## 권장 학습 명령

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v1 \
  --reset-model \
  --total-timesteps 1000000
```

학습 후에는 아래처럼 확인한다.

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v1/pmk_cf_self_rally_v1_best_model.zip \
  --episodes 100
```

그리고 rebound analysis에서 반드시 봐야 할 지표는 다음이다.

- `mean_useful_bounces`
- `two_or_more_rate`
- `three_or_more_rate`
- `mean_next_intercept_xy_error`
- `useful_contact_mean_next_intercept_xy_error`
- `next_intercept_reachable_rate`
- `mean_easy_next_ball_score`
- `mean_outgoing_velocity_error_norm`

## 판단 기준

이 작업이 올바른 방향인지 판단하는 기준은 reward가 아니라 다음 계약이다.

- 공이 위로 떠도 다음 intercept XY가 라켓 anchor에서 멀면 실패에 가깝다.
- useful contact는 `next_intercept_xy_error <= 0.04m` 근처로 유지되어야 한다.
- lateral outgoing velocity가 줄어야 한다.
- target apex가 너무 낮아서 다음 타격 주기가 과도하게 짧아지면 실패로 본다.
- 라켓 pitch/roll은 큰 탐색 변수가 아니라 planner velocity를 맞추는 작은 primitive/residual로 써야 한다.

## 현재 상태

구조 변경과 테스트는 완료했다. 다만 이 문서의 구현은 "학습이 가능하도록 문제 정의를 바꾼 것"이지, 이미 긴 학습으로 최종 모델이 완성됐다는 뜻은 아니다. 다음 단계는 `contact_frame_self_rally_candidate`로 새로 학습하고, 이전 `pmk_cf_zero_init_eval_v2`, `pmk_cf_conservative_v1`과 rebound 지표를 비교하는 것이다.
