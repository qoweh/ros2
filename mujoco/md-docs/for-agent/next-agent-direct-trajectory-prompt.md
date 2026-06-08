# 다음 에이전트 프롬프트: 직접 궤적 목표로 keep-up 문제 다시 잡기

너는 `pingpong_rl2` 프로젝트를 이어받는 에이전트다.

작업 루트:

```bash
/Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
```

## 1. 최종 목표

Franka Panda 로봇팔이 탁구채로 탁구공을 계속 위로 튕기는 강화학습을 완성한다.

지금 문제가 계속 반복된다.

- 공을 맞히긴 한다.
- 위로도 어느 정도 친다.
- 하지만 공이 로봇팔/라켓이 다시 칠 수 있는 위치가 아니라 먼 방향으로 간다.
- 그래서 `2+ useful bounce`가 거의 유지되지 않는다.

따라서 이번 작업의 핵심은 더 이상 `assist weight`, `vx penalty`, `bootstrap filter`를 조금씩 바꾸는 것이 아니다.

이번 작업의 핵심:

> contact 순간 공이 따라야 할 “목표 outgoing trajectory”를 직접 정의하고, policy가 그 궤적을 만들도록 reward/control/diagnostic을 다시 잡는다.

## 2. 먼저 읽을 파일

반드시 읽어라.

- `agent-answer1.md`
- `agent-answer2.md`
- `agent-answer3.md`
- `docs/report/12_followup_strike_bootstrap_report.md`
- `docs/report/08_easy_next_ball_completion_plan.md`
- `src/pingpong_rl2/envs/keepup_env.py`
- `src/pingpong_rl2/envs/pingpong_sim.py`
- `scripts/run_ppo_learning.py`
- `scripts/run_ppo_rebound_analysis.py`
- `scripts/run_heuristic_keepup_diagnostic.py`

## 3. 현재 방향에 대한 판단

최근 작업은 너무 우회적이다.

시도한 것들:

- post-contact return assist weight 조정
- easy-next-ball metric
- bootstrap filtering
- staged resume
- follow-up contract

이 작업들은 일부 지표를 개선했지만, 최종 문제를 직접 닫지 못했다.

근본 원인:

> “다음에 다시 칠 수 있는 공”을 contact 직후의 물리적 목표 궤적으로 직접 정의하지 않았다.

현재 reward는 대체로 아래처럼 되어 있다.

- 공 아래로 가라
- 위로 쳐라
- apex height를 맞춰라
- 실패하지 마라

하지만 빠진 것은 이것이다.

- 맞은 직후 공의 수평/수직 속도가 다음 strike zone으로 돌아오는 궤적인가?

`x velocity`만 줄이는 것도 부족하다.  
`projected apex xy`만 보는 것도 부족하다.  
`assist target`만 바꾸는 것도 부족하다.

contact 순간에 원하는 공의 다음 궤적을 계산하고, 실제 outgoing velocity가 그와 맞는지 봐야 한다.

## 4. 이번 작업에서 하지 말 것

- assist weight만 더 찍어보지 마라.
- bootstrap filter를 더 복잡하게 만들지 마라.
- `vx penalty`를 재튜닝하지 마라.
- 새 reward를 여러 개 동시에 추가하지 마라.
- 바로 1M 학습하지 마라.
- SAC로 바로 바꾸지 마라.

## 5. 반드시 해야 할 핵심 작업

### Step 1. contact 직후 목표 궤적을 정의하라

공이 라켓에 맞은 직후, 다음 공이 어디로 가야 하는지 계산해야 한다.

목표는 robot base가 아니다.

목표는:

```text
target_xy = racket home / controller anchor / nominal strike zone center
target_z_apex = racket_z + desired_keepup_height
```

contact 순간 공 위치를 `p_contact`, 목표 apex를 `target_apex`라고 하자.

중력만 고려하면 desired outgoing velocity는 대략 이렇게 계산할 수 있다.

```python
gravity = abs(model.opt.gravity[2])
target_apex_z = anchor_z + target_ball_height
height_delta = max(target_apex_z - contact_ball_z, min_height_delta)
desired_vz = sqrt(2.0 * gravity * height_delta)
time_to_apex = desired_vz / gravity
desired_vxy = (target_xy - contact_ball_xy) / max(time_to_apex, 1e-6)
desired_velocity = [desired_vx, desired_vy, desired_vz]
```

이게 이 프로젝트의 missing objective다.

의도:

- 공을 위로 보낸다.
- 공의 apex 또는 다음 낙하 궤적이 strike zone 근처가 된다.
- 너무 먼 쪽으로 튀는 공은 자동으로 벌어진다.
- `x` 하나가 아니라 `vx, vy, vz` 전체가 목표와 맞는다.

### Step 2. 먼저 analysis metric으로 넣어라

바로 reward에 넣지 말고, `run_ppo_rebound_analysis.py`와 env info에 metric을 추가해라.

contact event마다 기록:

- `desired_outgoing_velocity_x/y/z`
- `actual_outgoing_velocity_x/y/z`
- `outgoing_velocity_error_norm`
- `outgoing_velocity_xy_error`
- `outgoing_velocity_z_error`
- `desired_time_to_apex`
- `desired_target_xy`
- `predicted_apex_xy_from_actual_velocity`
- `predicted_apex_xy_error`

그리고 summary에 아래를 추가해라.

- all contact mean outgoing velocity error
- useful contact mean outgoing velocity error
- 2+ episode contact mean outgoing velocity error
- zero-bounce episode contact mean outgoing velocity error

목표:

> 실제로 2+ bounce episode가 desired outgoing velocity error가 낮은지 확인한다.

이 상관이 없으면 target 정의가 틀린 것이다.

### Step 3. scripted controller로 물리 가능성 확인

PPO 전에 먼저 진단해야 한다.

질문:

> 이 MuJoCo 환경과 라켓 물리에서, scripted controller가 desired outgoing trajectory를 만들 수 있는가?

필요하면 `run_heuristic_keepup_diagnostic.py` 또는 새 script에서 실험해라.

scripted controller는 완벽할 필요 없다.

목표:

- 3회 이상 keep-up이 가능한지
- 공을 strike zone 쪽으로 다시 보낼 수 있는지
- 라켓 pitch/normal sign이 맞는지
- contact 후 velocity가 desired velocity와 가까워지는지

만약 scripted controller도 못 하면, PPO 문제가 아니라 아래 중 하나다.

- 라켓 orientation/sign 문제
- MuJoCo contact/friction/restitution 문제
- controller가 충분한 racket velocity를 못 만듦
- action/control surface가 원하는 contact를 만들기 어려움

이 경우 reward 튜닝을 중단하고 물리/제어를 먼저 고쳐라.

### Step 4. reward로 승격

analysis에서 desired outgoing velocity error가 실제 keep-up 성공과 맞으면, 그때만 reward로 넣어라.

추천 reward:

```python
if contact_event and contact_ball_velocity_z > 0:
    velocity_error = norm((actual_v - desired_v) / scale)
    reward_terms["trajectory_match_term"] = weight * exp(-velocity_error)
```

또는 penalty:

```python
reward_terms["trajectory_match_term"] = -weight * velocity_error
```

초기값:

- `weight=0.5` 이하로 시작
- contact event에서만 적용
- useful contact 조건에 묶지 말고, upward racket contact면 적용

중요:

- 기존 `outgoing_x_term`은 끄거나 제거 후보로 둔다.
- `return_target_xy_term`과 동시에 켜지 마라.
- 새 reward는 한 번에 하나만 비교한다.

### Step 5. observation 보강

policy가 desired trajectory를 만들려면 아래 정보가 observation에 필요할 수 있다.

후보:

- racket face normal
- racket velocity
- relative velocity: `ball_velocity - racket_velocity`
- phase one-hot
- next intercept target xy/time
- desired outgoing velocity
- time since last contact 또는 bounce count clipped

바로 다 넣지 말고, 최소 후보부터:

1. racket velocity
2. racket face normal
3. relative velocity
4. phase one-hot
5. desired outgoing velocity

특히 contact dynamics를 다루려면 `racket_face_normal`과 `relative_velocity`는 매우 중요하다.

## 6. 기존 결과를 어떻게 봐야 하나

현재까지 가장 나은 방향:

- `followup_strike_candidate` control contract
- plain bootstrap으로 first-bounce acquisition 확보
- best checkpoint에서 contract 아래 resume PPO

하지만 이건 아직 근본 해결이 아니다.

왜냐하면:

- `mean_useful_bounces`가 아직 낮다.
- `two+ rate`가 낮다.
- 공이 왜 먼 방향으로 튀는지 contact physics 목표가 직접 정의되지 않았다.

따라서 이번 작업의 결론은 이래야 한다.

> staged training은 유지할 수 있지만, 이제 학습 목표를 contact trajectory matching으로 바꿔야 한다.

## 7. 실험 순서

### 7.1 metric-only

먼저 기존 best 모델들에 대해 새 metric을 계산한다.

대상:

- `followup_strike_contract_v1_best_model.zip`
- `followup_strike_bootstrap_v1_best_model.zip`
- `followup_bootstrap_resume_contract_v1_best_model.zip`
- 가능하면 `clean_tnp_return_assist_v1_best_model.zip`

분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path <best_model_zip> \
  --episodes 50 \
  --analysis-name <name> \
  --compare-apex-targets
```

새 metric이 추가된 뒤 summary를 비교한다.

### 7.2 scripted diagnostic

desired outgoing trajectory를 따르는 heuristic/scripted controller를 확인한다.

목표:

- 최소 2+ useful bounce가 안정적으로 나오는지
- 나오지 않는다면 contact/physics/control 문제를 먼저 찾기

### 7.3 reward 실험

metric과 scripted 가능성이 확인된 뒤에만 PPO reward 실험을 한다.

첫 실험:

```bash
python scripts/run_ppo_learning.py \
  --preset followup_strike_candidate \
  --run-name trajectory_match \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000 \
  --seed 7 \
  --trajectory-match-weight 0.3
```

옵션 이름은 예시다. 실제 구현에 맞게 정하라.

비교:

- 같은 preset, 같은 seed, reward만 다르게
- 50-episode rebound analysis 필수

### 7.4 staged schedule 재사용

trajectory reward가 promising하면, 그 다음에만 staged resume에 붙인다.

```bash
python scripts/run_ppo_learning.py \
  --preset followup_strike_candidate \
  --resume-from artifacts/ppo_runs/followup_strike_bootstrap_v1/followup_strike_bootstrap_v1_best_model.zip \
  --run-name trajectory_match_resume \
  --run-version v1 \
  --total-timesteps 50000 \
  --trajectory-match-weight 0.3
```

## 8. 성공 기준

성공은 아래 중 다수를 만족해야 한다.

- `mean_useful_bounces` 증가
- `two_or_more_useful_bounce_rate` 증가
- `ball_out_of_bounds` 감소
- useful contact의 `outgoing_velocity_error_norm` 감소
- 2+ episode의 `outgoing_velocity_error_norm`이 zero-bounce episode보다 낮음
- viewer에서 공이 먼 쪽이 아니라 strike zone 쪽으로 돌아오는 것이 보임

실패:

- contact count만 증가
- first bounce만 증가하고 two+는 그대로
- desired velocity error는 줄지만 useful bounce가 줄어듦
- 공이 로봇 몸통 쪽으로 가서 body contact 증가

## 9. 산출물

작업 후 아래 문서를 만들어라.

```text
docs/report/13_direct_trajectory_objective_report.md
```

반드시 포함:

1. 왜 기존 weight/assist/bootstrap 미세조정이 부족한지
2. desired outgoing trajectory 정의
3. 기존 best 모델들의 trajectory error 비교
4. scripted controller가 가능한지 여부
5. reward 승격 여부
6. PPO 실험 결과
7. 다음에 계속할지/버릴지 판단

## 10. 핵심 한 문장

지금 문제는 “공을 로봇 쪽으로 보내라”가 아니다.

정확한 문제는:

> 라켓 contact 순간, 공이 다음 strike zone으로 돌아오는 물리적으로 올바른 outgoing velocity를 만들도록 학습시키는 것.

이걸 직접 목표로 삼지 않으면, 계속 보조 metric과 assist weight만 만지게 된다.
