# 다음 에이전트 작업 지시사항

## 0. 상황 요약

이 프로젝트의 최종 목표는 단순히 공을 한 번 맞히는 것이 아니다.

목표:

> Franka Panda 로봇팔이 탁구채로 탁구공을 계속 위로 튕기고, 공이 다시 칠 수 있는 strike zone으로 돌아오게 하는 강화학습 환경/학습 설계를 완성한다.

최근 작업은 방향이 뒤섞였다. 하나를 고치면 다른 문제가 생기고, reward/policy/CLI 옵션이 계속 늘어나면서 “최종 목표에 가까워지는지”보다 “개별 실험 하나가 좋아졌는지”만 보게 됐다.

이번 작업의 핵심은 새 reward를 하나 더 붙이는 것이 아니다.

이번 작업의 핵심:

1. 현재 reward, action mode, assist, observation, CLI 옵션을 정리한다.
2. 기본값과 실험용 옵션을 분리한다.
3. 최종 목표 기준으로 reward/policy 설계를 다시 단순화한다.
4. 그 설계를 검증 가능한 실험 프로토콜로 고정한다.

## 1. 먼저 읽을 파일

작업 전에 반드시 읽어라.

- `pingpong_rl2/docs/report/05_project_completion_plan.md`
- `pingpong_rl2/docs/report/06_learning_design_checklist.md`
- `pingpong_rl2/docs/analysis/reward_dependency_analysis.md`
- `pingpong_rl2/docs/analysis/control_structure_analysis.md`
- `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
- `pingpong_rl2/scripts/run_ppo_learning.py`
- `pingpong_rl2/scripts/run_ppo_rebound_analysis.py`

## 2. 작업 원칙

### 하지 말 것

- 바로 1M 학습을 돌리지 마라.
- reward를 또 하나 덧붙이는 식으로 시작하지 마라.
- CLI 옵션을 더 늘리는 것부터 하지 마라.
- `position`, `position_strike`, `position_tilt`, velocity observation, tilt assist, reward 변경을 무작정 섞지 마라.
- historical run 설명을 그대로 믿지 말고 summary json의 실제 config를 기준으로 판단해라.

### 해야 할 것

- 먼저 현재 코드에 존재하는 모든 학습 관련 옵션을 표로 정리해라.
- “기본 학습 경로에서 쓰는 옵션”과 “실험용으로만 남길 옵션”을 분리해라.
- 최종 목표를 기준으로 reward를 다시 정의해라.
- 실험 command는 가능한 한 짧고 고정된 preset을 쓰도록 정리해라.

## 3. 1단계: 현재 복잡도 정리

`keepup_env.py`와 `run_ppo_learning.py`를 보고 아래 표를 만들어라.

### 3.1 Action / Control 옵션

정리할 것:

- `action_mode`
- `position`
- `position_strike`
- `position_tilt`
- `strike_tilt_ramp_pitch`
- `strike_tilt_assist_limit`
- `target_pitch_range`
- `initial_target_tilt`
- `lateral_action_limit`
- `vertical_action_limit`
- `tilt_action_limit`
- `target_tilt_limit`

각 항목에 대해 적어라.

- 기본값
- 최종 baseline에서 사용할지
- 실험용으로만 둘지
- 제거/비활성화 후보인지
- 해당 옵션이 최종 목표에 직접 도움이 되는 이유

### 3.2 Observation 옵션

정리할 것:

- joint positions
- joint velocities
- racket position
- racket velocity
- target position
- ball position
- ball velocity
- ball relative position
- predicted intercept xy/time
- relative velocity
- racket face normal
- target tilt
- `include_velocity_domain_observation`

특히 아래를 구분해라.

- 최종 baseline에 반드시 필요한 observation
- 없어도 되는 observation
- heuristic leakage가 될 수 있는 observation

### 3.3 Reward 옵션

정리할 것:

- `tracking_term`
- `contact_bonus`
- `apex_match_term`
- `outgoing_x_term`
- `failure_penalty`
- `tilt_angle_penalty`
- `tilt_action_delta_penalty`
- `tracking_during_contact_scale`
- `useful_contact_outgoing_x_penalty_weight`
- `desired_outgoing_ball_velocity_x`

각 항목에 대해 적어라.

- 어떤 행동을 유도하는가
- sparse/dense 중 무엇인가
- exploit 가능성
- 최종 목표와 직접 연결되는가
- 계속 유지할지, 기본값 0으로 둘지, 제거 후보인지

## 4. 2단계: CLI 옵션 정리

현재 command line 옵션이 너무 많아서 실험마다 조금씩 달라지고 있다. 이것 때문에 결과 해석이 어려워졌다.

`run_ppo_learning.py`의 옵션을 아래처럼 분류해라.

### 4.1 항상 명시해야 하는 필수 옵션

예:

- `--run-name`
- `--run-version`
- `--reset-model` 또는 resume 여부
- `--total-timesteps`
- `--seed`

### 4.2 기본값 고정 옵션

평소에는 command line에서 건드리지 않게 할 것.

예:

- `--n-envs`
- `--n-steps`
- `--batch-size`
- `--learning-rate`
- `--gamma`
- `--reset-xy-range`
- `--reset-velocity-xy-range`
- `--reset-velocity-z-range`
- `--max-episode-steps`

각 기본값을 summary와 코드에서 확인해서 문서화해라.

### 4.3 preset으로 묶어야 하는 옵션

현재는 여러 옵션을 사람이 조합해서 실수하기 쉽다.

다음처럼 preset을 제안해라.

- `baseline_position`
- `strike_position`
- `strike_velocity_obs`
- `tilt_experiment`
- `final_candidate`

각 preset은 내부적으로 어떤 env kwargs를 쓰는지 명확히 정의해야 한다.

목표:

```bash
python scripts/run_ppo_learning.py --preset strike_velocity_obs --run-name ... --run-version ...
```

처럼 쓰게 만드는 것이다. 당장 구현하지 않아도, 구현 방향을 문서화해라.

## 5. 3단계: 최종 목표 기준 reward 재설계

현재 핵심 목표는 이것 하나다.

> 탁구공을 계속 칠 수 있는 위치로 올려치는 것.

이 목표는 세 부분으로 나뉜다.

1. 공 아래로 들어간다.
2. 공을 위로 친다.
3. 공이 다시 칠 수 있는 strike zone으로 돌아온다.

reward도 이 세 부분만 표현해야 한다.

### 5.1 유지할 가능성이 높은 reward

- descending strike window alignment
- useful upward contact bonus
- projected apex height quality
- terminal failure penalty

### 5.2 재검토할 reward

- global outgoing `vx` penalty

이유:

- `vx`만 줄이면 평균 x 속도는 줄 수 있다.
- 하지만 공이 실제로 다음 strike zone으로 돌아오는지는 보장하지 않는다.
- robot base 중심이 아니라 racket strike zone이 목표여야 한다.

### 5.3 새 reward 후보

`vx`가 아니라 다음 궤적 품질을 보상해라.

추천 후보:

```python
time_to_apex = max(contact_ball_velocity_z, 0.0) / abs(gravity_z)
projected_apex_xy = contact_ball_xy + contact_ball_velocity_xy * time_to_apex
target_xy = strike_zone_anchor_xy
apex_xy_error = norm(projected_apex_xy - target_xy)
reward_terms["apex_xy_term"] = -weight * apex_xy_error
```

주의:

- 매 step reward로 주지 마라.
- contact event에서만 적용해라.
- 먼저 metric으로 분석하고, 그다음 reward로 승격해라.
- target은 robot base가 아니라 racket home 또는 controller anchor 기반 strike zone이어야 한다.

## 6. 4단계: 정책/학습 구조 재정렬

PPO가 문제인지, reward/환경 설계가 문제인지 아직 단정하지 마라.

먼저 PPO에서 아래 네 가지 clean preset을 같은 조건으로 비교해라.

1. `position`
2. `position_strike`
3. `position_strike + timed negative pitch`
4. `position_strike + velocity-domain observation`

조건:

- 같은 timestep budget
- 같은 seed
- 같은 reset distribution
- 같은 reward
- 한 번에 한 preset만 비교

그 다음에도 막히면 SAC를 검토한다.

SAC는 바로 전환하지 말고 별도 branch/preset으로 계획해라.

## 7. 5단계: 실험 프로토콜

긴 학습 전에 반드시 짧은 run으로 검증해라.

추천 순서:

1. smoke
2. 500k
3. 1M

각 실험 후 반드시 실행:

```bash
python scripts/run_ppo_rebound_analysis.py --run-name ... --run-version ... --episodes 50
```

비교 지표:

- `mean_useful_bounces`
- `max_useful_bounces`
- `failure_counts`
- `ball_out_of_bounds` 비율
- total contacts
- useful contact rate
- contact 후 `ball_velocity_x/y/z`
- projected apex xy error
- 두 번째 useful contact 발생 비율

성공 판단:

- useful bounce가 늘어야 한다.
- `ball_out_of_bounds`가 줄어야 한다.
- contact만 늘고 useful rate가 줄면 실패다.
- `vx`만 줄고 다시 칠 수 있는 위치로 안 돌아오면 실패다.

## 8. 산출물

이번 작업이 끝나면 아래 md 문서를 만들어라.

### 필수 문서

- `pingpong_rl2/docs/report/07_reward_policy_cleanup_plan.md`

내용:

1. 현재 옵션/기본값 정리표
2. 유지할 reward와 버릴 reward
3. 유지할 action mode와 실험용 action mode
4. CLI preset 제안
5. 최종 목표 기준 reward 설계
6. 다음 clean ablation 명령어
7. 1M 학습 전에 확인해야 할 체크리스트

### 선택 구현

문서 정리가 끝난 뒤에만 최소 구현을 해라.

구현 후보:

- CLI preset 추가
- projected apex xy metric을 rebound analysis에 추가
- reward term 이름/summary 정리

단, reward를 새로 학습에 적용하는 것은 metric 검증 뒤에만 해라.

## 9. 최종 보고 형식

작업이 끝나면 아래 형식으로 보고해라.

```text
1. 지금까지 뒤섞였던 부분
2. 정리한 기본값/필수 옵션
3. 최종 baseline으로 추천하는 preset
4. 제거/비활성화할 reward와 이유
5. 새로 추가해야 할 metric 또는 reward 후보
6. 다음에 실제로 돌릴 command
7. 아직 1M을 돌리면 안 되는 이유 또는 돌려도 되는 조건
```

## 10. 가장 중요한 판단

지금 필요한 것은 “옵션 하나 추가”가 아니라 “학습 문제 정의 재정렬”이다.

최종 목표를 항상 이 문장으로 확인해라.

> 이 변경은 공을 더 자주 맞히게 하는가, 그리고 맞힌 공을 다시 칠 수 있는 strike zone으로 돌려보내는가?

이 질문에 답하지 못하는 변경은 하지 마라.
