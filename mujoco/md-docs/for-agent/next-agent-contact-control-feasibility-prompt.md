# 다음 에이전트 프롬프트: reward 튜닝 중단, contact/control feasibility부터 검증

너는 `pingpong_rl2` 프로젝트를 이어받는 에이전트다.

작업 루트:

```bash
/Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
```

## 최종 목표

Franka Panda 로봇팔이 탁구채로 탁구공을 계속 위로 올려치는 강화학습을 완성한다.

현재 핵심 실패는 이것이다.

- 첫 타격은 어느 정도 된다.
- 공이 위로도 올라간다.
- 하지만 공이 다음에 로봇팔/라켓이 다시 치기 쉬운 위치로 돌아오지 않고 바깥쪽으로 간다.
- 그래서 `2+ useful bounce`는 드물고 `3+`는 거의 없다.

이번 작업의 방향은 `reward weight`를 더 찍어보는 것이 아니다.

> 먼저 이 환경의 라켓 접촉/제어 구조가 원하는 outgoing ball velocity를 만들 수 있는지 증명하라. scripted/diagnostic도 `3+`를 못 만들면 PPO reward를 아무리 바꿔도 해결되지 않는다.

## 반드시 먼저 읽을 파일

- `../agent-answer-new-1.md`
- `../agent-answer-new-2.md`
- `docs/report/13_direct_trajectory_objective_report.md`
- `src/pingpong_rl2/envs/keepup_env.py`
- `src/pingpong_rl2/envs/pingpong_sim.py`
- `src/pingpong_rl2/controllers/ee_pose_controller.py`
- `src/pingpong_rl2/controllers/heuristic_keepup.py`
- `scripts/run_heuristic_keepup_diagnostic.py`
- `scripts/run_ppo_learning.py`

## 현재까지 확정된 판단

`docs/report/13_direct_trajectory_objective_report.md`의 결론을 반복하지 말고 이어받아라.

- `desired outgoing velocity` metric 자체는 유효하다.
- 기존 best model들에서 `2+ useful bounce episode`가 `zero-bounce episode`보다 outgoing velocity error가 낮았다.
- 하지만 `trajectory_match_reward_weight`를 reward로 바로 올린 실험은 regression이었다.
- heuristic diagnostic도 narrow reset에서 `mean_useful_bounces=0.54`, `two_or_more_rate=0.05`, `three_or_more_rate=0.00` 수준이었다.

따라서 다음 병목은 reward가 아니라 control/physics/contact geometry일 가능성이 높다.

## 이번 작업에서 하지 말 것

- 바로 PPO 1M 학습을 돌리지 마라.
- `trajectory_match_reward_weight`를 키우는 실험을 반복하지 마라.
- `easy_next_ball_reward`, `outgoing_x_penalty`, `return_assist_weight`, `followup_tilt` 값을 랜덤하게 조합하지 마라.
- 새 reward term을 여러 개 추가하지 마라.
- 평균 reward만 보고 개선됐다고 판단하지 마라.
- scripted/diagnostic이 `3+`를 못 만드는 상태에서 PPO로 넘어가지 마라.

## 핵심 가설

현재 `position_strike` action은 라켓의 목표 위치/tilt를 간접적으로 바꾼다.

하지만 탁구공의 다음 위치는 contact 순간의 아래 값들로 결정된다.

- contact point
- incoming ball velocity
- racket velocity
- racket face/contact normal
- restitution/friction/contact solver behavior
- timing

지금 policy/control surface가 이 값들을 충분히 직접 제어하지 못하면, reward를 잘 써도 공을 다시 칠 수 있는 곳으로 보내기 어렵다.

## 1단계: contact trace를 먼저 정확하게 만들어라

현재 `PingPongSim.step_with_contact_trace()`는 contact가 처음 관측된 substep의 값을 기록한다. 이 값이 진짜 "contact 직후 outgoing velocity"인지 불명확하다.

먼저 contact trace를 진단용으로 확장하라.

필요한 값:

- contact 직전 substep의 ball velocity
- contact가 관측된 substep의 ball velocity
- contact 이후 1-5 substep 뒤 ball velocity
- contact가 끝난 직후 ball velocity
- MuJoCo contact position
- MuJoCo contact frame normal
- racket center velocity
- racket face normal
- relative velocity: `ball_velocity - racket_velocity`

중요:

- `racket_face_normal`만 믿지 말고 `data.contact[i].frame[:3]`의 실제 contact normal도 기록하라.
- ball/racket contact pair를 찾을 때 `ball_geom`, `racket_head`만 대상으로 해라.
- `outgoing_velocity_error_norm` 계산은 가능하면 contact 이후 안정된 post-contact velocity로 다시 계산해라.

산출물:

- `docs/report/14_contact_trace_sanity_report.md`
- 기존 metric이 "충돌 도중 속도"를 보고 있었는지, "진짜 충돌 후 속도"를 보고 있었는지 결론.

## 2단계: 라켓 접촉 feasibility map을 만들어라

PPO 전에, 작은 scripted 실험으로 아래 질문에 답하라.

> 이 MuJoCo 모델에서 pitch/roll/라켓 속도/타이밍을 바꾸면 desired outgoing velocity를 만들 수 있는가?

새 스크립트 후보:

```text
scripts/run_contact_feasibility_map.py
```

실험 조건:

- reset은 deterministic/narrow로 시작한다.
- 공을 라켓 위 정해진 위치에 둔다.
- incoming ball velocity는 몇 가지 대표값만 쓴다.
- pitch/roll을 작은 grid로 sweep한다.
- strike z boost 또는 target z pulse를 sweep한다.
- 필요하면 target x/y pre-offset도 sweep한다.

기록할 summary:

- pitch
- roll
- requested target z pulse
- actual contact racket velocity x/y/z
- actual contact normal
- pre-contact ball velocity
- post-contact ball velocity
- desired outgoing velocity
- outgoing velocity error norm
- predicted next apex xy
- useful bounce 여부
- episode useful bounce count

판정:

- 어떤 pitch/roll/velocity 조합에서도 desired outgoing error가 낮아지지 않으면 physics/contact/normal/controller 문제다.
- 특정 조합에서만 낮아지면 그 조합을 strike primitive로 만들어야 한다.
- pitch sign이 반대로 보이면 `followup_strike_target_tilt=(-0.03, 0.0)` 같은 고정값을 계속 쓰지 말고 sign부터 고쳐라.

최소 통과 기준:

- deterministic reset에서 scripted sweep best가 `max_useful_bounces >= 3`를 최소 한 번 보여야 한다.
- 가능하면 100 episode narrow reset에서 `three_or_more_useful_bounce_rate >= 0.10` 이상을 목표로 한다.
- 이 기준 전에는 PPO 학습을 하지 마라.

산출물:

- `docs/report/15_contact_feasibility_map_report.md`
- best 조합 표
- "가능/불가능/보류" 결론

## 3단계: 문제가 physics인지 controller인지 분리하라

feasibility map이 실패하면 아래를 순서대로 분리 진단하라.

1. **contact normal sign**
   - 실제 MuJoCo contact frame normal과 `racket_face_normal`의 방향이 일치하는지 확인한다.
   - pitch/roll sign이 생각과 반대로 공을 보내는지 확인한다.

2. **post-contact velocity measurement**
   - first contact substep velocity가 아니라 contact 이후 속도를 쓰고 있는지 확인한다.

3. **racket velocity ceiling**
   - `RacketCartesianController.compute_joint_targets()`는 target position error를 `max_position_step=0.05`로 자르고 joint position target만 만든다.
   - 이 구조가 contact 순간 충분한 upward/lateral racket velocity를 만들 수 있는지 수치로 확인한다.
   - `contact_racket_velocity_z`, `contact_racket_velocity_x/y` 분포를 best scripted/PPO 모두에서 비교한다.

4. **contact parameters**
   - `assets/franka/panda.xml`의 `racket_head`
   - `assets/scene.xml`의 `ball_geom`
   - `solref`, `solimp`, `friction`, ball mass, racket geom orientation을 sanity check한다.
   - 물리 파라미터를 바꿀 때는 `run_bounce_sanity.py` 또는 micro rollout으로 전후 비교표를 남겨라.

## 4단계: 가능하면 strike primitive/action mode로 바꿔라

feasibility map에서 "이 조합이면 공이 다시 strike zone으로 온다"가 나오면, PPO가 raw position residual로 그걸 발견하길 기다리지 말고 primitive로 제공해라.

후보 구현:

```text
action_mode = "contact_primitive" 또는 "hybrid_strike"
```

정책 action은 아래처럼 고수준 residual만 제어하게 한다.

- next apex target xy residual
- target apex height residual
- strike timing/urgency residual
- racket normal pitch/roll residual
- optional z-pulse scale

primitive 내부는 scripted하게 처리한다.

- 공이 내려올 때 intercept xy로 이동
- contact 직전에는 feasibility map에서 찾은 pitch/roll/velocity pulse를 적용
- contact 이후에는 predicted next intercept 쪽으로 복귀

주의:

- policy가 매 step raw target position으로 모든 것을 직접 배우게 두지 마라.
- "공을 계속 치기 쉬운 위치로 보내기"는 contact primitive 안에서 기본 동작으로 보장하고, RL은 residual만 배우게 하는 쪽이 맞다.

## 5단계: scripted primitive가 먼저 3+를 해야 한다

새 primitive를 만든 뒤 PPO 전에 아래를 통과시켜라.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_primitive_narrow_100ep \
  --variant-name contact_primitive \
  --episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01
```

통과 기준:

- `max_useful_bounces >= 3`
- `three_or_more_useful_bounce_rate > 0`
- `zero_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm`보다 `two_or_more...`가 낮아야 함
- viewer로 봤을 때 공이 라켓에서 멀어지는 방향이 아니라 라켓/anchor 근처로 돌아와야 함

## 6단계: 그 다음에만 RL 학습을 재개하라

scripted/primitive가 통과하면 새 preset을 하나로 고정해라.

예:

```text
--preset contact_primitive_candidate
```

이 preset에 필요한 값을 모두 넣고, 커맨드라인 옵션 조합으로 실험이 흩어지지 않게 해라.

처음 학습은 narrow curriculum으로만 한다.

추천 순서:

1. deterministic/narrow reset, 100k-300k
2. narrow에서 `three_or_more_useful_bounce_rate > 0` 확인
3. reset xy/velocity range를 한 단계만 넓힘
4. 다시 같은 평가

평가 gate:

- `mean_useful_bounces`
- `one_or_more_useful_bounce_rate`
- `two_or_more_useful_bounce_rate`
- `three_or_more_useful_bounce_rate`
- `outgoing_velocity_error_norm`
- `failure_reason` distribution

## 최종 산출물

이번 에이전트는 최소 아래 문서를 남겨라.

- `docs/report/14_contact_trace_sanity_report.md`
- `docs/report/15_contact_feasibility_map_report.md`
- 구현했다면 `docs/report/16_contact_primitive_training_plan.md`

각 문서에는 반드시 아래를 포함한다.

- 무엇을 바꿨는가
- 왜 바꿨는가
- 어떤 명령어를 실행했는가
- 숫자로 어떤 결과가 나왔는가
- 다음 단계가 PPO인지, physics/controller 수정인지 명확한 결론

## 한 줄 결론

지금은 "공을 로봇팔 쪽으로 보내는 reward"를 더 쓰는 단계가 아니다.  
먼저 contact 순간 라켓이 원하는 outgoing velocity를 만들 수 있는지 검증하고, 가능하면 그 동작을 strike primitive/action mode로 제공한 뒤 RL을 붙여라.
