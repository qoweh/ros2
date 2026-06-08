# v25 이전 시행착오 기록

작성일: 2026-06-05

## 한 줄 요약

git 기록과 `docs/report/`의 초기 진단 문서들을 보면 v25 이전에도 충분히 많은 시행착오가 있었다. v25 이후가 “성능을 발표 가능한 수준으로 끌어올린 구간”이라면, v25 이전은 “왜 단순 PPO/reward 튜닝으로는 안 되는지 이해하고, RL에게 어떤 action을 맡길지 정한 구간”이다.

## git 기록에서 보이는 흐름

아래 commit들은 v25 이전 시행착오를 발표에서 말할 때 근거로 쓰기 좋다.

| commit | 의미 |
| --- | --- |
| `106cc1c` | `PingPongSim`과 기본 학습 유틸 구현 |
| `f798d5c` | checkpoint resume/reset 옵션 추가. 학습 운영 문제가 실제로 있었음 |
| `50d8156` | 프로젝트 완료 계획, 학습 설계 체크리스트 추가 |
| `faa7976` | keep-up task 자체를 다시 정의하는 계획 문서 추가 |
| `3214c70` | contact trace sanity, contact feasibility map, heuristic diagnostic 대량 추가 |
| `901bfba` | contact-frame primitive 구현 |
| `07dd6eb` | contact-frame primitive에 residual-RL action mode와 evaluation metric 확장 |
| `4259a0d` | 5D에서 8D velocity residual action으로 action ownership 확장 |
| `65d0867` | low-apex termination과 reset height 개선 |
| `c61480f` | stable cycle objective 추가 |
| `ae9b28f` | v20 boundary brake/contact offset 조정 |
| `e8b1817` | apex/timing residual action mode 추가 |
| `1202645` | v22 low-stable window, apex height criteria 개선 |
| `3063fc7` | v25 long horizon과 30+ metric 추가 |

해석:

- v25 이전은 “그냥 PPO를 더 돌리기”가 아니라, 실패를 측정하기 위한 도구를 먼저 만들고 있었다.
- contact trace, feasibility map, next-intercept metric, action ownership 문서가 모두 v25 이전에 생겼다.
- 즉 발표에서 초반 구간을 생략하면, 최종 모델이 왜 residual RL 구조가 됐는지 설명이 약해진다.

## 초반 시행착오 1: reward만 늘리면 해결되지 않았다

관련 문서:

- [06_learning_design_checklist.md](../report/06_learning_design_checklist.md)
- [08_easy_next_ball_completion_plan.md](../report/08_easy_next_ball_completion_plan.md)
- [26_learning_runtime_parallel_and_v2_diagnosis.md](../report/26_learning_runtime_parallel_and_v2_diagnosis.md)

초반에는 useful bounce, next-intercept, easy-next-ball 같은 reward/metric을 추가하는 방향을 검토했다. 하지만 바로 reward로 승격하지 않고 analysis-only metric으로 먼저 본 기록이 있다.

중요한 교훈:

- 좋은 metric이 있어도 reward로 바로 넣으면 causal signal이 지저분할 수 있다.
- `first contact` 기준에서 useful episode와 zero-bounce episode가 깔끔하게 분리되지 않으면 reward로 쓰기 위험하다.
- 그래서 초반에는 reward보다 control-side assist와 contact quality logging이 우선이었다.

발표 문장:

> 처음에는 다음 공을 치기 쉬운 위치로 보내는 reward를 넣으면 해결될 것처럼 보였지만, 분석해보니 좋은 contact 자체가 너무 드물었습니다. 그래서 reward를 늘리기 전에 contact trace와 next-intercept metric을 먼저 만들었습니다.

## 초반 시행착오 2: scripted controller에도 ceiling이 있었다

관련 문서:

- [15_contact_feasibility_map_report.md](../report/15_contact_feasibility_map_report.md)

contact feasibility map은 PPO를 더 돌리는 대신, scripted controller가 현재 action/control surface에서 반복 랠리를 만들 수 있는지 먼저 확인한 실험이다.

결과 요지:

- coarse/finalist sweep에서 좋은 contact 조합은 일부 찾았다.
- 하지만 narrow evaluation에서도 `max useful bounces`가 `2`를 넘지 못했다.
- 따라서 병목은 reward가 아니라 physical/contact execution 쪽이었다.

발표 문장:

> 단순 heuristic이나 scripted controller가 이미 충분했다면 RL이 필요 없었을 겁니다. 하지만 feasibility sweep에서 scripted 조합은 반복적으로 2회 근처에서 막혔고, 이것이 residual RL action을 열게 된 이유입니다.

## 초반 시행착오 3: bootstrap 이후 PPO continuation이 skill을 망가뜨리기도 했다

관련 문서:

- [21_contact_frame_primitive_report.md](../report/21_contact_frame_primitive_report.md)

contact-frame primitive 구간에서는 bootstrap으로 만든 skill을 PPO로 이어서 학습했을 때 오히려 성능이 망가지는 기록이 있다.

핵심 교훈:

- 좋은 low-level skill이 생겼다고 PPO continuation이 항상 개선하지 않는다.
- checkpoint/evaluation 기준이 맞지 않으면, RL update가 기존 skill을 손상할 수 있다.
- 이후 v25/v32/v34에서 conservative resume, 낮은 learning rate, 작은 clip range를 쓰게 된 배경이다.

발표 문장:

> PPO를 더 오래 돌리면 자동으로 좋아지는 것이 아니었습니다. 이미 배운 접촉 skill을 망가뜨리는 경우도 있어서, 이후에는 전이 학습과 보수적인 fine-tuning을 사용했습니다.

## 초반 시행착오 4: action ownership를 점진적으로 넘겼다

관련 문서:

- [36_rl_action_ownership_and_8d_residual_plan.md](../report/36_rl_action_ownership_and_8d_residual_plan.md)
- [37_v16_8d_residual_review_and_v17_direction.md](../report/37_v16_8d_residual_review_and_v17_direction.md)
- [38_v17_contact_timing_velocity_tilt_residual.md](../report/38_v17_contact_timing_velocity_tilt_residual.md)
- [39_v17_action_scale_and_v18_lateral_residual.md](../report/39_v17_action_scale_and_v18_lateral_residual.md)

초기에는 RL action이 라켓 위치/tilt residual 중심이었다. 하지만 `ball_out_of_bounds`가 계속 남으면서, 공을 어느 방향으로 보낼지 직접 조절하는 outgoing velocity residual이 필요하다는 결론이 나왔다.

진화 흐름:

```text
5D position/tilt residual
-> 8D velocity residual
-> tilt/lateral/apex timing residual
-> 15D contact-frame residual
-> 17D tracking residual
```

발표 문장:

> 모든 것을 RL에게 맡긴 것이 아니라, controller가 안정적으로 할 수 있는 부분은 유지하고, 실패 모드를 줄이는 데 필요한 residual action만 단계적으로 열었습니다.

## 초반 시행착오 5: low-apex와 low-bounce loop를 구분해야 했다

관련 문서:

- [30_v5_low_apex_and_height_reward_fix.md](../report/30_v5_low_apex_and_height_reward_fix.md)
- [31_v6_low_bounce_loop_and_strict_cycle_fix.md](../report/31_v6_low_bounce_loop_and_strict_cycle_fix.md)
- [32_v7_low_apex_recovery_reward_fix.md](../report/32_v7_low_apex_recovery_reward_fix.md)
- [40_v18_low_loop_and_v19_height_qualified_reward.md](../report/40_v18_low_loop_and_v19_height_qualified_reward.md)

초반에는 공을 위로만 보내면 성공처럼 보일 수 있었지만, 낮은 통통 loop가 생기면 실제 keep-up 과제와 달라진다. 그래서 useful contact 기준에 apex height와 next-intercept quality를 넣는 방향으로 갔다.

발표 문장:

> 낮게 계속 튀기는 것도 겉으로는 공을 살린 것처럼 보일 수 있지만, 목표는 다음 타격이 가능한 안정 랠리였습니다. 그래서 low-apex loop를 성공으로 세지 않도록 useful contact 기준을 강화했습니다.

## 발표에 넣는 추천 방식

본 발표에는 너무 많은 early version을 모두 나열하지 말고, 3개의 교훈으로 압축하는 것이 좋다.

1. 측정부터 만들었다.
   - contact trace, feasibility map, rebound analysis
2. PPO를 더 돌리는 것만으로는 안 됐다.
   - bootstrap continuation이 skill을 망가뜨린 기록
3. action ownership를 점진적으로 넘겼다.
   - 5D -> 8D -> 15D -> 17D

이 세 가지를 말한 뒤 v25 이후 성능 상승 그래프로 넘어가면 자연스럽다.
