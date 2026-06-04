# 발표 슬라이드 구성 후보

## 8장 구성

### 1. 문제 정의

메시지:

- 목표는 단순히 공을 한 번 올리는 것이 아니라, 여러 초기 위치/속도에서 장기 랠리를 유지하는 것이다.

넣을 내용:

- MuJoCo + 로봇 라켓 + 공 self-rally
- useful contact 정의
- 최종 목표: `contacts 300 / useful 100`에 가까운 장기 랠리

### 2. 왜 어려운가

메시지:

- 접촉이 희소하고, 실패 원인이 low-apex, ball-out, body contact로 갈라진다.

넣을 내용:

- v25 이전 git 기록: contact trace, feasibility map, action ownership 문서가 먼저 생김
- 초기 v2/v15 실패 사례 요약
- PPO를 오래 돌리는 것보다 action/control abstraction이 중요했다는 결론

선택 그림:

- [assets/02_failure_modes_by_version.png](assets/02_failure_modes_by_version.png) 일부 또는 간단한 실패 모드 표

예비 근거:

- [00_pre_v25_trial_history.md](00_pre_v25_trial_history.md)

### 3. 학습 세팅의 핵심: residual RL

메시지:

- controller가 물리적으로 가능한 기본 타격을 만들고, RL은 residual을 학습한다.

넣을 내용:

- planner/controller와 PPO policy의 역할 분리
- “PID/퍼지와 달리 policy가 상황별 residual을 선택한다”는 설명

선택 그림:

- [assets/07_observation_action_diagram.png](assets/07_observation_action_diagram.png)

### 4. 시행착오 1: horizon과 평가 기준

메시지:

- 600 step에서는 긴 랠리 능력을 볼 수 없어서 v25에서 1800 step과 30+ 기준을 넣었다.

넣을 내용:

- v23/v24/v25 비교 요약
- `stable_cycle_reward_cap=12`
- `30+ useful` checkpoint 기준

선택 그림:

- [assets/01_version_timeline_metrics.png](assets/01_version_timeline_metrics.png)

### 5. 시행착오 2: reset distribution curriculum

메시지:

- 시작 영역/속도를 넓히는 것은 별도 과제이고, 한 번에 넓히면 성능이 무너진다.

넣을 내용:

- v28: xy/velocity/spin을 한 번에 크게 줘서 실패
- v30: xy만 `0.10m`로 넓혀 안정화
- v31: `0.14m`는 너무 빨라 ball-out/body contact 증가

선택 그림:

- [assets/01_version_timeline_metrics.png](assets/01_version_timeline_metrics.png)
- [assets/02_failure_modes_by_version.png](assets/02_failure_modes_by_version.png)

### 6. 시행착오 3: 15D에서 17D로 확장

메시지:

- 17D는 기존 안정 정책을 버린 것이 아니라, 15D policy를 전이하고 tracking residual 2축만 추가했다.

넣을 내용:

- `15D + tracking_vx/vy`
- action head prefix copy
- 새 action row zero-init
- v32 이후 실제 tracking 축 사용 확인

선택 그림:

- [assets/07_observation_action_diagram.png](assets/07_observation_action_diagram.png)
- [assets/05_action_usage_17d.png](assets/05_action_usage_17d.png)

### 7. 최종 개선: v34 장기 랠리와 v35 trade-off

메시지:

- v34는 더 넓은 z/xy/velocity 범위에서 장기 랠리 지표가 개선됐고, v35는 안정성 강화가 항상 장기 랠리 개선으로 이어지지는 않음을 보여줬다.

넣을 내용:

- `reset_xy=0.12`
- `height=[0.22,0.52]`
- `velocity_xy=0.035`
- `velocity_z=[-0.12,0.04]`
- `stable_cycle_reward_cap=30`
- `low_apex_grace=6`

선택 그림:

- [assets/03_long_horizon_target_hits.png](assets/03_long_horizon_target_hits.png)
- [assets/04_apex_height_distribution.png](assets/04_apex_height_distribution.png)

핵심 수치:

- v34 long eval20 mean contacts `318.55`
- mean useful `116.05`
- `contacts>=300 & useful>=100`: `13/20`
- `contacts>=400 & useful>=150`: `9/20`
- v35는 robot body contact를 `3/20 -> 1/20`으로 줄였지만, `contacts>=400 & useful>=150`은 `4/20`으로 하락

### 8. 검증과 한계

메시지:

- action 차원이 많아 보일 수 있어 usage와 ablation으로 검증했다. 현재는 17D 유지가 타당하고, 다음 개선은 v34 기반 balanced fine-tune이다.

넣을 내용:

- outgoing_x, strike_plane_z는 제거하면 성능 급락
- tracking/centering은 사용량은 작지만 제거하면 성능 하락
- 현재 다음 병목은 ball-out과 link5 body contact
- v35처럼 강하게 안정성 reward를 밀면 body contact는 줄지만 long rally 품질이 떨어질 수 있음

선택 그림:

- [assets/06_action_ablation_mean_useful.png](assets/06_action_ablation_mean_useful.png)

예비 근거:

- [07_v35_training_review_and_next_plan.md](07_v35_training_review_and_next_plan.md)

마무리 문장:

> 최종 모델은 단순한 반복 제어가 아니라, 물리 primitive 위에서 RL이 residual action을 학습해 더 긴 horizon과 넓은 초기 조건을 다루도록 만든 결과입니다.

## 예비 슬라이드 후보

### A. 로그와 학습 운영

그림:

- [assets/08_monitor_training_curves.png](assets/08_monitor_training_curves.png)

사용 시점:

- “학습이 멈춘 것처럼 보였던 경험”을 말할 때
- 구현/실험 운영 역량을 보여주고 싶을 때

### B. 1800-step vs 7200-step

핵심 설명:

- 1800-step은 30+ bounce 평가용
- 7200-step은 300 contact/100 useful 목표 평가용

질문 대응:

- “왜 어떤 분석에서는 v34가 30 정도밖에 안 나오나?”라는 질문에 사용

### C. low-apex 기준

그림:

- [assets/04_apex_height_distribution.png](assets/04_apex_height_distribution.png)

핵심 설명:

- threshold를 낮춰 성공으로 세는 것이 아니라, grace로 회복 기회만 늘렸다.

### D. 2공/바람/외란 확장

현재 결론:

- 2공은 observation/reward/termination이 모두 달라져 별도 env/model이 필요하다.
- 바람이나 lateral disturbance는 가능하지만, 먼저 단일 공에서 reset velocity/position curriculum과 residual policy 검증을 발표하는 것이 안정적이다.

말할 수 있는 확장 방향:

- wind/disturbance domain randomization
- two-ball scheduling
- observation ablation/permutation
- radial/tangent outgoing residual action mode
