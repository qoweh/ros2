# 시각화 자료 카탈로그

발표에서 모든 그림을 다 넣을 필요는 없다. 핵심 발표에는 4개 정도만 쓰고, 질문 대응/예비 슬라이드로 나머지를 두는 구성이 좋다.

## 우선순위 A: 본 발표에 넣기 좋은 그림

### 1. Version Timeline

파일: [assets/01_version_timeline_metrics.png](assets/01_version_timeline_metrics.png)

효과:

- v25 이후 실험이 무작위 튜닝이 아니라 단계적이었다는 것을 보여준다.
- `reset_xy_range`, `15D/17D`, `mean useful`, `30+ rate`를 한 그림에 담는다.
- v31처럼 범위를 넓혔다가 성능이 떨어진 시행착오도 숨기지 않고 보여줄 수 있다.

발표 문장:

> v25에서는 긴 horizon으로 30회 이상 랠리를 평가할 수 있게 했고, v26/v30에서는 reset 범위를 넓혔습니다. v31처럼 너무 빨리 넓히면 실패가 늘었고, 이후 17D와 long-horizon reward cap으로 다시 성능을 올렸습니다.

주의:

- v34 수치는 training summary의 7200-step evaluation이므로 v32/v30의 3600-step evaluation과 완전히 같은 조건은 아니다.
- 그래서 “완전 공정 비교”보다는 “프로젝트 방향 변화”를 보여주는 그림이라고 말한다.

### 2. Long Horizon Target Hits

파일: [assets/03_long_horizon_target_hits.png](assets/03_long_horizon_target_hits.png)

효과:

- 사용자의 목표였던 `contacts 300 / useful 100`을 직접 보여준다.
- v34가 단순 30회 성공이 아니라 장기 랠리 목표에 가까워졌다는 근거가 된다.
- v35처럼 일부 안정성 지표가 좋아져도 long-horizon 목표가 떨어질 수 있다는 trade-off를 보여준다.

발표 문장:

> 1800-step 평가에서는 300 contact를 보기 어렵기 때문에, 장기 랠리 목표는 7200-step 분석으로 따로 봤습니다. v34는 20 episode 중 13번 `contacts 300/useful 100`을 넘겼고, v35는 body contact는 줄였지만 같은 목표가 11번으로 내려갔습니다.

주의:

- 20 episode라 샘플 수가 작다. 최종 방어용이면 50 또는 100 episode long eval을 추가하면 더 좋다.

### 3. Action Ablation

파일: [assets/06_action_ablation_mean_useful.png](assets/06_action_ablation_mean_useful.png)

효과:

- “17D가 정말 필요한가?”라는 질문에 직접 대응한다.
- action magnitude만 보지 않고, 실제 성능 하락으로 판단했다는 점이 강화학습답다.

발표 문장:

> action 사용량만 보면 약해 보이는 축이 있지만, 그 축을 0으로 막으면 성능이 떨어졌습니다. 그래서 현재는 차원을 줄이기보다, 어떤 축이 어떤 실패 모드를 줄이는지 ablation으로 검증했습니다.

주의:

- 이 ablation은 빠른 비교용 `12 episodes / 3600 steps`다.
- “최종 통계”라기보다 “삭제 판단을 위한 진단 실험”이라고 설명한다.

### 4. Observation/Action Diagram

파일: [assets/07_observation_action_diagram.png](assets/07_observation_action_diagram.png)

효과:

- `17D action`과 `55D observation` 혼동을 깔끔하게 정리한다.
- 심사위원이 “output이 55D인가?”라고 물을 때 바로 설명할 수 있다.

발표 문장:

> 정책 네트워크는 55차원의 상태를 입력으로 받고, 17차원의 residual action을 출력합니다. 55D는 출력이 아니라 로봇/공/다음 접촉 예측을 포함한 observation입니다.

## 우선순위 B: 질문 대응/예비 슬라이드

### 5. Failure Modes By Version

파일: [assets/02_failure_modes_by_version.png](assets/02_failure_modes_by_version.png)

효과:

- 성능이 좋아지며 실패 모드가 어떻게 바뀌는지 보여준다.
- v34에서 low-apex가 줄고 ball-out/body contact가 새 병목이 된 점을 설명하기 좋다.

발표 문장:

> 실패 모드를 보면 개선 방향이 바뀝니다. v33까지는 low-apex가 컸지만, v34에서는 low-apex가 줄고 ball-out과 body contact가 다음 병목으로 나타났습니다. v35는 body contact를 줄였지만 ball speed와 장기 랠리 hit rate에서 trade-off가 생겼습니다.

주의:

- 각 version의 evaluation step limit과 episode 수가 다를 수 있다. 조건 차이는 말해두는 것이 좋다.

### 6. Apex Height Distribution

파일: [assets/04_apex_height_distribution.png](assets/04_apex_height_distribution.png)

효과:

- low-apex threshold를 낮출지 말지 논의할 때 유용하다.
- useful minimum `0.20m`와 low-apex termination `0.14m`를 분리해서 보여준다.

발표 문장:

> 낮은 공을 모두 성공으로 세지 않기 위해 useful 기준은 유지했습니다. 대신 회복 가능한 낮은 contact를 바로 종료하지 않도록 grace를 늘렸습니다.

### 7. Action Usage

파일: [assets/05_action_usage_17d.png](assets/05_action_usage_17d.png)

효과:

- 17D action 중 policy가 어느 축을 크게 쓰는지 한눈에 보인다.
- ablation 그림과 함께 쓰면 “많이 씀”과 “중요함”이 다를 수 있다는 메시지가 좋다.

발표 문장:

> action magnitude만 보면 tilt roll, outgoing x, strike plane z가 크게 쓰입니다. 하지만 제거 실험을 해보면 tilt roll은 애매하고, outgoing x와 strike plane z는 치명적입니다.

### 8. Monitor Training Curves

파일: [assets/08_monitor_training_curves.png](assets/08_monitor_training_curves.png)

효과:

- “학습 중 로그가 안 떠서 멈춘 줄 알았다”는 시행착오를 설명하기 좋다.
- monitor CSV가 실제 학습 진행 확인 자료였다는 것을 보여준다.

발표 문장:

> 터미널 로그가 항상 즉시 보이는 것은 아니어서 monitor CSV로 학습 진행을 확인했습니다. 중간 중단 때문에 monitor 파일이 여러 개 생겼고, 이후에는 `python -u`와 monitor tail을 같이 사용했습니다.

## 추천 조합

짧은 발표:

1. Version Timeline
2. Long Horizon Target Hits
3. Observation/Action Diagram
4. Action Ablation

긴 발표:

1. Version Timeline
2. Failure Modes
3. Apex Distribution
4. Long Horizon Target Hits
5. Observation/Action Diagram
6. Action Usage
7. Action Ablation
8. Monitor Training Curves

방어 질문 대비:

- `왜 17D가 필요한가?` -> Action Usage + Action Ablation
- `55D는 output인가?` -> Observation/Action Diagram
- `왜 v34는 1800-step에서 별로 안 좋아 보이나?` -> Long Horizon Target Hits
- `v35까지 했는데 왜 v34를 쓰나?` -> Long Horizon Target Hits + Failure Modes
- `왜 low-apex 기준을 더 낮추지 않았나?` -> Apex Height Distribution
- `학습이 제대로 돈 게 맞나?` -> Monitor Training Curves
