# pingpong_rl 문서 기반 미완료 작업 정리

## 1. 이 문서의 목적

이 문서는 현재 `docs/log`와 `docs/report`에 흩어져 있는 "다음에 하자", "아직 안 했다", "보류했다" 항목을 한 번에 모아 현재 코드 기준으로 다시 정리한 것이다.

핵심 목적은 두 가지다.

- 예전에 하기로 했는데 그냥 넘어간 항목을 찾기
- 지금 실제로 남은 일이 무엇인지 우선순위를 다시 세우기

## 2. 지금까지 이미 된 것

문서 초안 시점에는 다음이 미완료 또는 보류였지만, 현재 코드에는 이미 들어가 있다.

- `racket_center` 기준 EE delta action 구조
- flat observation `(26,)` 계약
- `PingPongEEDeltaEnv`와 `PingPongEEDeltaGymEnv`
- success / failure / time-limit 분리
- rollout analysis CSV/JSON export
- PPO logging bridge
- PPO baseline 학습 스크립트
- PPO render 스크립트
- heuristic `keep-up` baseline 스크립트 (`run_keepup_baseline.py`)
- passive viewer 경로에서 home pose + ball reset 적용
- `README.md` 기반 구조 설명
- 성공 시 즉시 종료하지 않고, 연속 bounce를 계속 세는 EE env 계약
- `successful_bounce_count` / `episode_success_reason` 기반 logging 반영

즉 문서 중간에 자주 나오던 아래 항목은 현재 기준으로는 이미 완료다.

- Gymnasium wrapper 연결
- EE delta env 계약 고정
- PPO logging schema 연결
- viewer와 baseline 실행 경로 분리

## 3. 아직 안 된 것

현재 문서와 코드를 함께 보면, 정말로 아직 남아 있는 항목은 아래다.

### 3.1 scripted keep-up baseline은 들어갔지만 아직 안정적이진 않다

이건 여러 문서에서 반복해서 "RL보다 먼저 필요"하다고 적혀 있었고, 지금은 첫 heuristic baseline이 추가됐다.

현재 있는 것은:

- `run_bounce_baseline.py`: 물리 sanity check
- `run_keepup_baseline.py`: heuristic keep-up baseline
- `run_viewer.py`: hold/joint/EE demo

하지만 아직 부족한 것은:

- 장시간 time-limit까지 안정적으로 가는 scripted baseline
- repeated contact로 과도한 에너지를 넣지 않는 더 정교한 접촉 분리 전략
- 여러 초기 조건에서 재현 가능한 keep-up baseline

즉 "없다" 단계는 지났지만, 아직 기준선으로 완성됐다고 보긴 어렵다.

### 3.2 reward는 아직 완성본이 아니다

현재 reward는 여전히 single-bounce 초안 수준이다.

현재 들어간 항목:

- contact bonus
- 공이 라켓보다 위에 있을 때의 height term
- success bonus
- failure penalty

최근 수정으로 아래는 보강됐다.

- transient contact trace 기준 success 판정
- success 시 명시적 bonus 지급
- XY alignment penalty 추가
- second/third bounce에서 더 큰 보상을 주는 multi-bounce weighting 추가

하지만 아직 없는 것:

- XY drift penalty
- action smoothness penalty
- hit timing 안정성 보상
- multi-bounce contact count 보상
- second/third bounce 가중치

즉 reward는 "완전히 비어 있던 상태"는 아니지만, 아직 실험용 초안에서 벗어나지 않았다.

### 3.3 success threshold 재보정

문서에서는 여러 번 "먼저 분포를 더 보고 threshold를 조정하라"고 적혀 있다.

현재 상태:

- 기본 threshold는 그대로 `0.5`
- success 판정은 contact trace 기반으로 더 맞게 바뀌었음

하지만 아직 안 한 것:

- multi-episode distribution을 다시 모아서 threshold 재검토
- end-of-step contact와 transient contact 차이까지 반영한 기준 재설정

즉 threshold 값 자체는 아직 사실상 1차값이다.

### 3.4 orientation 관련 확장

문서에서 계속 보류된 항목인데, 지금도 그대로다.

아직 안 된 것:

- orientation action
- racket orientation observation
- orientation reward

현재 프로젝트는 위치 3축 delta 중심이고, orientation은 의도적으로 닫아 둔 상태다.

### 3.5 observation 확장

현재 observation 계약은 `(26,)`으로 고정돼 있고, 이건 괜찮다. 다만 문서상 다음 후보들은 아직 안 들어갔다.

- racket velocity
- racket orientation
- ball spin
- 추가 normalization 정책 정리

즉 지금 observation은 single-bounce 입문 단계에는 충분하지만, 더 어려운 탁구 과제에는 아직 부족하다.

### 3.6 long-run PPO baseline과 본격 튜닝

현재는 smoke/baseline 수준의 실행 경로와 logging은 있다. 하지만 아래는 아직 안 한 것으로 보는 편이 맞다.

- 장기 timesteps 기반 baseline 성능 측정
- MPS/device 최적화 본실험
- PPO hyperparameter sweep
- reward coefficient sweep

참고: device 자체는 이미 확인했다. 현재 환경에서 `torch`는 `MPS` 사용 가능이지만, `run_ppo_baseline.py`의 `--device auto`는 실제로 `cpu`를 고른다. `--device mps`를 명시해야 Apple GPU를 강제로 쓸 수 있다.

즉 학습 인프라는 있지만, 성능 튜닝 단계는 아직 본격적으로 시작하지 않았다.

### 3.7 actual ping-pong task로의 확장

여러 문서가 강조하듯, 현재 프로젝트는 아직 `탁구 경기`가 아니라 `keep-up / multi-bounce` 쪽에 더 가깝다.

다만 이 항목은 현재 목표 기준으로는 우선순위가 내려갔다. 지금 사용자가 원하는 것은 실제 탁구가 아니라, 공을 떨어뜨리지 않고 계속 튕기는 것이다.

아직 안 된 것:

- table 추가
- net 추가
- landing target zone
- incoming ball distribution 확대
- scripted incoming-ball intercept baseline
- limited rally / multi-bounce task 설계

즉 현재 env는 당분간 `keep-up` 과제에 집중하고, 실제 탁구 task 확장은 뒤로 미뤄도 된다.

## 4. 예전에 하기로 했는데 그냥 넘어간 가능성이 큰 것

문서 전체를 기준으로 보면, 특히 아래 항목은 "해야 한다"고 자주 나왔지만 아직 코드로 안 들어간 대표 항목이다.

1. scripted single-bounce baseline controller
2. success threshold 재보정용 multi-episode distribution 재수집
3. keep-up 기준 reward 추가 재설계
4. random action / `check_env` 수준의 명시적 점검 루프 정리
5. multi-bounce로 가기 위한 curriculum 설계

이 다섯 개는 그냥 문서 메모에만 있고, 현재 저장소 핵심 구현에는 아직 직접 반영되지 않았다.

## 5. 지금 우선순위로 다시 정리하면

현재 코드 상태에서 가장 현실적인 다음 순서는 아래다.

### 5.1 1순위

- scripted keep-up baseline 안정화

이게 필요한 이유:

- 물리 파라미터가 맞는지 확인 가능
- reward가 목표 행동을 포착하는지 확인 가능
- RL 실패 시 원인 분리 가능

### 5.2 2순위

- `run_ee_rollout_analysis.py`와 PPO baseline 로그로 success/contact/failure 분포 다시 보기
- `success_velocity_threshold` 재검토
- reward 항목별 누적 비중 다시 점검

즉 threshold와 reward coefficient를 바로 감으로 바꾸기보다, 분포를 다시 보는 게 먼저다.

### 5.3 3순위

- 이어학습 가능한 PPO baseline으로 single-bounce 성능 올리기

현재는 같은 run-name으로 다시 실행하면 모델을 이어서 학습하는 쪽으로 정리되어 있으므로, 이 단계는 이제 실험 가능하다.

### 5.4 4순위

- multi-bounce reward / curriculum 설계

예:

- 2회 contact bonus
- 3회 contact bonus
- XY offset 확대
- 약한 수평 속도 추가

### 5.5 5순위

- table / net / target landing을 넣는 실제 탁구 task 확장

이건 지금 바로 들어가기보다, single-bounce baseline과 PPO 성능이 어느 정도 잡힌 뒤가 맞다.

## 6. 결론

현재 문서 기준으로 정말 중요한 미완료 항목은 많지 않다. 하지만 중요한 것만 남아 있다.

핵심 미완료 3개:

1. scripted single-bounce baseline
2. reward + success threshold 재정리
3. multi-bounce curriculum 설계

즉 지금 단계에서 가장 먼저 해야 할 일은 새로운 알고리즘을 찾는 것이 아니라,

- baseline을 만들고
- 분포를 다시 보고
- reward를 한 번 더 정리한 뒤
- 그 다음 PPO를 더 길게 돌리는 것

이다.
