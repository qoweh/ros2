# 경기형 탁구 논문에서 upward bounce RL로 가져갈 것

## 1. 이 문서의 목적

이 문서는 `Achieving Human Level Competitive Robot Table Tennis` 프로젝트를 `참고만` 하되, 내 실제 목표인

- 로봇팔 끝의 탁구채로
- 탁구공을 계속 위 방향으로 쳐서
- 가능한 오래 공중에 유지하는 강화학습

으로 바꿔서 볼 때, 저 프로젝트에서 무엇을 가져가야 하고 무엇은 과감히 버려야 하는지 정리한다.

핵심 전제:
- 내 목표는 `경기`가 아니다.
- 내 목표는 `반복 바운스 제어`다.
- 따라서 상대방, 네트, 서브, 전략, 사람 적응은 지금은 중심이 아니다.

## 2. 가장 중요한 결론

저 프로젝트에서 가져가야 하는 것은 `탁구 경기 규칙`이 아니라 `문제를 푸는 방식`이다.

가져갈 것:
- 문제를 작게 쪼개는 방식
- 학습 전에 제어와 물리를 먼저 정리하는 방식
- 좁은 분포에서 시작해 점점 어렵게 만드는 커리큘럼
- reward보다 별도 지표와 failure 분류를 같이 보는 습관
- headless 학습과 별도 render/eval 루프를 분리하는 방식

버릴 것:
- 인간 상대 경기 설계
- HLC / LLC 전체 구조
- 상대 약점 공략, 서브 처리, spin classifier
- 대규모 human-human 데이터셋 루프
- zero-shot sim-to-real 중심 설계

즉 네 과제에서는 `논문의 목적`이 아니라 `논문의 개발 태도`를 가져가는 편이 맞다.

## 3. 저 프로젝트에서 실제로 얻어갈 수 있는 것

### 3.1 순수 end-to-end RL 하나로 밀지 않았다

논문은 거대한 정책 하나에게 모든 걸 맡기지 않았다.

- 물리 모델링을 먼저 다듬고
- 저수준 제어 레이어를 만들고
- 그 위에 정책을 올리고
- 다시 실제 성능을 별도 통계로 측정했다.

이 점은 네 과제에도 그대로 적용된다.

네 목표가 upward bounce라면, 정책이 배워야 하는 것은

- 공이 어디로 오고 있는지 보고
- 라켓 중심을 어디로 얼마나 움직일지 정하고
- 적절한 타이밍으로 공을 위로 다시 보내는 것

이어야지,

- Franka 7개 관절의 모든 움직임을 처음부터 끝까지 무에서 유로 발견하는 것

이어서는 안 된다.

### 3.2 학습 분포를 처음부터 좁게 잡아야 한다

논문은 `모든 가능한 탁구공`을 한 번에 학습시키지 않았다. 실제 사람이 자주 치는 공 분포를 모아서 시작했고, 그 분포를 점점 넓혔다.

이 사고방식은 upward bounce 과제에 더 중요하다.

좋은 시작 분포:
- 공이 라켓 중심 위에서 거의 수직으로 떨어짐
- 초기 XY 오프셋이 매우 작음
- 초기 속도도 작음
- spin 없음

나쁜 시작 분포:
- 높은 높이, 큰 XY 편차, 다양한 속도, 다양한 각도, spin까지 한 번에 랜덤화

네가 지금 감을 못 잡는 큰 이유 중 하나가 바로 이것일 가능성이 높다. 학습 문제가 너무 넓으면 정책은 `첫 성공 경험` 자체를 거의 못 만든다.

### 3.3 reward만 보지 말고 별도 성공 지표를 둬야 한다

논문은 단순 reward 합계만 보지 않았다.

- return rate
- landing location
- hit velocity
- LLC별 강점/약점
- 사람 상대 승률

등을 따로 봤다.

네 과제에도 같은 습관이 필요하다. upward bounce에서는 아래 지표가 reward보다 중요하다.

- 첫 contact까지 걸린 step 수
- contact 이후 `ball_velocity_z`
- 최고 높이
- 연속 contact 횟수
- episode 길이
- floor contact 비율
- out-of-bounds 비율

즉 `reward가 올랐다`보다 `진짜로 몇 번 연속으로 튕겼는가`가 더 중요하다.

### 3.4 failure를 명시적으로 분류해야 한다

논문은 성능이 안 나오는 이유를 단순히 `학습 실패`로 뭉개지 않았다.

네 과제에서도 아래 failure 구분이 필요하다.

- 공이 라켓에 안 맞음
- 맞았지만 위로 안 감
- 너무 옆으로 빠짐
- 바닥에 닿음
- 수치 폭주
- 에피소드 시간 제한 종료

지금 저장소의 `failure_reason()` 구조는 이 방향으로 가는 좋은 출발점이다. 이건 저 논문에서 가져갈 수 있는 아주 좋은 습관이다.

### 3.5 학습 루프와 시각화 루프를 분리해야 한다

논문은 실제 경기와 시뮬레이션 학습을 분리했다. 네 과제에서도 마찬가지다.

가져갈 원칙:
- 학습은 빠르게, headless로
- 확인은 짧게, viewer로
- 결과는 로그와 체크포인트로 남기기

즉 `학습시키는 걸 실시간으로 계속 보면서 디버깅`하는 방식은 직관은 좋지만, 실제로는 학습 속도를 심하게 깎고 viewer 입력 문제까지 섞여서 더 헷갈리게 만든다.

### 3.6 커리큘럼이 필요하다

논문은 data cycle을 돌리며 자동 커리큘럼을 만들었다. 네 과제에서도 더 작은 형태의 커리큘럼이 필요하다.

예시:

1. 공이 라켓 중심 바로 위에서 수직 낙하
2. XY 오프셋 조금 추가
3. 약한 수평 속도 추가
4. 공이 더 높이 또는 더 낮게 시작
5. 라켓 자세나 공 속도 범위 확대
6. 여러 번 연속 바운스를 목표로 reward 변경

이 순서를 건너뛰고 처음부터 전부 랜덤화하면 학습은 거의 막힌다.

## 4. 저 프로젝트에서 가져가면 안 되는 것

### 4.1 경기용 상위 전략기

논문의 HLC는 `어떤 스킬을 고를지` 결정하는 상위 전략기다. 네 과제에서는 지금 필요 없다.

이유:
- 목표가 한 가지다.
- 상대가 없다.
- 타격 스타일을 고를 상황도 거의 없다.

지금은 HLC가 아니라 `single bounce controller + single bounce policy`가 맞다.

### 4.2 serve / topspin / underspin 처리

논문은 topspin, underspin, serve return까지 다뤘지만, upward bounce 과제에서 이것은 너무 이르다.

지금은 아래로 제한하는 편이 좋다.

- no spin
- near-vertical drop
- fixed racket orientation 또는 아주 제한된 orientation

spin은 나중에 공을 일부러 까다롭게 만들고 싶을 때만 추가하면 된다.

### 4.3 사람 데이터 기반 task distribution

논문은 인간과 실제 경기를 했기 때문에 사람 데이터가 필수였다. 네 과제는 simulation-only upward bounce라서 그 정도 데이터 루프는 필요 없다.

대신 네 과제에 맞는 더 작은 분포 설계가 중요하다.

- hand-crafted initial state sampler
- curriculum schedule
- failure bucket별 재샘플링

이 정도면 충분하다.

### 4.4 sim-to-real 최적화 집착

논문은 real hardware로 가야 했기 때문에 sim-to-real gap이 중심 문제였다. 현재 네 목표는 simulation-only라고 했으니 우선순위가 다르다.

당장 필요 없는 것:
- perception latency modeling
- real paddle motion capture
- zero-shot transfer tricks
- online opponent adaptation

당장 필요한 것:
- simulation 자체가 안정적이어야 함
- reward와 observation이 목표와 맞아야 함
- 공이 실제로 반복 바운스 가능한 물리 파라미터를 가져야 함

## 5. 네 과제에 맞게 번역하면 무엇이 남는가

논문에서 네 과제로 번역해서 남는 핵심은 딱 다섯 가지다.

### 5.1 action space는 물리적으로 의미 있게 줄여라

논문은 고속 탁구를 풀면서도 완전한 raw action 탐색보다 구조화된 정책을 썼다.

네 과제에서는 아래 중 하나가 현실적이다.

- 라켓 중심 delta position
- 라켓 중심 target velocity
- 고정 orientation + delta xyz

반대로 처음부터 raw joint torque나 raw joint target 7개를 직접 학습시키는 것은 너무 어렵다.

### 5.2 baseline 없이 RL만 돌리지 마라

논문도 결국 정책의 능력을 별도로 측정했다. 네 과제에서는 scripted baseline이 더 중요하다.

필수 baseline:
- 고정 drop에서 공을 한 번은 위로 다시 보내는 스크립트

좋은 baseline이 있어야 하는 이유:
- 물리 문제인지
- 제어 문제인지
- reward 문제인지
- RL 문제인지

를 분리할 수 있다.

### 5.3 reward보다 success contract를 먼저 고정하라

논문은 공이 실제로 return됐는지 같은 명시적 성능 기준을 봤다. 네 과제도 아래 기준을 먼저 정해야 한다.

예시:
- success A: 첫 contact 후 `ball_velocity_z > threshold`
- success B: 최고 높이가 일정 기준 이상
- success C: floor contact 전 두 번째 contact 발생
- success D: 연속 N번 bounce

reward는 이 success contract를 보조하는 도구여야 한다.

### 5.4 작은 범위에서 signs of life를 먼저 봐라

논문의 강점은 복잡한 목표를 작게 쪼갠 것이다. 네 과제도 마찬가지다.

첫 번째 signs of life는 아래면 충분하다.

- 고정 drop에서
- 10 episode 중 몇 episode에서
- 첫 contact 후 공이 위로 튀는가

여기서 아직 안 되는데 `계속 하늘 방향으로 오래 유지`를 바로 학습하려고 하면 거의 막힌다.

### 5.5 로그와 재현성을 먼저 잡아라

논문은 평가와 기록이 촘촘했다. 네 과제에서도 아래는 초기에 잡는 것이 좋다.

- training summary JSON
- episode CSV
- contact CSV
- rollout analysis script
- fixed seed smoke runs

이건 지금 저장소에 이미 일부 들어가 있으니, 좋은 방향이다.

## 6. 현재 저장소 기준으로 가장 직접적인 적용점

현재 저장소에서 upward bounce 기준으로 유지할 것:

- `PingPongSim`
- `RacketCartesianController`
- `PingPongEEDeltaEnv`
- bounce baseline 스크립트
- rollout / reward / contact logging 구조

현재 저장소에서 바꿔야 할 것:

- reward를 `height bonus` 위주에서 `multi-bounce objective` 쪽으로 점진 전환
- observation에 필요한 경우 racket velocity, orientation 추가
- scripted bounce controller 추가
- PPO만 보지 말고 SAC도 비교
- episode curriculum 설계

## 7. 최종 정리

`Achieving Human Level Competitive Robot Table Tennis`에서 네가 가져가야 하는 것은 `복잡한 경기 시스템`이 아니다.

정확히는 아래다.

- 학습 문제를 작게 자르는 방식
- 제어 레이어와 정책 레이어를 분리하는 방식
- 좁은 초기 분포에서 시작하는 방식
- reward와 별개로 명시적 성공 지표를 보는 방식
- headless training + separate render/eval 방식
- failure를 이유별로 분류하고 로그를 남기는 방식

반대로 지금은 가져오지 말아야 하는 것도 분명하다.

- 상대 전략
- serve / spin 대응
- HLC / LLC 전체 구조
- 대규모 사람 데이터 루프
- sim-to-real 최적화

즉 지금 네 과제의 핵심은 `탁구를 경기로 만드는 것`이 아니라 `반복 바운스가 가능한 좁고 안정적인 학습 문제를 만드는 것`이다.