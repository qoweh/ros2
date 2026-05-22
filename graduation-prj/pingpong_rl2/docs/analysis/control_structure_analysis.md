# Control Structure Analysis

## 목적

기존 제어 구조가 왜 학습을 어렵게 만들었는지 정리하고, `pingpong_rl2`에서 어떤 최소 제어 구조를 유지할지 결정한다.

## 기존 구조에서 직접 문제를 만든 부분

### 1. target 누적 방식 자체가 hidden state였다

- action은 EE position delta이다.
- 실제 controller는 현재 racket pose가 아니라 내부 target position을 누적해서 관리한다.
- 이 누적 target은 anchor-relative workspace clipping도 거친다.

즉, policy는 같은 action을 내도 내부 target 상태에 따라 다른 결과를 받는다. 이 자체는 문제는 아니지만, observation이 그 상태를 명확히 설명하지 못하면 학습 난이도가 올라간다.

### 2. pre-contact motion이 너무 자유로우면 팔 몸통 충돌로 바로 간다

- 공이 아직 높고 멀리 있을 때도 controller target이 빠르게 XY/Z로 움직일 수 있었다.
- 그 결과 racket으로 들어가기 전에 link5/link6/link7/hand가 공 경로를 가로막는 일이 많았다.
- 이 문제는 reward보다 control guard 문제였다.

### 3. tracking assist가 “정책 출력”과 “외부 목표”를 섞었다

- controller target 위에 heuristic keep-up target이 다시 섞이면, 최종 EE motion은 policy만의 결과가 아니게 된다.
- 학습이 잘 되든 안 되든 원인을 정책과 assist 사이에서 분리하기 어려워진다.

## 왜 뒤로 빠졌다가 다시 치는 전략이 잘 안 나왔는가

가능한 원인은 아래 네 가지다.

1. strike zone 근처 정렬 reward가 즉시 따라붙기 행동을 선호했다.
2. controller target 누적 상태를 policy가 충분히 복원하기 어려웠다.
3. reset 분포가 좁으면 매번 비슷한 위치에서 즉시 대응하는 정책이 유리했다.
4. assist가 있으면 기다리거나 크게 돌아가는 전략을 policy가 직접 배울 이유가 줄어든다.

즉, “정책이 멍청해서”가 아니라 “구조가 즉시 추종 정책을 더 쉽게 만든 것”에 가깝다.

## `pingpong_rl2` 제어 원칙

### 유지

- `RacketCartesianController`의 Jacobian 기반 작은 IK step
- anchor-relative workspace clip
- pre-contact XY clamp
- pre-contact upward Z cap
- body keep-out

### 제거

- tracking assist target 혼합
- tilt tracking assist
- heuristic controller의 env 내부 사용

## `pingpong_rl2` 최소 action / observation 제안

### action

- 기본 action은 `3D EE delta`만 사용한다.
- tilt는 baseline에서 제거한다.

### observation

baseline에는 아래만 둔다.

- joint positions
- joint velocities
- racket position
- target position
- ball position
- ball velocity
- ball relative position to racket

여기서 핵심은 `target_position`과 `ball_relative_position`을 동시에 관찰하게 해서, 누적 target hidden state와 공 상대 위치를 정책이 함께 해석할 수 있도록 하는 것이다.

## v2에서 검증할 가설

가설: assist 없이도, pre-contact guard + relative observation + 단순 reward만 있으면 policy는 최소한 racket-first contact와 반복 bounce로 향하는 행동을 다시 학습할 수 있다.

이 가설이 틀리면 그때 observation 확장이나 action 구조 변경을 검토한다.