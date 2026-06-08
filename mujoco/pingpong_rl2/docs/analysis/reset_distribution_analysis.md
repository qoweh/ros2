# Reset Distribution Analysis

## 목적

기존 프로젝트에서 reset 분포가 왜 제대로 해석되지 않았는지 정리하고, `pingpong_rl2`에서는 reset이 어디서 정의되는지 명확하게 고정한다.

## 기존 문제

### 1. env 기본값과 training entrypoint 기본값이 달랐다

- env에서 `reset_xy_range`를 바꿔도 training script가 별도 기본값을 넘기면 실제 rollout에는 반영되지 않는다.
- 이 상태에서 실험 로그만 보면 어느 값이 실제로 적용됐는지 즉시 알기 어렵다.

### 2. curriculum이 reset 분포를 다시 바꿨다

- run 시작 시점의 값과 중간 학습 시점의 값이 다를 수 있었다.
- 따라서 실패 원인이 reward인지, policy capacity인지, reset widening인지 분리하기 어려웠다.

### 3. 지나치게 좁은 초기 분포는 즉시 대응 정책을 강화한다

- 거의 같은 XY 위치에서만 공이 떨어지면, 뒤로 빠졌다가 다시 들어오는 전략을 학습할 유인이 약하다.
- 반대로 너무 넓은 분포를 처음부터 쓰면 first contact 자체가 무너질 수 있다.

## `pingpong_rl2`의 기본 원칙

### 1. reset 분포는 env config 한 곳에서만 정의한다

- baseline에서는 curriculum으로 reset을 바꾸지 않는다.
- training script는 전달한 값을 그대로 기록만 한다.

### 2. 기본 분포는 너무 좁지도 너무 넓지도 않게 둔다

baseline 시작점:

- `ball_height`: 고정 기본값 유지
- `reset_xy_range`: small-but-nonzero
- `reset_velocity_xy_range`: very small
- `reset_velocity_z_range`: small downward or near-zero band

핵심은 매번 같은 수직 낙하만 보여주지 않되, first contact가 완전히 무너질 정도로 넓히지 않는 것이다.

### 3. seed와 실제 적용값을 반드시 저장한다

- 각 run summary에 `seed`, `n_envs`, `reset_xy_range`, `reset_velocity_xy_range`, `reset_velocity_z_range`를 남긴다.
- 그래야 실험 결과를 다시 재현할 수 있다.

## `pingpong_rl2` 기본 reset 방향

현재 baseline은 아래 방향이 적절하다.

1. curriculum 없음
2. 명시적 `reset_xy_range` 사용
3. 약한 XY/Z velocity randomization 유지
4. 모든 vector env에 `base_seed + env_index` 적용

이렇게 해야 멀티 env에서도 분포가 서로 얽히지 않고, 단일 env 디버깅 결과와 비교가 가능해진다.

## 추가 실험이 필요한 지점

아래는 아직 baseline에 고정하지 않는다.

- staged curriculum 재도입 여부
- spawn height range를 넓힐지 여부
- launch velocity를 더 공격적으로 섞을지 여부

이 항목들은 baseline 학습 결과를 본 뒤 결정한다.