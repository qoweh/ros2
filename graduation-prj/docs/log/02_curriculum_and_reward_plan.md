# curriculum 및 reward 설계 메모

## 1. 왜 curriculum이 필요한가

현재 task에는 동시에 풀어야 할 요구가 많다.

- 공을 라켓 중심 근처로 맞추기
- 공을 위로 충분히 올리기
- 라켓 면을 수평에 가깝게 유지하기
- 관절을 과도하게 흔들지 않기
- 움직임이 뚝뚝 끊기지 않게 만들기

이걸 처음부터 한 reward 안에 모두 강하게 넣으면, 정책은 `무엇부터 맞춰야 하는지`를 배우기 어려워진다.

그래서 현재 방향은 `처음에는 기본 hit`, `그다음에는 안정성`, `마지막에는 자세와 부드러움` 순서로 나누는 것이다.

## 2. 현재 curriculum stage

### 2.1 bootstrap

목표:

- 일단 공을 위로 치는 최소한의 패턴 만들기

특징:

- reset randomization 제거
- success threshold 완화
- tilt/joint/smoothness 항목 off

### 2.2 stabilize

목표:

- 어느 정도 맞추는 동작이 나온 뒤, 반복 성공률을 높이기

특징:

- 약한 reset randomization 복귀
- 자세 유지와 관절/부드러움 penalty를 약하게 켬

### 2.3 refine

목표:

- 실제로 더 자연스럽고 안정적인 keep-up 행동 만들기

특징:

- full threshold 복귀
- action filter 사용
- tilt/joint/smoothness penalty 강화

## 3. 관절 보상을 왜 late-stage로 미뤘는가

관절 관련 penalty를 너무 일찍 넣으면, policy가 공을 치기 전에 먼저 `안 움직이려는 방향`으로 수렴할 수 있다.

따라서 현재 원칙은 아래다.

- stage 1: 공을 올릴 수 있는가
- stage 2: 그 동작을 반복할 수 있는가
- stage 3: 그 동작이 더 수평하고 덜 튀는가

즉, 관절 penalty는 `기본 동작 형성 이후`에 들어가야 한다.

## 4. 부자연스러운 뚝뚝 끊김을 줄이는 방법

현재 코드에 반영된 방법:

- `action_smoothness_term`
- `action_filter_alpha`
- `joint_motion_term`

추가로 이후 검토 가능한 방법:

- PPO action std 스케줄 조정
- control_dt / n_substeps 재조정
- jerk penalty를 joint acceleration 기준으로 추가
- policy network를 더 작은 action scale로 재학습

## 5. 다음 실험 권장 순서

1. 새 run name으로 curriculum 기본값 유지한 채 재학습
2. 150k~200k마다 렌더로 upward strike 패턴 확인
3. strike는 있는데 motion이 거칠면 smoothness 계열 weight만 미세 조정
4. strike 자체가 약하면 reward보다 controller/threshold를 먼저 다시 확인
