# Reward Dependency Analysis

## 목적

기존 `pingpong_rl`의 reward가 왜 문제 분리를 어렵게 만들었는지 정리하고, `pingpong_rl2`에서 무엇을 줄일지 결정한다.

## 기존 reward 구조의 문제

기존 env는 다음 성격의 reward를 동시에 사용했다.

- contact bonus
- success bonus
- bounce progress bonus
- height target shaping
- lift shaping
- tracking alignment shaping
- contact centering shaping
- lateral rebound shaping
- rebound direction shaping
- active hit shaping
- orientation penalty
- joint motion penalty
- action smoothness penalty
- terminal failure penalty
- single-bounce-out terminal penalty

문제는 각 항이 individually 잘못되었다기보다, policy가 무엇 때문에 움직였는지 분리하기 어렵다는 점이다.

## 왜 학습 해석이 어려웠는가

### 1. reward가 contact 이전과 이후를 동시에 강하게 규정했다

- contact 이전에는 tracking / centering / preparation 계열이 작동한다.
- contact 시점에는 active hit / contact / lift / direction 계열이 동시에 작동한다.
- 실패 시점에는 terminal penalty가 들어간다.

이 구조에서는 “공 아래로 이동한 이유”, “위로 올려친 이유”, “한 번 치고 밖으로 보낸 이유”를 각각 분리해서 보기 어렵다.

### 2. 첫 성공 보상이 policy를 너무 쉽게 편향시켰다

- 첫 bounce 보상이 크면, 이후 episode 전체를 망가뜨리더라도 첫 contact 하나만 확보하는 방향이 강화된다.
- 실제로 useful first bounce 뒤에 lateral exit가 반복 exploit으로 나타났다.

### 3. 실패 패턴에 따라 reward를 계속 덧붙이는 방식이 구조를 더 불투명하게 만들었다

- body contact가 많으면 pre-contact shaping을 손본다.
- outward exit가 많으면 rebound penalty를 추가한다.
- low tap이 많으면 success 조건과 height shaping을 더 강화한다.

이 방식은 국소 문제 대응에는 도움이 되지만, 장기적으로는 학습 objective를 사람이 추적하기 어렵게 만든다.

## `pingpong_rl2` 최소 reward 원칙

### 기본 원칙

- contact 이전 shaping은 한 종류만 둔다.
- “유용한 upward racket contact”를 핵심 이벤트로 둔다.
- target height 근접 여부는 success event의 품질 보정 정도로만 쓴다.
- failure penalty는 유지하되, exploit 전용 terminal penalty는 기본 세트에 넣지 않는다.

### v2 baseline reward 초안

`pingpong_rl2` baseline은 아래 정도로 제한한다.

1. 매 바운스의 descending strike window 안에서의 XY/height alignment reward
2. useful upward contact bonus
3. projected apex가 target height에 가까울수록 주는 작은 품질 보정 reward
4. floor / body contact / out-of-bounds failure penalty

이 네 가지면 “공 아래로 들어간다 -> 적절한 순간 contact를 만든다 -> 목표 높이 근처로 올린다”라는 최소 과정을 유지할 수 있다. 첫 contact 이후에도 같은 strike-window alignment를 유지해야 반복 바운스 학습 신호가 사라지지 않는다.

## 기본 baseline에서 의도적으로 뺄 항목

- tracking assist 기반 reward coupling
- tilt 관련 reward
- joint motion penalty
- action smoothness penalty
- rebound direction bonus/penalty
- single-bounce-out 전용 terminal penalty

이들은 필요하면 다시 넣을 수 있지만, baseline 해석 가능성을 먼저 확보해야 한다.

## 재도입 조건

아래 중 하나가 명확히 관찰될 때만 추가한다.

1. 반복 bounce는 생기지만 특정 failure mode가 지배적으로 남는다.
2. minimal reward로는 useful contact 자체가 거의 발생하지 않는다.
3. exploit가 반복되며, 해당 exploit를 막는 가장 작은 reward term이 명확하다.

그 전에는 reward를 더 늘리지 않는다.
