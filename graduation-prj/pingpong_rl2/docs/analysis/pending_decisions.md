# Pending Decisions

## 원칙

아래 항목들은 `pingpong_rl2` baseline에 바로 넣지 않는다. 이유가 충분히 분리되지 않았거나, 구조적으로 다시 얽힐 위험이 있기 때문이다.

## 보류 항목

### 1. heuristic keep-up baseline 재도입 여부

- 디버깅용 scripted baseline은 유용할 수 있다.
- 하지만 env 내부 assist와 섞이면 다시 원인 분리가 어려워진다.
- 필요하면 별도 script로만 유지한다.

### 2. tilt action 재도입 여부

- tilt는 실제로 필요할 수 있다.
- 하지만 baseline에서 action 차원을 늘리면 first contact 학습 자체가 어려워질 수 있다.
- position-only baseline 결과를 본 뒤 판단한다.

### 3. rebound direction shaping 재도입 여부

- outward exploit를 줄이는 데는 도움이 될 수 있다.
- 그러나 first contact 학습을 같이 약화시킨 전례가 있다.
- baseline에서 exploit가 재현될 때만 최소 항으로 다시 검토한다.

### 4. single-bounce-out 전용 terminal penalty 유지 여부

- 현상 대응에는 효과가 있었다.
- 하지만 baseline objective를 오염시킬 수도 있다.
- 먼저 minimal reward로 실제 exploit가 남는지 본다.

### 5. curriculum 재도입 여부

- first contact bootstrap에는 도움이 될 수 있다.
- 동시에 원인 분리를 크게 해친다.
- baseline without curriculum 결과를 먼저 확보한 뒤 단계적으로 붙인다.

### 6. observation 확장 여부

- 현재 baseline은 relative ball position까지만 넣는다.
- 향후 필요하면 racket velocity, predicted intercept, last-contact summary 등을 추가할 수 있다.
- 다만 baseline에서 너무 많은 derived feature를 넣으면 다시 heuristic leakage가 생긴다.

## baseline에서 먼저 확인할 지표

다음 세 가지를 먼저 본다.

1. racket-first contact 비율
2. `robot_body_contact`가 episode 실패에서 차지하는 비율
3. useful bounce count 분포와 one-bounce-out 비율

이 세 지표가 baseline에서 설명 가능하게 나오면, 그 다음 항목을 하나씩 추가한다.