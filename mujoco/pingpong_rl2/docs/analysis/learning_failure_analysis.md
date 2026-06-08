# pingpong_rl Learning Failure Analysis

## 목적

이 문서는 기존 `pingpong_rl`이 왜 안정적인 keep-up 학습으로 이어지지 않았는지, reward 문제만이 아니라 구조 문제까지 포함해서 정리한다. `pingpong_rl2`는 이 분석을 바탕으로 최소 정책 기반 구조만 남기고 다시 시작한다.

## 핵심 결론

기존 실패는 단일 원인이 아니다. 아래 네 가지가 서로 얽히면서 실제 학습 원인을 가렸다.

1. reward가 과하게 많고 서로 상충했다.
2. heuristic tracking assist가 policy 행동에 섞여 들어갔다.
3. controller target guard와 workspace clip이 hidden state처럼 동작했다.
4. reset distribution과 curriculum이 학습 입력 분포를 계속 바꿨다.

즉, “reward를 더 잘 만들면 된다”가 아니라 “정책이 실제로 무엇을 학습하는지 보이지 않는 구조”가 더 큰 문제였다.

## 관찰된 실패 패턴

### 1. 공을 채가 아니라 팔 몸통으로 맞는 경우

- `robot_body_contact`가 초반 step에서 반복적으로 발생했다.
- 많은 episode가 첫 racket contact 이전에 끝났다.
- 이는 reward 부족보다 pre-contact target motion이 너무 공격적이었던 구조와 더 직접적으로 연결된다.

### 2. 낮은 탭이나 한 번 튕기고 바깥으로 보내는 exploit

- 정책은 “계속 높이를 유지”보다 “한 번 유효 contact를 만들고 episode 보상을 챙기는 방향”으로 쉽게 수렴했다.
- 첫 bounce 보상이 강하고, 이후 lateral rebound가 충분히 억제되지 않으면 이 exploit이 자연스럽게 발생했다.
- reward 조정만으로 막으려 하면 다시 초기 contact 학습 자체가 무너지는 부작용이 생겼다.

### 3. ball 아래로 들어가서 기다리는 전략이 잘 나오지 않음

- observation과 reward가 “지금 바로 공 쪽으로 따라가라”에 더 가까운 신호를 줬다.
- controller target은 anchor-relative hidden state로 누적되는데, policy는 이 누적 target의 의미를 직접적으로 분리해서 보기 어려웠다.
- curriculum과 tracking assist가 있으면, 정책이 실제로 retreat/re-approach를 배운 것인지 assist를 따라간 것인지 구분이 안 됐다.

### 4. 파라미터를 바꿔도 결과가 설명되지 않음

- env 기본값을 바꿔도 training entrypoint와 curriculum이 다시 덮어썼다.
- 그 결과 “어떤 변경이 실제로 rollout에 반영되었는가”를 바로 해석하기 어려웠다.

## `pingpong_rl2`에 반영할 구조 원칙

### 남길 것

- MuJoCo scene / physics interface
- Panda racket EE controller
- 최소한의 pre-contact safety guard
- 단순한 position-only action space
- 하나의 명시적 reset distribution

### 제거하거나 기본 비활성화할 것

- tracking assist 기본 주입
- heuristic keep-up controller의 env 내부 혼입
- 기본 curriculum stage mutation
- 다수의 reward term 조합
- tilt action을 기본 학습 경로에 넣는 것

## `pingpong_rl2`의 첫 성공 기준

아래가 먼저 만족되어야 한다.

1. policy 외부 assist 없이 racket first contact를 반복적으로 만든다.
2. body contact가 초반 episode 지배 실패 원인이 아니어야 한다.
3. one-bounce-then-out exploit 없이 반복 bounce가 증가하는지 직접 관찰할 수 있어야 한다.
4. reset distribution과 reward 구성의 실제 적용값을 한 파일에서 설명할 수 있어야 한다.

그 다음에만 reward나 curriculum을 추가 검토한다.