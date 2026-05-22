# Ball Bounce와 EE Delta Env 도움말

이 문서는 이번 단계에서 추가된 두 가지를 설명한다.

- 왜 ball이 이전에는 거의 튀지 않았는가
- `EE delta xyz` 기반 env를 왜 지금 형태로 만들었는가

## 1. 왜 이전 ball은 거의 안 튀었는가

문제는 크기나 질량보다 contact 응답이었다.

현재 프로젝트에서 ball 스펙 자체는 이미 꽤 맞았다.

- radius: `0.02m`
- mass: `0.0027kg`

즉 외형과 질량은 탁구공에 가까웠다.

하지만 floor drop을 해보면, 첫 충돌 뒤 최고점이 거의 바닥 높이 근처에 머물렀다.

원인:
- ball `solref`가 너무 damping-heavy
- floor도 기본 plane 상태라 rebound 성향이 약함
- 두 geom의 조합 결과가 “가벼운 공”이 아니라 “잘 안 튀는 공”처럼 보였음

## 2. 이번에 어떤 식으로 조정했는가

이번 단계에서는 완전 현실적인 탁구 physics를 바로 넣지 않았다.

대신 목표를 이렇게 잡았다.

- floor에 닿으면 눈에 띄는 rebound가 있어야 함
- 그래도 지나치게 chaotic한 high-bounce는 피해야 함

그래서 pure floor drop 기준으로 여러 조합을 직접 비교했고, 그중 moderate bounce에 해당하는 값을 선택했다.

현재 선택:
- ball
  - `friction="0.02 0.001 0.0001"`
  - `solref="0.001 0.08"`
- floor
  - `friction="0.2 0.005 0.0001"`
  - `solref="0.0006 0.03"`

이 값으로 1m floor drop 기준 post-impact peak가 약 `0.371m`까지 올라온다.

즉 완전 현실적인 탁구공은 아니지만, 이전처럼 바닥에 붙는 수준은 벗어났다.

## 3. 왜 여기서 멈췄는가

더 강하게 튀게 만들 수도 있었다.

예를 들어 더 낮은 damping 조합은 1m drop에서 `0.5m` 이상 또는 그 이상으로도 튀었다.

하지만 지금 단계에서 그 값을 바로 쓰지 않은 이유는 다음과 같다.

- 제어기 디버깅이 더 어려워진다.
- reward/termination이 아직 고정되지 않았다.
- orientation도 아직 안 들어간 상태라 원인 분리가 어려워진다.

즉 이번 선택은:
- realism 최우선
이 아니라
- control architecture 안정화 우선

이라는 현재 프로젝트 방향에 맞춘 타협값이다.

## 4. EE delta env는 왜 새 파일로 분리했는가

선택지는 두 가지였다.

- 기존 `PingPongSim`에 RL용 action/reward/termination까지 전부 섞기
- low-level sim은 유지하고, 그 위에 얇은 task env를 하나 더 두기

이번에는 두 번째를 선택했다.

이유:
- `PingPongSim`은 scene/reset/contact 같은 low-level 역할만 유지할 수 있다.
- RL 인터페이스 실험을 해도 기존 시뮬레이터를 덜 건드린다.
- observation/action/reward를 이후에 바꾸기 쉽다.

현재 새 env는 `PingPongEEDeltaEnv`다.

## 5. 현재 action 계약

현재 action은 orientation 없이 위치 3축만 쓴다.

형태:
- `(dx, dy, dz)`

현재 의미:
- `racket_center` target 위치에 더할 delta position
- 1 step당 `action_limit=0.03m`로 clip

즉 지금은 “라켓 중심점을 얼마나 이동시키고 싶은가”만 다룬다.

## 6. 왜 observation은 flat vector로 지금 고정했는가

이번에는 public env observation을 dict가 아니라 flat vector로 고정했다.

이유:
- 현재 observation 항목이 모두 고정 길이 연속값이다.
- 곧 RL에 붙일 인터페이스라면 마지막에 한 번 더 flatten하는 비용만 남긴다.
- MLP baseline과 Gymnasium 스타일 wrapper에 바로 연결하기 쉽다.

현재 순서:
- `joint_positions` `(7,)`
- `joint_velocities` `(7,)`
- `racket_position` `(3,)`
- `target_position` `(3,)`
- `ball_position` `(3,)`
- `ball_velocity` `(3,)`

즉 최종 flat observation은 `(26,)`이다.

## 7. 왜 `target_position`을 observation에 넣었는가

처음에는 `racket_position`과 `ball_position`만 있으면 될 것처럼 보일 수 있다.

하지만 현재 action은 `controller.target_position`에 delta를 누적하는 방식이다.

즉 controller 내부에는 다음 step의 거동에 영향을 주는 상태가 하나 더 있다.

그 값이 `target_position`이다.

이걸 observation에 넣지 않으면:
- 외부에서 보기엔 같은 observation처럼 보여도
- 내부 controller target이 다르면
- 다음 transition이 달라질 수 있다.

그래서 이번에 flat vector로 바꾸면서 `target_position`도 같이 넣어 RL 기준 상태를 닫았다.

사람이 보기 쉬운 형태가 필요하면 코드에서는 다음 helper를 쓰면 된다.

- `observation_dict()`
- `unflatten_observation()`

## 8. 현재 reward/termination은 어떤 수준인가

reward는 아직 draft다.

현재 항목:
- racket contact bonus
- ball이 racket보다 위에 있을 때의 작은 height term
- floor/failure penalty

반환 형식은 Gymnasium 스타일로 맞췄다.

- `reset() -> (observation, info)`
- `step() -> (observation, reward, terminated, truncated, info)`

termination은 현재 `failure_reason()`에 맞춘다.

그리고 success termination도 하나 넣었다.

현재 규칙:
- `ball_geom`와 `racket_head`가 contact이고
- 그 순간 `ball_velocity_z > success_velocity_threshold`
- 이면 `success_reason = upward_racket_bounce`

기본 threshold는 `0.5`로 두었다.

이 값은 현재 기본 scene에서 첫 라켓 contact 시 측정한 `ball_velocity_z`가 약 `0.81` 수준인 것을 기준으로, 단순 스침보다 조금 더 보수적으로 잡은 초기값이다.

그리고 이제는 time limit도 env 내부에 같이 넣었다.

- 기본 `max_episode_steps = 300`
- `reset()`에서 `step_count = 0`
- `step()`마다 `step_count += 1`
- failure가 없고 `step_count >= max_episode_steps`면 `truncated = True`

즉 현재는:
- 실패 종료는 `terminated`
- 성공 종료도 `terminated`
- 시간 종료는 `truncated`

로 분리된다.

디버그와 로깅을 위해 `info`에는 아래도 넣는다.

- `step_count`
- `time_limit_reached`
- `success_reason`

즉 지금 단계의 목적은 “학습이 잘 되게 reward를 완성”이 아니라, `env.step(action)` 형태가 실제로 성립하는지 고정하는 것이다.
