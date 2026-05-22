# EE Task-Space 다음 단계 메모

이 문서는 scene 세팅 이후, EE 기반 강화학습 인터페이스로 넘어가기 전에 바로 결정해야 할 항목만 정리한 메모다.

## 1. 현재 기준점

현재는 다음이 준비되어 있다.

- `racket_center` 기준 spawn/reset
- `racket_center` 기준 contact 확인
- `RacketCartesianController` 기반 위치 IK 경로
- viewer에서 opt-in EE demo 확인 가능

즉 다음 단계의 중심 기준점은 hand가 아니라 `racket_center`로 잡는 편이 자연스럽다.

## 2. action 설계 선택지

### Option A. delta position 3축
- 구성: `dx, dy, dz`
- 장점:
  - 가장 단순하다.
  - 현재 controller와 바로 연결된다.
  - 초기 학습 안정성을 보기 쉽다.
- 단점:
  - orientation을 직접 제어하지 못한다.
  - 강한 스매시나 정교한 면 각도 조절은 어렵다.

### Option B. delta position + orientation
- 구성: `dx, dy, dz` + 회전 항
- 장점:
  - 더 실제 탁구 동작에 가깝다.
  - 라켓 면 각도를 직접 다룰 수 있다.
- 단점:
  - IK/controller가 더 복잡해진다.
  - 초기 불안정성이 커질 수 있다.

## 3. 지금 추천하는 다음 단위

현재 단계에서는 Option A가 더 적절하다.

이유:
- 현재 controller가 이미 위치 3축 기준으로 검증되었다.
- 아직 reward와 termination도 정해지지 않았다.
- orientation까지 같이 열면 디버깅 범위가 너무 넓어진다.

즉 다음 구현 단위는 아래가 적절하다.

1. action = `racket_center` 기준 delta xyz
2. observation = joint + racket_center + ball position/velocity
3. reward/termination 초안 정의

## 4. 다음 구현 목표

다음 코딩 목표는 아래처럼 검증 가능하게 잡는 것이 좋다.

- 목표: `env.step(action)`에서 EE delta action을 받아 `RacketCartesianController`를 통해 racket_center target을 갱신할 수 있게 만들기
- 검증:
  - action을 넣으면 racket_center가 기대 방향으로 이동
  - floor contact 시 termination 또는 reset 조건이 일관됨
  - observation shape가 고정됨
