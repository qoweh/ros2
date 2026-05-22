# 로봇팔 탁구 졸업작품 방향 정리

이 문서는 현재 프로젝트 상태를 기준으로, 아래 질문에 답하기 위해 작성한다.

- 지금 진행 방향이 맞는가
- 로봇을 어떻게 움직이게 해야 하는가
- 어디까지가 강화학습 영역인가
- 라켓 / 공 / 테이블 / 네트를 MuJoCo에서 어떻게 세팅해야 하는가
- 현재 상태에서 무엇을 다음 목표로 잡아야 하는가

## 1. 현재 프로젝트 상태 진단

현재 코드와 문서를 보면, 이 프로젝트는 `탁구 경기 환경`보다는 `라켓-공 접촉과 EE 제어 실험 환경`에 더 가깝다.

현재 확인된 상태:

- `pingpong_rl/assets/scene.xml`
  - floor
  - ball
  - Franka include
  - 즉 top-level scene에는 table / net이 없다.

- `docs/report/01_mujoco_scene_setup_report.md`
  - `초기 table은 제거`라고 명시돼 있다.

- `pingpong_rl/assets/franka/panda.xml`
  - `hand` body 아래에 `racket` body를 직접 붙였다.

- `pingpong_rl/src/pingpong_rl/envs/ee_delta_env.py`
  - reward는 사실상 아래 수준이다.
    - contact bonus
    - 공 높이 보상
    - 실패 패널티
  - success도 `upward_racket_bounce` 기준이다.

- `docs/log/step/03_ee_delta_env_contract.md`
  - 아직 observation에 racket orientation, racket velocity, ball spin이 없다.

즉 현재 단계는 아래처럼 해석해야 한다.

- 지금 하고 있는 것은 `탁구 RL` 자체가 아니라
- `탁구 RL을 하기 전 최소한의 물리/제어 발판`을 만드는 단계다.

이건 잘못된 방향은 아니다. 다만 `지금 상태로 PPO를 오래 돌리면 탁구가 되겠지`라고 기대하면 거의 확실하게 빗나간다.

## 2. 지금 방향이 왜 아직 탁구 학습이 아닌가

현재 환경이 실제로 학습하는 것은 대략 다음이다.

- 라켓 중심 근처에서 공을 맞추기
- 공이 위로 튀게 만들기
- 실패하지 않기

반면 `탁구`에는 최소한 아래가 필요하다.

- 반대편 테이블이 존재해야 함
- 공이 네트를 넘는지 판단해야 함
- 어느 목표 구역에 떨어졌는지 판단해야 함
- incoming ball distribution이 다양해야 함
- 단순 drop이 아니라 날아오는 공을 읽고 맞춰야 함
- spin 또는 최소한 다양한 속도 / 각도 변화가 있어야 함

따라서 현재 프로젝트는 `무의미하다`기보다, `문제가 아직 축소돼 있다`가 더 정확하다.

## 3. 로봇을 어떻게 움직이게 해야 하는가

이 질문에서 가장 중요한 답은 다음이다.

`로봇을 움직이는 것 전체를 강화학습이 다 맡기게 하면 안 된다.`

탁구 같은 문제에서는 보통 아래처럼 역할을 나눈다.

### 3.1 레벨 0: MuJoCo 물리 모델

이 레벨이 담당하는 것:
- 로봇 관절 / 링크
- 라켓 형상과 질량
- 공 질량 / 반경 / 자유운동
- table / net contact
- 필요하면 공기저항과 회전

이 단계는 RL이 아니라 모델링이다.

### 3.2 레벨 1: 저수준 제어기

이 레벨이 담당하는 것:
- desired racket pose를 joint target으로 바꾸기
- IK 또는 Jacobian 기반 EE controller
- 관절 제한 준수
- 지나치게 튀는 motion 억제

현재 프로젝트의 `RacketCartesianController`는 이 레벨의 시작점이다.

이 단계도 보통 RL이 아니라 제어다.

### 3.3 레벨 2: 타격 정책

이 레벨이 담당하는 것:
- 이번 공에 대해 라켓 중심을 어디로 보낼지
- 언제, 어떤 속도로 들어갈지
- 어떤 자세로 칠지

여기서부터 RL을 쓰기 시작하는 것이 일반적이다.

즉 RL이 직접 해야 할 일은 `모든 joint torque를 처음부터 끝까지 알아서 만들어내기`보다,

- target racket position
- target racket velocity
- target hit pose
- skill selection

같은 더 구조화된 action을 내는 쪽이 훨씬 현실적이다.

### 3.4 레벨 3: 전략 / 적응

이 레벨이 담당하는 것:
- forehand / backhand 선택
- 어느 쪽 코스로 보낼지
- 상대 약점을 공략할지
- conservative / aggressive 전략을 바꿀지

이건 탁구를 `경기`로 만들고 싶을 때 필요한 레벨이다.

## 4. 강화학습의 영역은 어디까지인가

짧게 말하면 아래처럼 나누는 것이 좋다.

강화학습이 아닌 것:
- 로봇 MJCF 구성
- 라켓 부착 방식
- 공 / 테이블 / 네트 물성 설정
- 기본 EE controller
- 충돌 검출
- scripted baseline

강화학습으로 해볼 만한 것:
- incoming ball에 대한 target hit position 선택
- target hit timing / approach velocity 선택
- 목표 구역으로 리턴하는 정책
- forehand / backhand 또는 target zone 선택

즉 졸업작품에서 RL은 `문제의 전부`가 아니라 `잘 정의된 상위 decision layer`를 맡기는 것이 맞다.

## 5. 현재처럼 공도 잘 못 맞추면 학습이 되나

현재 사용자의 의문은 타당하다. 답은 아래와 같다.

`지금 상태에서도 학습은 될 수 있지만, 그 학습은 탁구가 아니라 현재 reward가 요구하는 더 쉬운 행동만 학습할 가능성이 크다.`

왜냐하면 현재 env는 본질적으로 다음을 장려한다.

- contact만 만들기
- 공을 위로 띄우기
- 바닥에 떨어지기 전에 뭔가 해보기

하지만 이 reward로는 아래를 직접 배울 이유가 없다.

- 네트를 넘기기
- 반대편 테이블 안에 넣기
- 목표 구역을 맞추기
- 다양한 incoming ball을 처리하기

즉 `학습이 되느냐`보다 `원하는 행동이 학습 목표로 들어가 있느냐`가 핵심이다.

## 6. 그래서 먼저 필요한 것은 scripted baseline이다

강화학습 전에 꼭 필요한 것이 하나 있다.

`아주 제한된 incoming ball에 대해서는 deterministic하게 치는 baseline`

왜 필요한가:

1. 기하가 맞는지 검증할 수 있다.
2. reward 설계가 맞는지 볼 수 있다.
3. table / net / target zone 정의가 맞는지 확인 가능하다.
4. RL이 실패했을 때 알고리즘 문제인지 환경 문제인지 분리할 수 있다.

권장 기준:

- 고정된 3~5개 incoming trajectory에 대해
- 스크립트 제어만으로도
- 일정 확률 이상 반대편 target zone에 넣을 수 있어야 한다.

이 baseline이 없으면 RL은 보통 아래 둘 중 하나로 간다.

- 아무 것도 못 배움
- reward loophole만 찾음

## 7. MuJoCo에서 라켓을 로봇팔에 어떻게 붙일까

선택지는 크게 두 가지다.

### 7.1 현재 방식: hand 자식 body로 직접 부착

장점:
- 가장 단순하다.
- 디버깅이 쉽다.
- hand pose를 그대로 따라간다.

단점:
- robot asset 내부를 수정하게 된다.
- 나중에 라켓 교체 실험을 하려면 분리가 덜 깔끔하다.

현재 프로젝트는 이 방식을 사용하고 있다.

현재 XML 핵심:

- `hand` 아래 `body name="racket" pos="0.005 0 0.1"`
- `racket_head`: 얇은 cylinder contact geom
- `racket_center`: 타격 중심 site

이 방식은 현재 단계에서는 맞다. 아직은 라켓 자산 독립성보다, `공이 어디를 기준으로 맞는가`를 안정적으로 보는 것이 더 중요하기 때문이다.

### 7.2 대안: 별도 라켓 body + site + weld/equality

장점:
- 자산을 더 독립적으로 관리할 수 있다.
- 여러 패들을 쉽게 바꿔볼 수 있다.

단점:
- 초기 정렬과 안정화가 번거롭다.
- 지금 단계에서는 디버깅 포인트만 늘 수 있다.

결론:
- 지금은 direct child 유지가 합리적이다.
- 패들 종류 비교 실험이 필요해질 때 weld 구조를 검토하면 된다.

## 8. 현재 프로젝트의 라켓 세팅은 어떻게 읽어야 하나

현재 라켓 구성은 꽤 좋은 출발점이다.

좋은 점:
- `racket_center` site를 따로 뒀다.
- 손잡이 원점과 타격 중심을 분리했다.
- `racket_head`만 contact를 담당하게 했다.
- `racket_handle_*`, `racket_rim`, `racket_head_back`은 불필요한 contact를 줄였다.

이 구조가 좋은 이유:
- RL과 controller가 `어디를 맞춰야 하는지` 명확하다.
- 디버깅할 때 기준점이 흔들리지 않는다.
- 손잡이 / rim까지 contact가 살아 있으면 학습이 더 불안정해질 수 있다.

현재 수치 예시:
- `racket_head` 반경: `0.084`
- 두께: `0.006`
- 질량: `0.11`
- friction: `0.22 0.001 0.0001`

다만 나중에는 아래를 보강해야 한다.

- 라켓 전체 질량과 관성 재확인
- front/back face 비대칭 모델링 여부
- 공과 접촉 시 rebound calibration

## 9. 탁구공은 시뮬레이션에서 어떻게 세팅해야 하나

현재 프로젝트의 공 세팅은 아주 기본적인 시작점으로는 적절하다.

현재 값:
- sphere
- size `0.02`
- mass `0.0027`
- freejoint

이건 실제 40mm, 2.7g 공과 맞는 편이다.

하지만 탁구로 가려면 앞으로 더 중요해지는 것은 아래다.

1. 선속도뿐 아니라 각속도(spin)
2. 공기저항과 Magnus 효과
3. ball-table bounce calibration
4. ball-paddle bounce calibration

현재처럼 `freejoint + 선속도 reset`만으로는 `공 띄우기`는 가능하지만 `spin 있는 탁구`는 어렵다.

## 10. table, net은 어떻게 세팅해야 하나

탁구를 하려면 결국 table과 net이 필요하다.

최소 권장 구성:

- table top
  - 규격에 맞는 크기와 높이
  - contact geom 분리

- net
  - 충돌 판정용 얇은 geom
  - 시각용 geom과 분리 가능

- robot side / opponent side target zone
  - reward 계산용 site 또는 area 정의

현실 기준으로 시작할 값:
- table size: 약 `2.74m x 1.525m`
- table height: `0.76m`
- net height: `0.1525m`

실무 팁:
- 처음부터 복잡한 mesh를 쓰지 말고 box/capsule/plane으로 시작한다.
- contact 검증이 끝나면 그 다음 visual을 예쁘게 만드는 것이 낫다.

## 11. 현실 물체를 시뮬레이션으로 옮길 때 무엇을 맞춰야 하나

중요도 순으로 보면 보통 아래 순서다.

### 11.1 1차: 기하와 질량

- 공 반경 / 질량
- 테이블 높이 / 크기
- 네트 높이
- 라켓 strike face 위치
- 라켓 전체 질량 중심

### 11.2 2차: contact 반발

- friction
- `solref`
- `solimp`
- 필요하면 damping / compliance

### 11.3 3차: 회전 / 공기역학

- ball spin
- drag
- Magnus lift

### 11.4 4차: 노이즈 / 랜덤화

- 초기 공 위치 / 속도 perturbation
- contact parameter randomization
- 관측 지연 / 노이즈

졸업작품에서는 1차와 2차만 먼저 맞추고, 3차는 선택적으로 올리는 것이 현실적이다.

## 12. 지금 프로젝트에서는 어떤 알고리즘이 맞는가

중요한 것은 알고리즘보다 task redesign이지만, 그래도 선택지를 정리하면 아래와 같다.

### 12.1 PPO 유지

장점:
- 현재 코드에 이미 연결돼 있다.
- smoke test와 인터페이스 검증에 좋다.

단점:
- sample efficiency가 낮다.
- sparse하거나 정교한 contact task에서는 오래 걸릴 수 있다.
- jerk한 정책이 나오기 쉽다.

판단:
- 지금 PPO는 `환경 배선이 맞는지 보는 용도`로는 괜찮다.
- 최종 탁구 정책 후보로 바로 믿기엔 이르다.

### 12.2 SAC 또는 TD3로 전환

장점:
- 연속 action에 보통 더 잘 맞는다.
- sample efficiency가 PPO보다 나은 편이다.

단점:
- reward와 reset 설계가 불안정하면 여전히 잘 안 된다.

판단:
- `target return` 단계부터는 SAC가 더 현실적인 후보일 수 있다.

### 12.3 Scripted baseline + RL 하이브리드

장점:
- 가장 현실적이다.
- 학습 난이도를 크게 낮춘다.
- 논문 구조와도 더 가깝다.

단점:
- 순수 end-to-end RL보다 덜 멋져 보일 수 있다.

판단:
- 졸업작품에서는 이 선택지가 가장 안전하다.

## 13. 추천 커리큘럼

현재 프로젝트에서 바로 full table tennis로 뛰지 말고 아래 순서를 권장한다.

### 단계 1. contact sandbox

목표:
- 고정 drop에서 라켓 중심으로 공 맞추기

현재 프로젝트는 거의 이 단계다.

### 단계 2. scripted incoming ball intercept

목표:
- 미리 정한 포물선 3~5개를 받아치기

필요:
- table / net 추가 전이라도 가능
- target hit point 계산 baseline 필요

### 단계 3. over-table return

목표:
- 반대편 테이블 안 특정 영역으로 보내기

필요:
- table / net 추가
- reward를 target landing 기반으로 변경

### 단계 4. varied launcher policy

목표:
- 속도 / 각도 / 높이가 다양한 incoming ball 처리

필요:
- ball distribution 설계
- observation 확장

### 단계 5. limited rally

목표:
- scripted opponent 또는 launcher와 짧은 랠리 유지

필요:
- multi-ball state 관리
- between-shot reset 전략 재설계

### 단계 6. 전략 레이어

목표:
- forehand / backhand 또는 target zone 선택

필요:
- skill-based 또는 hierarchical policy

## 14. 졸업작품으로 현실적인 목표 선택지

여기서는 선택지를 나눠서 보는 것이 좋다.

### 옵션 A. 단일 incoming ball을 목표 구역으로 리턴

설명:
- scripted ball launcher가 공을 보내면
- 로봇이 반대편 target zone으로 보내기

장점:
- 가장 현실적이다.
- RL 결과를 수치로 보여주기 쉽다.
- 보고서 쓰기가 좋다.

단점:
- `경기` 느낌은 약하다.

### 옵션 B. 제한된 랠리 유지

설명:
- fixed distribution 안에서 2~5회 정도 랠리 유지

장점:
- 탁구다워 보인다.

단점:
- 상태 관리와 안정성이 훨씬 어려워진다.

### 옵션 C. full competitive game 지향

설명:
- 상대, 전략, 점수, 적응까지 포함

장점:
- 가장 인상적이다.

단점:
- 현재 프로젝트 상태와 졸업 일정 기준으로는 너무 크다.

이 문서 기준 권장 판단은 아래다.

- 보고서와 구현을 모두 살리려면 `옵션 A` 또는 `옵션 A + 일부 B`
- 바로 `옵션 C`로 가면 논문 흉내만 내고 실제 성과가 약해질 가능성이 크다.

## 15. 현재 코드 기준으로 가장 먼저 바꿔야 할 것

우선순위를 좁히면 아래 다섯 가지다.

1. table과 net을 scene에 다시 넣기
   - 지금 reward는 탁구가 아니라 upward bounce를 학습한다.

2. scripted baseline 만들기
   - 고정 incoming trajectory에 대해 목표 구역으로 보내는 baseline 필요

3. observation 확장하기
   - racket orientation
   - racket velocity
   - ball spin 또는 time-to-contact 관련 정보

4. reward를 task 중심으로 바꾸기
   - `contact`보다 `return quality` 중심으로 변경
   - 예: net 통과, 반대편 착지, 목표 오차, 미스 패널티

5. 학습 목표를 축소하기
   - full game 대신 `single return`부터 성공률 확보

## 16. 최종 정리

현재 프로젝트는 잘못 가고 있는 것이 아니라, 아직 `탁구 이전 단계`에 있다.

지금 중요한 판단은 이것이다.

- 라켓을 어떻게 붙일까
- 공 / 테이블 / 네트를 어떻게 모델링할까
- controller와 RL의 경계를 어디에 둘까
- 어떤 수준까지를 졸업작품 목표로 삼을까

논문을 기준으로 보면, 탁구를 정말 하려면 RL만 돌리는 것이 아니라 아래를 같이 해야 한다.

- 물리 모델링
- 제어 계층 설계
- 현실적인 ball distribution 설계
- curriculum
- 필요하면 hierarchical policy

따라서 다음 단계는 `PPO 더 오래 돌리기`가 아니라, `탁구 task 자체를 scene와 reward에 다시 넣는 것`이다.