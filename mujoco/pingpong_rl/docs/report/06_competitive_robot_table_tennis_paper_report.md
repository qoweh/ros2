# Achieving Human Level Competitive Robot Table Tennis 논문 정리

## 1. 문서 목적

이 문서는 `Achieving Human Level Competitive Robot Table Tennis` 논문과 공개 프로젝트 페이지를 바탕으로, 이 연구가 무엇을 목표로 했고, 왜 그런 구조를 택했으며, 실제로 어떤 절차로 시스템을 만들었는지를 정리한 보고용 문서다.

정리 기준:
- 1차 자료: 로컬 PDF `TableTennisRobot.pdf`
- 보강 자료: 논문 프로젝트 페이지, arXiv 요약 페이지
- 로컬 PDF는 2024년 8월 버전이고, arXiv에는 이후 개정본이 있으나 핵심 주장과 구조는 동일하다.

## 2. 한 줄 요약

이 논문은 `로봇이 탁구공을 가끔 치는 데모`가 아니라, `처음 보는 인간과 실제 경기 규칙에 가깝게 탁구 게임을 할 수 있는 시스템`을 만드는 것이 목표였다. 이를 위해 단일 end-to-end 정책 하나로 해결하지 않고, 아래 세 가지를 같이 만들었다.

- 여러 개의 저수준 타격 스킬
- 상황에 따라 스킬을 고르는 상위 전략기
- 실제 사람과의 플레이 데이터로 학습 분포를 계속 갱신하는 sim-to-real 루프

핵심 메시지는 다음과 같다.

- 탁구는 단순한 contact task가 아니다.
- `공을 맞춘다`와 `경기를 한다` 사이에는 큰 간극이 있다.
- 이 간극을 메우려면 RL만으로는 부족하고, 시스템 구조, 물리 모델링, 데이터 수집 방식, 온라인 적응까지 함께 설계해야 한다.

## 3. 이 연구가 풀려는 문제

### 3.1 단순 리턴이 아니라 경쟁 경기

기존 로봇 탁구 연구는 주로 아래 수준에 머무르는 경우가 많았다.

- 공을 되돌려 보내기
- 특정 위치로 보내기
- 스매시 같은 한 가지 기술 보여주기
- 사람과 협동 랠리 유지하기

이 논문은 여기서 한 단계 더 나아가 아래 문제를 풀려 했다.

- `처음 보는 인간 상대`와
- `실제 물리 환경`에서
- `경쟁 경기`를 수행하기

즉 목표가 `한 번 잘 치기`가 아니라, `상대 스타일에 적응하면서 점수를 주고받는 게임`이었다.

### 3.2 왜 탁구가 어려운가

논문이 탁구를 어려운 문제로 보는 이유는 크게 네 가지다.

1. 저수준 제어가 매우 빠르다.
   - ms 단위 타이밍으로 공을 맞춰야 한다.

2. 고수준 전략이 필요하다.
   - 어디로 보낼지, 얼마나 세게 칠지, 상대 약점을 찌를지 결정해야 한다.

3. 물리 모델이 까다롭다.
   - 공이 가볍고 작아서 공기저항, 회전, 패들 재질, 반발 특성의 영향이 크다.

4. sim-to-real gap이 크다.
   - 시뮬레이터에서 배운 정책을 실제 로봇에 옮길 때, 작은 오차도 미스나 네트 실수로 이어진다.

## 4. 실제 시스템 구성

### 4.1 하드웨어

논문 시스템은 아주 단순한 로봇팔 하나가 아니다.

- 6 DoF ABB IRB 1100 로봇팔
- 테이블 좌우 이동용 x-gantry
- 테이블 앞뒤 이동용 y-gantry
- 125Hz 카메라 기반 공 추적
- 인간 패들 자세를 추적하는 모션캡처 시스템
- 3D 프린트 패들 핸들 + short pips rubber 패들

즉 로봇팔 자체의 자유도만으로는 넓은 테이블 공간을 커버하기 어려워서, gantry를 붙여서 작업공간을 확장했다.

### 4.2 문제 정의

논문은 탁구를 MDP로 두되, episode를 `공 하나를 상대가 친 순간부터 로봇이 그 공을 처리할 때까지`로 잘랐다.

episode 시작:
- 상대 패들이 공을 친 직후

episode 종료:
- 로봇이 공을 성공적으로 반환
- 공이 플레이 밖으로 나감
- 로봇이 공을 놓침

이 설정이 중요한 이유는, 전체 게임을 한 번에 end-to-end로 학습하지 않고 `ball-by-ball decision`으로 분해했기 때문이다.

## 5. 핵심 구조: 계층형 정책

논문의 가장 중요한 설계는 `단일 정책` 대신 `계층형 구조`를 썼다는 점이다.

### 5.1 저수준 컨트롤러 LLC

LLC는 실제 공을 치는 정책들이다. 각 정책은 특정 스타일이나 기술에 특화되어 있다.

예시:
- forehand generalist
- backhand generalist
- left/right 타깃팅 정책
- fast hit 정책
- topspin serve return 정책
- underspin serve return 정책

최종 시스템에는 총 17개의 LLC가 들어갔다.

- serve용 4개
- rally용 13개
- forehand 스타일 11개
- backhand 스타일 6개

### 5.2 왜 여러 LLC로 쪼갰는가

논문은 monolithic policy를 쓰지 않은 이유를 명확히 든다.

1. catastrophic forgetting 회피
   - 잘 배운 스킬을 새 학습 때문에 잃지 않기 위함

2. 확장성
   - 새로운 스킬을 추가할 때 기존 전체 정책을 다시 설계하지 않아도 됨

3. 평가 효율
   - 스킬 하나가 실제로 무엇을 잘하고 못하는지 따로 측정 가능

4. 해석 가능성
   - 로봇이 왜 이 스킬을 골랐는지 설명하기 쉬움

5. 빠른 추론
   - 실제 경기 중 20ms 안에 의사결정을 끝내야 했음

### 5.3 상위 컨트롤러 HLC

HLC는 `이번 공에 어떤 LLC를 쓸지` 고르는 정책이다.

HLC가 사용하는 요소:
- style policy: forehand / backhand 선택
- spin classifier: serve의 topspin / underspin 구분
- skill descriptors: LLC별 능력표
- opponent statistics: 상대가 어디에 약한지, 어떤 공을 잘 받는지
- heuristic strategies: 랜덤, 빠른 공 우선, 먼 곳 우선, 약한 쪽 공략 등
- online preferences(H-values): 실제 경기 중 특정 상대에게 잘 먹히는 LLC에 가중치 부여

즉 HLC는 `그냥 하나 고정된 정책`이 아니라, 오프라인 통계와 온라인 적응을 함께 섞는 전략기다.

## 6. LLC를 어떻게 학습했는가

### 6.1 일반형 정책부터 만들고 세부 기술로 분기

LLC 학습 순서는 대략 이렇다.

1. forehand / backhand generalist를 먼저 학습
2. 그 generalist를 초기값으로 삼아 specialist를 미세조정
3. 성능이 확인되면 skill library에 추가

즉 처음부터 세부 스킬을 따로 학습한 것이 아니라, 넓게 커버하는 기반 정책 위에 세부 기술을 얹었다.

### 6.2 사용한 학습 알고리즘

논문은 PPO나 SAC 대신 BGS(Blackbox Gradient Sensing)를 사용했다.

이유:
- PPO/SAC로 학습한 정책은 실제 로봇에서 더 jerk한 동작을 보였음
- BGS가 더 부드러운 action을 만들어 zero-shot sim-to-real이 잘 됐음

즉 이 논문은 `RL이면 다 PPO`가 아니라, `실제 하드웨어 전이 성능과 동작 smoothness`를 보고 알고리즘을 선택했다.

### 6.3 LLC의 관측과 출력

논문 LLC는 아래 수준의 정보를 본다.

- 최근 8 timestep의 공 위치 / 속도
- 로봇 관절 상태
- 스타일 one-hot

출력은 50Hz joint velocity command다.

여기서 중요한 포인트는, `비전 원본 이미지 -> 바로 action`이 아니라, 이미 추정된 물리 상태를 입력으로 쓴다는 점이다.

## 7. Skill Descriptor: 이 논문의 진짜 핵심 중 하나

논문은 각 LLC가 무엇을 잘하는지 별도 테이블로 저장했다. 이를 `skill descriptor`라고 부른다.

구축 방식:
- 각 LLC를 28k개 ball state에서 여러 번 시뮬레이션
- 아래 메타데이터를 기록
  - land rate
  - hit velocity
  - landing location
  - 분산
- KD-tree 형태 lookup table 구성

의미:
- 현재 incoming ball이 들어오면
- 과거에 비슷한 ball state에서 해당 LLC가 얼마나 잘 쳤는지
- 즉시 조회할 수 있다.

왜 필요한가:
- 상위 정책이 `내가 뭘 잘하는지`를 알아야 전략이 성립하기 때문이다.
- 논문은 이것을 단순 reward history가 아니라 `상황별 능력 모델`로 만든다.

## 8. HLC는 어떻게 LLC를 고르는가

HLC 동작은 대략 아래 순서다.

1. 상대가 공을 치면 1 step 기다린다.
   - 0 step은 공 속도 추정이 불안정
   - 3 step은 너무 늦어 반응 시간이 부족

2. forehand / backhand 스타일을 고른다.

3. serve면 spin classifier로 topspin/underspin을 구분한다.

4. rally면 skill descriptor에서 LLC별 예상 성능을 읽는다.

5. 여러 heuristic이 각자 후보 LLC를 shortlist한다.

6. online preference(H-value)와 offline 성능을 합쳐 최종 LLC를 샘플링한다.

즉 `하나의 큰 neural net이 모든 걸 한 번에 결정`하는 방식이 아니다.

## 9. Sim-to-real을 어떻게 줄였는가

이 논문이 강한 이유는 정책 구조만이 아니라, sim-to-real 설계가 매우 집요하다는 점이다.

### 9.1 물리 모델링

논문이 신경 쓴 요소:

- 로봇 액추에이터 system identification
- 관절 damping / friction / armature
- 공의 fluid dynamics
- 패들 rubber의 spring-damper 근사
- ball-table contact
- ball-paddle contact
- 관측 지연과 노이즈
- domain randomization

즉 `탁구공은 sphere 하나 두면 된다` 수준이 아니다.

### 9.2 topspin / underspin의 비대칭성 처리

논문은 패들 rubber contact가 topspin과 underspin에서 다르게 동작하는 현상을 관찰했다.

그래서 아래 두 단계를 추가했다.

1. topspin correction
   - topspin 관련 접촉 파라미터로 fine-tuning

2. FiLM adapter
   - 원래 action 위에 얇은 adapter layer를 얹어 남은 sim-real gap 보정

즉 회전 문제를 단순 reward shaping으로만 해결하지 않고, 물리 파라미터와 정책 adapter를 같이 썼다.

### 9.3 task distribution을 실제 데이터로 만든다

논문이 특히 강조하는 부분이다.

초기 데이터:
- human-human play 40분
- ball thrower에서 480개 varied throws
- serve 데이터는 별도 수집

이후 루프:
- 현재 정책을 시뮬레이터에서 학습
- 실제 사람과 플레이
- 실제 플레이에서 나온 incoming ball state를 다시 데이터셋에 추가
- 다음 training cycle에 반영

최종 데이터 규모:
- rally 14.2k
- reflection 포함 28.5k
- serve 3.4k

이 접근의 핵심 이유:
- 공의 위치 / 속도 / 회전은 서로 독립이 아니다.
- uniform sampling으로는 실제 인간이 치는 공 분포를 잘 못 만든다.
- 따라서 현실적인 ball state dataset에서 직접 샘플링해야 한다.

논문은 이것을 `automatic curriculum`이라고 본다.

## 10. 실제 경기 배치 방식

학습은 공 하나 단위로 했지만, 실제 경기는 연속적이다. 이 간극을 메우기 위해 논문은 `sub-episode` 개념을 썼다.

- 상대가 공을 치면 하나의 sub-episode 시작
- 로봇이 그 공을 처리하면 종료
- 다음 공 직전에 내부 상태와 로봇 자세를 다시 정리

이것은 완전한 인간식 연속 플레이와는 다르지만, 학습 시 semantics를 배치 시점에도 최대한 보존하기 위한 타협이다.

## 11. 평가 결과

### 11.1 경기 성능

29명의 처음 보는 인간과 경기한 결과:

- 전체 match 승률: 45%
- beginner 상대로: 100%
- intermediate 상대로: 55%
- advanced / advanced+ 상대로: 0%

논문은 이를 `아마추어 intermediate 수준`으로 해석한다.

즉 인간 최고 수준을 이긴 것이 아니라, `인간과 실제로 경기 가능한 수준`에 도달한 것이 핵심이다.

### 11.2 질적 평가

참가자들은 전반적으로 아래 반응을 보였다.

- fun
- engaging
- 다시 플레이하고 싶음

프로젝트 페이지에서도 이 점을 강조한다. 단순히 높은 승률보다, `사람이 실제로 재미있게 느끼는 상호작용`이 중요한 성과로 다뤄졌다.

### 11.3 사람들이 발견한 약점

상대가 가장 많이 파고든 약점:

- underspin
- 아주 낮은 공
- 빠른 공
- 짧은 공
- backhand 약세

논문은 이 약점들을 숨기지 않고, 다음 cycle의 데이터 수집과 학습 목표로 다시 연결한다.

## 12. 한계

논문이 직접 인정한 핵심 한계는 아래와 같다.

- 빠른 공 대응 부족
- 낮은 공 처리 제한
- 극단적인 spin 읽기 어려움
- 완전한 multi-ball 전략 부재
- 모션캡처 의존성
- backhand 성능 열세
- 예측 가능성
- serve 대응 성능이 rally보다 낮음

즉 이 시스템도 `완성형 탁구 AI`가 아니라, `경쟁 스포츠형 로봇 학습의 중요한 이정표`에 가깝다.

## 13. 이 논문이 실제로 보여준 것

이 논문이 설득력 있는 이유는 아래 조합을 동시에 보여줬기 때문이다.

1. RL만 밀어붙이지 않았다.
   - skill library
   - capability model
   - heuristics
   - online adaptation
   - dataset iteration

2. 현실적인 task distribution을 만들었다.
   - `모든 가능한 공`이 아니라 `사람이 실제로 치는 공`을 배웠다.

3. 물리 모델을 탁구 수준까지 신경 썼다.
   - spin, rubber, fluid, latency, contact 모두 중요하게 다뤘다.

4. 진짜 인간 평가를 했다.
   - unseen human 29명과 match를 돌려 intermediate 수준임을 보였다.

## 14. 졸업작품 관점에서 얻어야 할 교훈

### 14.1 `공을 맞추는 RL`과 `탁구를 하는 RL`은 다르다

공이 라켓에 닿는 것만으로는 탁구가 아니다.

탁구에 가까워지려면 최소한 아래가 들어가야 한다.

- table / net
- incoming ball distribution
- target return objective
- spin 또는 최소한 다양한 속도 / 각도
- episode 정의
- 평가 지표

### 14.2 RL보다 먼저 필요한 것이 많다

이 논문은 오히려 아래를 먼저 보여준다.

- scene 설계
- contact calibration
- task decomposition
- baseline policy
- curriculum
- state design

즉 `어떤 알고리즘을 쓸까`보다 `무엇을 학습 문제로 정의할까`가 더 중요하다.

### 14.3 졸업작품에서는 범위를 줄이는 것이 맞다

이 논문 수준은 대규모 시스템 연구다.

졸업작품에서는 아래 정도가 더 현실적이다.

- 단일 incoming ball을 반대편 목표 구역으로 보내기
- 제한된 ball distribution에서 반복 성공률 올리기
- forehand / backhand 또는 target zone 선택 같은 단순 전략기 붙이기

`처음 보는 인간과 경쟁 경기`까지 바로 가는 것은 목표로 삼기엔 너무 크다.

## 15. 추가로 참고할 만한 공개 자료

- 논문 프로젝트 페이지에 하이라이트 영상과 전체 경기 영상이 있다.
- 논문 팀은 ball dataset도 공개했다.
- 같은 팀의 이전 연구로 `Robotic Table Tennis: A Case Study into a High Speed Learning System`, `i-Sim2Real`, `GoalsEye` 등이 연결된다.

즉 이 논문 하나만 보기보다, 아래 흐름으로 보면 더 잘 이해된다.

- 고속 공 추적 / 제어
- single-skill table tennis
- sim-to-real tightening
- competitive match layer 추가

## 16. 최종 정리

이 논문은 `탁구공 치는 로봇`을 만든 논문이 아니라, `계층형 skill library + 현실 데이터 기반 학습 분포 + 온라인 적응`을 묶어서 인간 수준 아마추어 intermediate 경기력을 만든 논문이다.

중요한 것은 RL 자체보다 다음 세 가지였다.

- 어떤 공들을 학습할 것인가
- 어떤 스킬 집합을 만들 것인가
- 실제 경기 중 어떻게 스킬을 고르고 수정할 것인가

따라서 이 논문을 졸업작품 레퍼런스로 사용할 때는 `알고리즘 이름`보다 `문제를 계층적으로 쪼개는 방식`과 `현실적인 training distribution을 만드는 방식`을 먼저 가져오는 것이 맞다.