# v19 Presentation Readiness And Future Roadmap

## 한줄 결론

v19는 "좁은 초기 조건에서 탁구공을 여러 번 위로 쳐 올리는 RL self-rally 프로토타입"으로는 발표 가능한 수준이다. 다만 "시작 위치와 높이에 관계없이 안정적으로 무한 저글링하는 완성 모델"이라고 말하기에는 아직 부족하다.

발표 전략은 v19/v20 중 더 좋은 모델을 "champion model"로 고르고, 정량 분석과 실패 모드 분석을 같이 보여주는 쪽이 가장 안전하다. 4학년 소프트웨어전공 졸업작품 관점에서는 RL 환경 설계, MuJoCo 시뮬레이션, PPO 학습, reward/action 구조 개선, 분석 자동화, 웹 배포까지 묶으면 충분히 설득력 있는 범위다. 대신 시연은 정직하게 "현재 가능한 범위"와 "향후 확장"을 분리해야 한다.

## v19를 어느 정도 괜찮은 모델로 볼 수 있는가

v19는 완성 모델은 아니지만, 무의미한 실패 모델도 아니다. v18 대비 중요한 개선이 있다.

| 항목 | v18 | v19 |
| --- | ---: | ---: |
| max useful bounces | 8 | 9 |
| `ball_out_of_bounds` | 30/100 | 8/100 |
| next-intercept reachable rate | 0.726 | 0.863 |
| upward contact below 0.20m | 0.505 | 0.380 |
| mean outgoing xy error | 0.093 | 0.055 |

특히 `time_limit` episode는 실패라기보다 600 step horizon까지 살아남은 episode다. v19에서 `time_limit` 22개 episode는 평균 useful bounces가 `5.045`, 최대 `9`였다. 즉 viewer에서 time_limit episode를 잡으면 "계속 치고 있다"는 느낌의 데모가 나올 수 있다.

하지만 전체 100 episode 기준으로 보면 아직 불안정하다.

- `low_apex_contact`: 69/100
- `ball_out_of_bounds`: 8/100
- `floor_contact`: 1/100
- terminal upward contact below 0.20m: 0.753

따라서 v19의 표현은 이렇게 잡는 것이 좋다.

- 좋은 표현: "좁은 초기 조건에서 안정적인 self-rally를 학습하는 데 성공했고, 일부 episode는 time-limit까지 유지된다."
- 위험한 표현: "탁구공을 어떤 위치에서도 계속 안정적으로 저글링한다."

## 졸업작품 발표 기준에서의 판단

7일 미만 남은 상황에서는 "더 완벽한 RL"보다 "보여줄 수 있는 완성도"가 중요하다.

현재 프로젝트는 졸업작품으로 볼 때 강점이 분명하다.

- 실제 물리 시뮬레이터 MuJoCo 기반이다.
- 로봇팔, 탁구공, 라켓, 접촉 물성, 반발, 실패 조건이 있는 환경이다.
- 단순 supervised demo가 아니라 PPO 강화학습으로 정책을 학습했다.
- action dimension을 5D, 8D, 11D, 13D로 확장하며 병목을 분석했다.
- reward shaping, low-apex termination, next-intercept reachability, lateral stability를 실험적으로 개선했다.
- 각 모델의 실패 이유를 CSV/JSON으로 분석하는 파이프라인이 있다.
- 웹사이트 배포를 붙이면 결과 확인성과 발표 완성도가 올라간다.

그래서 발표 수준 자체는 괜찮다. 다만 발표에서 "완성된 스포츠 로봇"처럼 보이려 하기보다, "강화학습 기반 로봇 제어 환경을 만들고, 안정적 공 튕기기를 향해 반복 개선한 시스템"으로 프레이밍하는 게 좋다.

추천 발표 구조:

1. 문제 정의: 라켓으로 탁구공을 적절한 높이와 방향으로 계속 띄우기
2. 환경: MuJoCo 로봇팔, 탁구공, 라켓, 접촉/반발, 실패 조건
3. 정책: PPO + contact-frame residual action
4. 개선 과정: v17 action std 문제, v18 13D 확장, v19 low-apex 보상/종료 개선
5. 결과: v19에서 out-of-bounds 감소, next-intercept reachable 증가, time-limit episode 존재
6. 한계: low-apex collapse와 임의 초기 위치 일반화는 아직 남음
7. 웹 서비스: 모델/분석 결과/시연 영상/학습 로그를 보여주는 데모 플랫폼

## v20에 대한 현실적 기대

v20은 v19를 완전히 다른 모델로 바꾸는 것이 아니라, v19의 남은 실패 모드 중 `ball_out_of_bounds`와 terminal low-apex collapse를 줄이는 fine-tune이다.

기대할 수 있는 개선:

- `ball_out_of_bounds`가 8/100에서 더 줄어듦
- 바깥쪽 x drift episode에서 라켓 lateral speed가 낮아짐
- terminal low-apex 비율이 일부 낮아짐
- viewer에서 공이 밖으로 튀는 episode가 줄어듦

기대하기 어려운 것:

- 갑자기 모든 episode가 time_limit이 되는 수준
- 임의 xy/z 시작 위치 일반화
- 완전히 일정한 높이의 무한 저글링

따라서 v20 결과가 v19보다 아주 드라마틱하지 않아도 이상하지 않다. v20은 "최종 대전환"이 아니라 "남은 실패 모드 하나를 더 줄이는 구조적 보정"으로 보는 게 맞다.

모델 선택 기준:

- v20이 `ball_out_of_bounds`를 줄이고 mean useful/stable을 크게 망치지 않으면 v20 사용
- v20이 너무 보수적으로 변해서 low-apex나 짧은 episode가 늘면 v19를 champion model로 유지
- 발표용 영상은 v19와 v20 둘 다 viewer로 보고 더 안정적인 checkpoint를 선택

## 임의 시작 위치에서 계속 칠 수 있는가

가능성은 있다. 하지만 v19를 그대로 두고 시작 위치만 "동서남북 0~360도, 다양한 xy/z"로 바꾸면 잘 될 가능성은 낮다.

현재 v19의 초기 조건은 매우 좁다.

- `reset_xy_range=0.028`
- `reset_ball_height_range=0.02`
- 초기 lateral velocity 거의 없음

즉 현재 정책은 "거의 비슷한 위치에서 떨어지는 공을 안정화하는 문제"를 배운 것이다. 임의 시작 위치는 다른 문제다. 특히 첫 타격이 어려워진다. 첫 타격이 안정적으로 성공해 공을 anchor 근처의 적정 apex로 되돌리면 이후에는 현재 self-rally 구조가 이어받을 수 있지만, 첫 타격 자체가 실패하면 episode가 바로 무너진다.

필요한 단계:

1. 타격 가능 영역을 먼저 계산한다.
   - 로봇팔 workspace
   - 라켓이 도달 가능한 xy/z
   - 공이 하강 중일 때 접촉 가능한 시간
   - 로봇팔 아래/몸통 충돌 영역 제외
   - 최소 접촉 높이와 최대 접촉 높이
2. 초기 위치 curriculum을 만든다.
   - 현재 범위 `0.028m`에서 시작
   - 성공률이 일정 기준을 넘으면 xy radius, z range, incoming velocity를 넓힘
   - 한 번에 0~360도 전체 random으로 열면 학습 난이도가 너무 급격히 오른다
3. "첫 공 복구 skill"과 "안정 rally skill"을 분리한다.
   - 첫 타격: 공을 안정 영역으로 되돌리는 recovery/centering skill
   - 이후 반복: 일정 높이와 방향을 유지하는 rally skill
4. 목표 조건을 observation에 넣는다.
   - desired apex xy/z
   - desired next-intercept
   - reset difficulty 혹은 target region

이건 단순히 "라켓 중심을 공에 맞춰 따라가는 로직"만으로는 부족하다. 공을 따라가는 것은 접촉을 만들기 위한 controller primitive이고, 강화학습은 "어디를 어떻게 맞아야 다음 공이 안정적인가"를 배워야 한다. 임의 위치 일반화를 하려면 둘 다 필요하다.

## 앞으로 개선할 수 있는 방법 리스트

### 1. 발표 전 우선순위: champion checkpoint 선정

남은 시간이 짧기 때문에 가장 먼저 해야 할 것은 학습을 무한히 늘리는 게 아니라, v19/v20/checkpoint 중 발표에 가장 안정적인 모델을 고르는 것이다.

권장:

- v19 final, v20 final, v20 best를 각각 100 episode 분석
- viewer로 time_limit episode와 high-useful episode를 직접 확인
- 발표용으로 가장 안정적인 모델을 freeze
- 발표에서는 평균 지표와 best-case viewer 영상을 둘 다 제시

### 2. 평가 기준 개선

지금은 평균 useful bounce만 보면 모델 품질이 완전히 드러나지 않는다. 발표/개발 모두에서 다음 기준이 더 명확하다.

- time_limit episode 비율
- `3+`, `5+`, `8+` useful bounce 비율
- terminal low-apex 비율
- terminal `ball_out_of_bounds` 방향별 분해
- projected apex height 평균/분산
- projected apex xy error 평균/분산
- contact별 outgoing z/xy error

최종 목표가 "적절한 높이와 방향으로 계속 치기"라면, 평균 bounce보다 "apex height 분산"과 "apex xy drift"가 더 직접적인 품질 지표다.

### 3. Curriculum learning

가장 중요하다. 임의 시작 위치 일반화까지 가려면 curriculum이 사실상 필수다.

자동 curriculum은 agent 능력에 맞춰 task difficulty를 조절해 sample efficiency, generalization, sparse reward 문제를 개선하는 방향으로 많이 쓰인다. Automatic Curriculum Learning survey에서도 task difficulty를 agent capacity에 맞추는 것을 핵심으로 설명한다.

우리 문제에 적용:

- level 0: 현재 v19 시작 범위
- level 1: xy radius 0.04, z range 0.03
- level 2: xy radius 0.06, z range 0.05
- level 3: incoming xy velocity 추가
- level 4: reachable workspace 전체 중 쉬운 subset
- level 5: reachable workspace 전체

성공률 기준:

- 최근 평가에서 `2+` useful rate > 0.65
- `ball_out_of_bounds` < 0.05
- terminal low-apex < 0.4

이런 식으로 다음 난이도를 여는 방식이 좋다.

### 4. Goal-conditioned policy

현재 정책은 사실상 하나의 목표를 향한다: anchor 주변으로 적절한 apex를 만들기. 임의 시작 위치와 다양한 목표 높이를 다루려면 policy input에 goal을 넣는 방식이 유리하다.

예:

- goal apex xy
- goal apex height
- next contact target
- desired outgoing velocity
- difficulty/context id

Universal Value Function Approximators는 state뿐 아니라 goal에도 일반화하는 value function 아이디어를 제시한다. 이 방향은 "시작 위치가 달라도 같은 안정 목표로 보내기"와 잘 맞는다.

### 5. HER / off-policy replay

현재 PPO는 on-policy라서 실패 trajectory 재사용이 제한적이다. 임의 위치 일반화와 sparse success로 가면 실패가 많아질 가능성이 높다. 이때 SAC/TD3 + replay buffer + goal relabeling을 검토할 수 있다.

Hindsight Experience Replay는 실패한 trajectory라도 "실제로 달성한 goal"로 다시 라벨링해 학습에 활용하는 방식이다. 로봇팔 manipulation sparse reward 문제에서 sample efficiency 개선을 목표로 제안됐다.

우리 문제 적용 가능성:

- 원래 goal: anchor 위 0.30m apex
- 실제 달성 goal: 특정 apex xy/z
- 실패 contact도 "그 방향으로 치는 방법" 데이터로 재사용

단점:

- 현재 PPO 코드 구조를 꽤 바꿔야 한다.
- off-policy algorithm으로 전환하거나 별도 buffer를 만들어야 한다.
- 발표 전에는 리스크가 크고, 장기 개선 후보에 가깝다.

### 6. Hierarchical / modular policy

DeepMind의 competitive robot table tennis 작업은 low-level skill controller와 high-level skill selector를 나누는 계층적/모듈형 구조를 사용했다. 우리 문제도 비슷한 분해가 자연스럽다.

후보 skill:

- `recover_first_ball`: 임의 위치 공을 anchor 근처 안정 영역으로 복구
- `stable_rally`: 이미 안정 영역에 있는 공을 일정 높이로 반복
- `low_apex_rescue`: 낮게 통통거리는 공을 강하게 위로 회복
- `boundary_rescue`: x/y 바깥으로 가는 공을 안쪽으로 복구
- `soft_centering`: 안정 상태에서 lateral drift를 작게 줄임

현재 v20의 lateral brake는 `boundary_rescue`를 hand-coded primitive로 살짝 넣은 것에 가깝다. 장기적으로는 high-level phase/skill selector가 더 깔끔할 수 있다.

### 7. Model-based feedforward + RL residual

우리 구조는 이미 hand-coded feedforward 위에 RL residual을 얹는 방향이다. 이 방향은 타당하다. 로봇 탁구 논문에서도 model-based/feedforward control은 sample efficiency와 안정성 측면에서 장점이 있다고 설명한다.

더 발전시키려면:

- incoming ball velocity, racket normal, racket velocity로 outgoing velocity를 예측하는 contact model 학습
- desired outgoing velocity를 만들기 위한 inverse contact solver
- RL은 solver 오차, 타이밍 오차, lateral drift만 residual로 보정

이렇게 하면 "왜 낮게 튕기는지"를 reward만으로 때우는 것보다 더 물리적으로 직접적이다.

### 8. Asymmetric actor-critic / privileged critic

시뮬레이션에서는 실제 로봇보다 더 많은 정보를 알 수 있다. actor는 실제 배포 가능한 observation만 보고, critic은 학습 중에 contact trace, projected apex, next-intercept, hidden simulator state를 더 볼 수 있게 하는 방식이 가능하다.

Asymmetric Actor-Critic 계열은 simulator의 full-state 정보를 critic에 활용해 더 좋은 policy 학습을 돕는 방향이다. 현재 우리 환경도 contact trace와 apex 예측 정보가 풍부하므로 잘 맞는다.

단점:

- Stable-Baselines3 기본 PPO 구조를 커스터마이즈해야 할 수 있다.
- 구현 난이도는 중간 이상이다.

### 9. Domain randomization / sim-to-real robustness

웹 발표나 시뮬레이션 졸업작품이면 필수는 아니지만, "실제와 유사한 탁구공 재질/반발"을 주장하려면 중요하다.

랜덤화 후보:

- ball mass
- ball radius
- ball/racket restitution
- ball/racket friction
- contact solref/solimp
- actuator delay
- controller gain
- gravity 소폭
- observation noise

OpenAI의 ADR 사례는 simulation randomization range를 성능에 따라 자동으로 넓히는 방식이고, Bayesian Domain Randomization은 reality gap을 줄이기 위해 domain parameter distribution을 조정하는 방향이다.

우리에게 당장 유용한 건 manual randomization + curriculum이다. 발표 전에는 넣지 않는 편이 안전하지만, 향후 연구 방향으로는 좋다.

### 10. Action dimension 확장 후보

v19/v20에서 action이 포화되지 않는 축이 많기 때문에 무조건 차원을 늘린다고 해결되지는 않는다. 하지만 다음 축들은 타당한 후보이다.

우선순위 높은 후보:

- desired apex height residual
  - 낮은 통통 루프를 직접 겨냥
  - 현재는 z/vz residual이 간접적으로만 apex를 조절
- strike plane / contact timing residual
  - 너무 낮게 맞거나 늦게 맞는 문제를 직접 조절
- follow-through time residual
  - 접촉 후 라켓 움직임이 공에 주는 z/xy 영향을 조절
- boundary rescue intensity
  - v20 brake를 RL이 상황별로 조절하게 만드는 축
- contact target inward offset scale
  - planner contact offset을 고정값이 아니라 정책이 선택

우선순위 낮은 후보:

- 단순 tilt limit 확대
- outgoing xy residual limit 확대
- 전체 action bound 확대

이유: v19 분석에서 action saturation은 거의 없었고, 남은 문제는 action bound 부족보다 controller/contact execution 쪽이었다.

2026-06-03 업데이트: v20 분석 후 첫 두 후보인 `desired apex height residual`과 `strike plane / contact timing residual`을 v21 실험으로 구현했다. 상세 내용은 `43_v20_review_and_v21_apex_timing_residual.md`에 정리했다.

### 11. Recurrent policy

현재 observation에 많은 상태가 들어가지만, 실제 접촉은 아주 짧고 timing/latency가 중요하다. LSTM policy나 frame stacking은 "직전 몇 step의 속도/접촉 흐름"을 더 잘 보게 할 수 있다.

다만 현재는 full simulator state에 가까운 observation을 쓰고 있으므로, recurrent policy는 우선순위가 action/ curriculum/ model-based contact보다 낮다.

### 12. Feasible reset map

임의 시작 위치를 하려면 먼저 "칠 수 있는 위치"를 알아야 한다.

구현 아이디어:

- xy grid, z grid, incoming velocity grid를 샘플링
- heuristic controller 혹은 oracle로 1회 접촉 가능성 평가
- 조건:
  - predicted descending intercept exists
  - intercept time within controller reach time
  - racket target within workspace
  - body collision 없음
  - contact height above racket/anchor feasible
  - first hit projected apex reaches target band
- 이 map을 curriculum reset distribution으로 사용

이걸 만들면 "완전 뒤죽박죽"이 아니라 "타격 가능한 뒤죽박죽"을 정의할 수 있다.

## 발표 전 추천 의사결정

남은 시간이 7일 미만이면 이렇게 하는 것이 현실적이다.

1. v20 학습이 끝나면 v19와 v20을 같은 100 episode 분석으로 비교한다.
2. 더 안정적인 모델을 발표 champion으로 정한다.
3. viewer에서 좋은 time_limit/high-useful episode를 녹화한다.
4. 웹 서비스에는 다음을 보여준다.
   - 모델 선택
   - 학습 결과 요약
   - episode별 실패 이유
   - contact/apex 그래프
   - viewer 영상 혹은 GIF
5. 발표에서 "현재 한계"를 먼저 인정하고, 이후 roadmap을 제시한다.

v19만으로도 발표는 가능하다. v20은 더 좋아질 수도 있지만, 발표 성공 여부를 v20 하나에 걸면 위험하다. v19를 fallback champion으로 보존하고, v20은 improvement candidate로 보는 게 좋다.

## 참고한 연구 방향

- DeepMind robot table tennis는 low-level skill controller와 high-level controller를 나눈 hierarchical/modular policy, task distribution curriculum, sim-to-real 기법을 핵심 기여로 설명한다: https://deepmind.google/research/publications/107741/
- model-based feedforward robotic table tennis 연구는 feedforward/feedback 구조와 prior dynamics가 sample complexity와 안정성에 유리하다고 설명한다: https://link.springer.com/article/10.1007/s10514-023-10140-6
- Residual Policy Learning은 좋은데 불완전한 controller 위에 RL residual을 얹는 방식이 복잡한 로봇 manipulation에서 유리하다는 관점이다: https://arxiv.org/abs/1812.06298
- Automatic Curriculum Learning survey는 task difficulty를 agent 능력에 맞춰 조절해 sample efficiency, exploration, generalization, sparse reward 문제를 개선하는 흐름을 정리한다: https://arxiv.org/abs/2003.04664
- Universal Value Function Approximators는 goal-conditioned 일반화의 기본 아이디어로, 다양한 목표 apex/target으로 확장할 때 참고할 수 있다: https://proceedings.mlr.press/v37/schaul15.html
- Hindsight Experience Replay는 실패 경험을 achieved goal로 relabeling해 sparse reward 로봇 학습의 sample efficiency를 높이는 방법이다: https://arxiv.org/abs/1707.01495
- Asymmetric Actor-Critic은 simulator의 privileged/full-state 정보를 critic에 활용해 robot policy 학습을 돕는 방향이다: https://arxiv.org/abs/1710.06542
- Automatic Domain Randomization은 simulation randomization 범위를 점진적으로 넓혀 sim-to-real robustness를 얻는 방향이다: https://openai.com/index/solving-rubiks-cube/
- Bayesian Domain Randomization은 sim-to-real reality gap을 줄이기 위해 domain parameter distribution을 적응적으로 조정하는 방향이다: https://www.dfki.de/en/web/research/projects-and-publications/publication/13662
