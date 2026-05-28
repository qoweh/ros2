# pingpong_rl2 학습 설계 체크리스트

## 1. 매 실험 전 체크

실험 하나는 가설 하나만 검증해야 한다.

실험 시작 전에 아래를 적는다.

- run name
- 바꾼 것
- 안 바꾼 것
- 좋아졌다고 판단할 지표
- 실패했다고 판단할 지표

예시:

```text
run_name: timed_neg_pitch_v1_100k
changed: strike window에서만 negative pitch ramp
unchanged: reward, reset distribution, PPO hyperparameter
success: mean_useful_bounces 증가, useful contact 후 mean vx 감소
failure: useful contact 감소, robot_body_contact 증가
```

## 2. 봐야 할 지표

### 기본 지표

- episode return
- episode length
- useful bounce count
- contact count
- failure reason count

### contact 지표

- first contact가 racket인지 body인지
- contact 시 `ball_velocity`
- contact 시 `racket_velocity`
- contact 시 `racket_face_normal`
- contact 시 `target_tilt`
- useful contact 후 outgoing `vx`, `vy`, `vz`

### 현재 가장 중요한 지표

- useful contact 후 `ball_velocity_x`
- `+x` 방향 contact 비율
- 두 번째 useful contact 발생 비율
- `ball_out_of_bounds` 비율

## 3. 실패 모드별 처방

### first contact 자체가 안 생김

가능 원인:

- reset 분포가 너무 넓다.
- strike guard가 너무 빡빡하다.
- action/controller가 공 아래로 갈 시간을 못 준다.

처방:

- reset xy range를 임시로 줄인다.
- 초기 ball height/velocity를 더 쉬운 분포로 둔다.
- reward 추가보다 먼저 contact 비율을 회복한다.

### 첫 contact는 있는데 useful bounce가 적음

가능 원인:

- upward racket velocity가 부족하다.
- contact timing이 너무 늦거나 빠르다.
- success threshold가 실제 physics와 안 맞는다.

처방:

- contact 시 `racket_velocity_z`, `ball_velocity_z`를 기록한다.
- strike window 높이와 upward ready timing을 조정한다.
- success threshold를 낮추기 전에 실제 contact velocity 분포를 본다.

### useful bounce는 있는데 바로 바깥으로 나감

가능 원인:

- racket normal이 outward rebound를 만든다.
- incoming velocity와 racket velocity의 relative direction이 좋지 않다.
- position-domain tilt signal이 rebound physics와 약하게 연결되어 있다.

처방:

- timing-gated negative pitch를 먼저 확인한다.
- 그래도 부족하면 velocity-domain observation/reward로 간다.
- contact event에만 outgoing direction term을 작게 붙인다.

### body contact가 많음

가능 원인:

- pre-contact XY 이동이 너무 공격적이다.
- racket보다 팔 링크가 먼저 공 경로에 들어간다.

처방:

- body keepout과 XY clamp를 유지한다.
- reset range를 줄여 first-contact 학습을 먼저 안정화한다.
- body contact penalty를 키우기 전에 어떤 body가 맞는지 기록한다.

### policy가 tilt를 안 씀

가능 원인:

- tilt action이 contact 결과에 도달하기 전에 neutral로 사라진다.
- PPO 입장에서는 tilt를 쓰지 않는 local optimum이 더 쉽다.
- reward가 tilt 사용의 이득을 직접 보여주지 않는다.

처방:

- base inward tilt + residual 형태를 고려한다.
- warm-start로 position policy를 먼저 학습한 뒤 tilt branch를 연다.
- tilt reward를 직접 주기보다는 contact 후 outgoing direction 개선을 보상한다.

## 4. observation 설계

현재 또는 다음 단계에서 중요한 observation 후보는 아래다.

유지:

- joint positions
- joint velocities
- racket position
- target position
- ball position
- ball velocity
- ball relative position

추가 후보:

- racket velocity
- racket face normal
- relative velocity: `ball_velocity - racket_velocity`
- predicted time-to-contact

우선순위:

1. racket velocity
2. racket face normal
3. relative velocity
4. time-to-contact

## 5. reward 설계

reward는 최소로 유지한다.

기본 구조:

- descending strike window alignment
- useful upward contact bonus
- projected apex quality
- failure penalty

추가 후보:

- useful contact event에서만 outgoing direction error
- useful contact event에서만 lateral velocity penalty

피할 것:

- 매 step lateral velocity penalty
- tilt 사용량 자체 reward
- 여러 reward를 한 번에 추가
- single-bounce-out penalty를 너무 일찍 넣기

## 6. 학습 운영

권장 run 길이:

- smoke: 1k
- 구조 확인: 50k
- 경향 확인: 100k
- 비교 가능: 300k
- 후보 검증: 500k
- 최종: 1M

권장 seed:

- 구조 확인은 1 seed
- 후보 검증은 2 seeds
- 최종 결과는 3 seeds

## 7. 결론 판단 규칙

다음 중 하나면 다음 단계로 넘어간다.

- useful bounce가 늘고 failure mode가 악화되지 않는다.
- useful contact 후 `+x` rebound가 줄어든다.
- viewer에서 3회 이상 keep-up이 반복 관찰된다.

다음 중 하나면 실험을 버린다.

- contact count가 크게 줄었다.
- body contact가 늘었다.
- tilt/action이 실제 contact 시점에 사용되지 않는다.
- 100k에서도 지표 방향이 baseline보다 나쁘다.

## 8. 최종 보고서 구조

프로젝트를 마무리할 때 최종 md는 아래 순서로 쓰면 된다.

1. 문제 정의
2. 환경과 로봇 구성
3. baseline 설계
4. 실패 원인 분석
5. 실험별 가설과 결과
6. 최종 선택한 학습 설계
7. 정량 평가
8. viewer 결과
9. 한계와 다음 작업
