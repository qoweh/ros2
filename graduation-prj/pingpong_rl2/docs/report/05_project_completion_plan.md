# pingpong_rl2 프로젝트 완성 계획

## 1. 현재 결론

지금 문제는 “PPO를 더 오래 돌리면 해결될지 모르는 상태”가 아니다.

현재까지 좁혀진 결론은 아래다.

- position-only baseline은 공을 맞추는 쪽으로는 어느 정도 간다.
- 하지만 반복 keep-up을 만들기 전에 공을 `+x` 바깥쪽으로 보내는 rebound 편향이 남아 있다.
- `neutral -> pre-contact ramp -> contact -> neutral` 구조는 맞다.
- fixed negative pitch는 outward 편향을 줄이는 방향으로 효과가 있었다.
- `intercept-center offset` 기반 dynamic tilt는 실패했다. 위치 기반 신호는 rebound 방향을 직접 설명하지 못했다.
- 다음 핵심은 reward를 많이 붙이는 것이 아니라, contact 순간의 velocity/normal 기반 변수로 rebound를 제어하는 것이다.

따라서 프로젝트 완성 방향은 다음 한 문장이다.

> 먼저 position-only로 “반복해서 공 밑으로 들어가는 능력”을 안정화하고, 그 다음 timing-gated inward tilt 또는 velocity-domain reward/observation으로 rebound 방향을 잡는다.

## 2. 완성 기준

최종 목표는 simulation viewer에서 “로봇팔이 탁구채로 공을 계속 위로 올려치는 것”이 눈으로 확인되는 것이다. 숫자 기준은 아래처럼 둔다.

### 최소 완성 기준

- 50 evaluation episodes 기준 `mean_useful_bounces >= 2.0`
- `max_useful_bounces >= 5`
- `robot_body_contact` 실패 비율 `< 10%`
- viewer에서 한 번 이상 5회 이상 연속 keep-up이 관찰됨

### 발표/보고서 기준

- 3개 seed에서 같은 경향 확인
- 각 seed 50 episodes 평가 저장
- 대표 viewer 영상 또는 캡처 확보
- 실패 모드가 문서에 정리되어 있음

### 더 욕심낼 기준

- 50 evaluation episodes 기준 `mean_useful_bounces >= 5.0`
- `ball_out_of_bounds` 실패 비율 `< 40%`
- useful contact 후 평균 `ball_velocity_x`가 0 근처 또는 로봇 쪽으로 돌아오는 방향

## 3. 작업 단계

### Phase 0. 실험 체계 고정

목표는 “뭘 바꿨더니 좋아졌는지”가 보이게 만드는 것이다.

해야 할 것:

- run name에 실험 의도를 넣는다.
- 모든 run summary에 env config, seed, total timesteps, eval 결과를 저장한다.
- 1M을 바로 돌리지 말고 50k, 100k, 200k, 500k 순서로 간다.
- viewer 판단은 마지막에 하고, 중간 판단은 수치로 한다. viewer판단이 필요할 시 멈춰서 사용자(client)에게 확인 요청한다.

필요한 코드 변경 후보:

- `scripts/run_ppo_learning.py`에 이미 summary 저장이 있다면 유지한다.
- rebound 분석 script가 없거나 부족하면 `scripts/analyze_rebound.py` 같은 파일을 두고 contact별 `ball_velocity`, `racket_velocity`, `racket_normal`, `target_tilt`를 csv/json으로 저장한다.

### Phase 1. position-only 반복 준비 능력 확인

목표는 tilt 없이도 첫 contact 이후 다음 공 밑으로 다시 들어가는 학습 신호가 살아있는지 확인하는 것이다.

실험:

- `position_repeat_v1_100k`
- `position_repeat_v1_300k`

볼 지표:

- `mean_useful_bounces`
- `max_useful_bounces`
- `failure_counts`
- 첫 useful contact 이후 두 번째 contact가 생기는 비율

성공이면:

- Phase 2로 넘어간다.

실패이면:

- reset 범위를 줄인다.
- strike window timing을 조금 넓힌다.
- reward를 늘리기 전에 “공 아래로 들어가는 dense signal”이 접촉 이후에도 살아있는지 확인한다.

### Phase 2. timing-gated negative pitch

목표는 이미 효과가 있었던 negative pitch만 strike 타이밍에 잠깐 켜는 것이다.

핵심 가설:

- 항상 기울이는 것이 아니라, 공이 내려오고 contact가 가까운 순간에만 inward face normal bias를 준다.
- contact 이후에는 neutral로 돌려 다음 준비를 방해하지 않는다.

실험:

- `timed_neg_pitch_v1_50k`
- `timed_neg_pitch_v1_100k`
- 좋아 보이면 `timed_neg_pitch_v1_300k`

대충 필요한 코드 변경:

- `keepup_env.py`에서 contact 준비 상태일 때만 target pitch를 `negative_pitch_magnitude` 쪽으로 ramp한다.
- 조건은 대략 `ball_velocity_z < 0`, `ball_height_above_racket`이 strike window 안, `xy_alignment_error`가 너무 크지 않을 때로 둔다.
- contact 이후 또는 ball 상승 구간에서는 target tilt를 `0`으로 되돌린다.
- 이 assist는 env 내부 학습 assist가 아니라 실험 옵션으로 opt-in이어야 한다.

판단 기준:

- useful contact 수가 position-only보다 줄면 실패.
- useful contact 후 `ball_velocity_x` 평균이 줄면 성공 후보.
- `mean_useful_bounces`가 늘면 Phase 3로 넘어간다.

### Phase 3. velocity-domain signal로 전환

timing-gated negative pitch가 부족하면 위치 기반 신호를 버리고 속도 기반으로 간다.

핵심 가설:

- rebound 방향은 `intercept-center offset`보다 contact 순간의 `relative velocity`와 `racket normal`에 더 강하게 묶여 있다.

대충 필요한 코드 변경:

- observation에 아래 후보를 추가한다.
  - `racket_velocity`
  - `racket_face_normal`
  - `ball_velocity - racket_velocity`
  - 가능하면 `predicted_time_to_contact`
- contact info에 아래를 저장한다.
  - contact 직전/직후 `ball_velocity`
  - contact 시 `racket_velocity`
  - contact 시 `racket_face_normal`
- reward 후보는 아주 작게 하나만 추가한다.
  - useful contact가 발생했을 때만 desired outgoing velocity와 실제 outgoing velocity 차이를 penalty로 준다.
  - 예: `outgoing_direction_term = -w * abs(contact_ball_velocity_x - desired_vx)`

주의:

- 매 step마다 lateral velocity penalty를 주면 이상한 회피 정책이 나올 수 있다.
- contact event에만 붙이는 것이 더 안전하다.

### Phase 4. 최종 학습

Phase 1~3 중 가장 좋은 구조가 정해지면 그때 긴 학습을 한다.

권장 순서:

1. 100k 단일 seed
2. 300k 단일 seed
3. 500k 2개 seed
4. 1M 3개 seed

최종 선택 기준:

- 평균 bounce 수만 보지 않는다.
- `ball_out_of_bounds`, `robot_body_contact`, `ball_speed_limit`을 같이 본다.
- viewer에서 움직임이 말이 되는지 확인한다.

## 4. 하지 말 것

- 1M을 여러 번 무작정 돌리지 않는다.
- reward term을 한 번에 여러 개 추가하지 않는다.
- tilt, spin, curriculum, reward shaping을 동시에 켜지 않는다.
- viewer에서 “그럴듯해 보임”만으로 성공 판단하지 않는다.
- `intercept-center offset` 기반 tilt를 계속 미세 조정하지 않는다. 이미 신호 자체가 약하다는 쪽으로 결론이 났다.

## 5. 최종 산출물

프로젝트를 끝낼 때 남겨야 할 것은 아래다.

- 최종 모델 zip
- 학습 summary json
- 50 episode evaluation json
- rebound/contact analysis csv 또는 json
- 대표 viewer 영상 또는 캡처
- 최종 보고서: 성공한 설계, 실패한 설계, 남은 한계
