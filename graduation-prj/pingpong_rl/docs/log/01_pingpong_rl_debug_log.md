# pingpong_rl 디버그 로그

## 1. 이번 정리의 목적

이 문서는 강화학습과 MuJoCo 세팅을 바꾸면서 실제로 무엇이 문제였고, 무엇이 문제처럼 보였지만 원인이 아니었는지를 기록하기 위한 로그다.

보고서 작성 시 필요한 핵심은 아래 세 가지다.

- 어떤 가설을 세웠는가
- 무엇이 실제 원인이었는가
- 어떤 수정이 이후 실험의 기준점이 되었는가

## 2. 현재 공 기본 높이

- `DEFAULT_BALL_HEIGHT = 0.5`
- `DEFAULT_VIEWER_RESET_HEIGHT = 0.5`

이제 학습과 렌더링 모두 기본적으로 더 높은 drop height를 사용한다.

이 변경 이유는 단순하다.

- 이전 기본 높이에서는 공이 너무 빨리 라켓 근처에 들어와 초반 학습이 조급하게 진행됐다.
- 사용자가 직접 렌더링으로 확인했을 때도 `0.5` 정도가 strike 준비 동작을 보기 더 쉬웠다.

## 3. 원인으로 의심했지만 주원인은 아니었던 것

### 3.1 탁구공 질량

현재 공 질량은 `0.0027 kg`이다.

이 값은 실제 탁구공 질량과 크게 어긋나지 않는다. 따라서 이번 이상 동작의 주원인을 `공이 너무 무겁다`로 보기는 어렵다.

### 3.2 로봇 actuator 힘 자체

Franka actuator gain/force 범위를 다시 확인했을 때, 근본 문제가 `힘이 약해서 절대 못 친다` 수준은 아니었다.

문제는 actuator 최대힘보다, 현재 EE controller 설정에서 라켓이 실제로 만들어내는 end-effector 속도가 너무 낮았다는 점이었다.

## 4. 실제로 확인된 문제

### 4.1 접촉 순간 떨기(jitter contact)

이전 active-hit score는 접촉 순간 `racket_acceleration_z`가 크면, 실제 `racket_velocity_z`가 위쪽이 아니어도 일부 점수를 받을 수 있었다.

그 결과 정책이 아래와 같은 이상 행동을 학습할 여지가 있었다.

- 위로 확실히 치기보다 contact 근처에서 잔진동 만들기
- 공을 올리는 stroke보다 contact 시점의 순간적인 acceleration spike 만들기

현재는 active-hit score가 `실제 upward racket velocity`를 먼저 만족해야만 양수가 되도록 수정했다.

즉, `가속도만 큰 떨기`는 이제 useful hit로 인정되지 않는다.

### 4.2 로봇팔 EE 응답 속도 부족

기존 controller 설정에서는 최대 upward action을 반복해도 라켓 z속도가 대략 `0.093 m/s` 수준이었다.

이건 `탁 치는` 행동이라기보다 `조금씩 따라가는` 움직임에 가깝다.

설정 수정 후 같은 probe에서 라켓 z속도는 대략 `0.467 m/s`까지 올라간다.

즉, 이전에는 policy가 소심해서 못 친다기보다, controller 응답이 실제 strike motion을 거의 만들어주지 못했다.

### 4.3 공이 로봇팔 본체를 뚫고 지나가던 문제

이건 reward 문제가 아니라 contact 설정 실수였다.

이전에는 `panda.xml`의 collision mesh 기본값에 `contype="0" conaffinity="0"`가 들어가 있어서, 라켓 head를 제외한 arm body collision이 사실상 꺼져 있었다.

그 결과:

- 공이 로봇 본체를 그냥 통과할 수 있었고
- 관측상 물리법칙을 어기는 장면이 보였다.

현재는 robot collision mesh를 다시 활성화했고, 공이 link/body에 닿으면 `robot_body_contact` failure로 기록되게 바꿨다.

예시 probe:

- inner-side drop에서 `failure_reason=robot_body_contact`
- 실제 contact body는 `link5`

즉, 이제는 본체를 통과하는 대신 `무엇에 부딪혔는지`가 명시적으로 드러난다.

## 5. 이번에 추가된 학습 안정화 요소

### 5.1 staged curriculum

`run_ppo_baseline.py`는 기본적으로 `keepup_v1` curriculum을 사용한다.

stage는 아래와 같다.

- `bootstrap`
  - reset randomization 제거
  - success threshold 완화
  - orientation/joint/smoothness 항목 비활성화
- `stabilize`
  - 약한 randomization 재도입
  - 라켓 수평 유지, 관절 속도, action 변화량 페널티를 약하게 활성화
- `refine`
  - full threshold 복귀
  - smoothness/tilt/joint penalty 강화
  - action filter 활성화

즉 초반에는 `일단 맞추고 올리는 동작`을 배우게 하고, 후반에는 `그 동작을 더 수평하고 덜 끊기게 유지`하도록 설계했다.

### 5.2 자세와 관절 관련 보상

현재 env에는 아래 penalty term이 추가됐다.

- `orientation_term`
  - 라켓 면 normal의 z성분이 작아질수록 감점
  - 의미: 라켓 면이 지면과 평행하게 유지되도록 유도
- `joint_motion_term`
  - joint velocity norm이 클수록 감점
  - 의미: 관절을 과도하게 흔드는 행동 억제
- `action_smoothness_term`
  - 현재 action과 직전 action 차이가 클수록 감점
  - 의미: 뚝뚝 끊기는 target jump 억제

추가로 `action_filter_alpha`가 later stage에서 활성화되므로, 실제 실행 action도 더 부드럽게 바뀐다.

## 6. 현재 결론

이번 구간의 핵심 결론은 아래다.

- `공 질량`은 주원인이 아니었다.
- `라켓/팔 body collision을 꺼둔 설정`은 명백한 오류였다.
- `접촉 순간 acceleration만 보는 보상`은 jitter를 학습하기 쉬운 구조였다.
- `controller 응답 속도`가 너무 느려서 명확한 upward strike가 잘 나오지 않았다.
- 따라서 다음 실험은 더 오래 학습하는 것보다, 수정된 물리/보상/커리큘럼 기준으로 새 모델을 다시 학습하는 것이 맞다.
