# 탁구공 upward bounce RL 전체 가이드

## 1. 이 문서가 답하려는 질문

이 문서는 아래 질문에 답하기 위해 쓴다.

- 지금 무엇을 만들려고 하는가
- MuJoCo와 RL을 어떻게 역할 분담해야 하는가
- MacBook M1에서 어떤 실행 루프로 개발해야 하는가
- `mjpython`, `python -m mujoco.viewer`, headless training은 각각 언제 쓰는가
- 왜 viewer 키 입력이 이상하게 느껴지는가
- 왜 적당히 움직이는 제어기를 먼저 만들고 그 위에 RL을 얹으라고 하는가
- 현재 저장소 구조에서 어떤 파일이 필요하고, 어떤 순서로 진행해야 하는가

이 문서는 `경기형 탁구`가 아니라

- `로봇팔 + 탁구채 + 탁구공`
- `공을 계속 위로 튕기기`

만을 목표로 한다.

## 2. 먼저 목표를 정확히 고정하자

네 목표는 현재 기준으로 아래처럼 정의하는 것이 가장 좋다.

### 2.1 최종 목표

- simulation 안에서
- 공이 라켓에 반복적으로 맞고
- 바닥에 닿기 전 다시 위로 올라가게 하며
- 가능한 오래 연속 바운스를 유지하는 정책을 학습하기

### 2.2 처음부터 목표를 너무 크게 잡지 말 것

바로 아래 목표로 가지 않기:

- 탁구 경기
- 네트 넘기기
- 반대편 코트에 보내기
- spin 대응
- 상대 적응

이것들은 모두 다음 단계다.

지금 네가 필요한 것은 `single bounce`와 `multi-bounce`를 분리해서 생각하는 것이다.

## 3. 왜 “규칙만 주고 무에서 유로 학습”이 잘 안 되는가

이건 네가 헷갈리는 핵심 지점이다.

짧게 말하면,

`MuJoCo contact + 7축 로봇팔 + 희소 reward를 한 번에 주면, 정책이 배워야 할 탐색 공간이 너무 크다.`

정책이 동시에 배워야 하는 것:
- 로봇 기구학
- 공의 낙하 예측
- contact timing
- 라켓 위치 조절
- 바운스 후 공의 방향 제어

이걸 아무 유도 없이 한 번에 맡기면 보통 아래 둘 중 하나가 된다.

- 공을 거의 못 맞춤
- 우연히 맞춰도 재현 불가

그래서 `적당히 움직이는 제어`를 먼저 만든다는 말은,

- 완성된 인간 수준의 행동을 하드코딩하라는 뜻이 아니라
- 정책의 탐색 공간을 학습 가능한 수준으로 줄이라는 뜻

이다.

예시:

- 나쁜 action space: joint torque 7개를 그대로 학습
- 현재보다 나은 action space: `racket_center` delta xyz
- 더 나은 다음 후보: `racket_center` delta xyz + 제한된 orientation

즉 RL이 해야 하는 것은 `공을 보고 타격 위치를 정하는 것`이지, `Franka 전체 관절 운동학을 처음부터 다시 발견하는 것`이 아니다.

## 4. 시스템을 레이어로 나누면 감이 잡힌다

현재 프로젝트를 아래 네 레이어로 보면 훨씬 쉽다.

### 4.1 레이어 A: 물리와 씬

담당:
- 로봇 MJCF
- 라켓 형상과 질량
- 공 형상, 질량, bounce
- 바닥

현재 저장소에서 대응:
- `pingpong_rl/assets/scene.xml`
- `pingpong_rl/assets/franka/panda.xml`
- `PingPongSim`

이 단계는 RL이 아니다.

### 4.2 레이어 B: 저수준 제어기

담당:
- 원하는 `racket_center` 움직임을 joint target으로 바꾸기
- joint limit 안에서 안정적으로 이동시키기

현재 저장소에서 대응:
- `RacketCartesianController`

이 단계도 RL이 아니다.

### 4.3 레이어 C: task env

담당:
- observation 정의
- action 정의
- reward 정의
- termination 정의

현재 저장소에서 대응:
- `PingPongEEDeltaEnv`
- `PingPongEEDeltaGymEnv`

여기서부터 RL이 붙는다.

### 4.4 레이어 D: 학습 알고리즘

담당:
- PPO, SAC, TD3 등으로 정책 학습
- 로그 저장
- 체크포인트 저장
- 평가 루프 분리

현재 저장소에서 대응:
- `run_ppo_baseline.py`
- `run_ppo_render.py`
- `PPOLoggingCallback`

즉 지금 저장소는 이미 A~D 레이어를 아주 초기 형태로는 갖고 있다. 네가 막막한 이유는 이 레이어들이 아직 `upward multi-bounce` 목표에 맞게 충분히 정리되지 않았기 때문이다.

## 5. MacBook M1에서 실제로 어떻게 개발해야 하나

현재 환경에서 확인된 사실:

- Python: `3.10.20`
- MuJoCo: `3.8.0`
- PyTorch: `2.11.0`
- `torch.backends.mps.is_available() == True`
- `mjpython` 사용 가능

즉 네 머신에서는 MuJoCo와 MPS 둘 다 쓸 수 있다.

### 5.1 꼭 기억할 점

MuJoCo physics 자체는 CPU에서 돈다.

MPS가 담당하는 것은 주로
- policy network forward
- gradient update

쪽이다.

그래서 작은 MLP + 단일 env 기준에서는 `device=mps`가 항상 큰 속도 향상을 주지 않는다. 오히려 아래처럼 비교해서 결정하는 것이 맞다.

옵션 A: CPU
- 장점: 단순하고 디버깅이 쉽다.
- 장점: 작은 모델에서는 오버헤드가 적다.
- 단점: 큰 배치나 큰 모델에서 느릴 수 있다.

옵션 B: MPS
- 장점: policy update 쪽은 빨라질 수 있다.
- 장점: CUDA가 없는 Mac에서 쓸 수 있는 유일한 GPU 옵션이다.
- 단점: env stepping은 여전히 CPU다.
- 단점: 작은 실험에서는 체감 차이가 작을 수 있다.

현재 목적이 `처음 감 잡기`라면 CPU부터 해도 충분하다. `속도 비교`가 궁금하면 같은 스크립트를 `--device cpu`와 `--device mps`로 각각 10k~50k step만 돌려 wall-clock을 비교하는 편이 낫다.

## 6. viewer와 training은 같은 것이 아니다

네가 지금 가장 헷갈리는 부분 중 하나가 이거다.

### 6.1 scene inspection

목적:
- scene이 로드되는지
- 카메라가 맞는지
- 라켓/공 위치가 맞는지

적합한 실행:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/mujoco
python -m mujoco.viewer --mjcf=pingpong_rl/assets/scene.xml
```

또는 현재 저장소의 interactive viewer 경로.

이건 `생김새와 기본 동작 확인용`이다.

### 6.2 passive scripted viewer

목적:
- Python 루프에서 직접 step을 돌리며
- scripted controller나 정책 출력을 같이 보기

macOS에서 중요:
- `launch_passive`는 `mjpython`으로 실행해야 한다.

현재 저장소 예시:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/mujoco
mjpython pingpong_rl/scripts/run_viewer.py --mode passive --demo-controller ee --ee-axis z --demo-amplitude 0.03 --demo-frequency 0.5
```

이건 `내 코드가 step을 돌리고 viewer는 따라오는 구조`다.

### 6.3 headless training

목적:
- 최대한 빨리 데이터 수집하고 학습하기

적합한 실행:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/mujoco
python pingpong_rl/scripts/run_ppo_baseline.py --total-timesteps 10000 --device cpu
```

또는 later:

```bash
python pingpong_rl/scripts/run_ppo_baseline.py --total-timesteps 10000 --device mps
```

이게 실제 학습 기본 루프가 되어야 한다.

### 6.4 checkpoint render

목적:
- 학습된 정책이 실제로 무슨 행동을 하는지 보기

현재 저장소 예시:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/mujoco
mjpython pingpong_rl/scripts/run_ppo_render.py --model-path docs/etc/ppo_runs/ppo_smoke/ppo_smoke_model.zip --episodes 3
```

즉 권장 루프는 아래다.

- 학습은 headless
- 확인은 checkpoint render

`학습과 viewer를 항상 같이 돌리는 것`은 가능은 하지만 보통 비추천이다.

## 7. “학습시키는 걸 보면서 학습”은 가능한가

가능은 하다. 하지만 기본 선택으로는 좋지 않다.

왜 느려지나:
- viewer 렌더링 비용
- `viewer.sync()` 비용
- `time.sleep()`로 wall-clock에 맞추게 되는 문제
- 사람이 보게 만들수록 rollout 속도가 크게 줄어듦

왜 더 헷갈리나:
- viewer 입력 문제와 RL 문제를 같이 보게 됨
- real-time 렌더를 보면 정책이 멍청해 보이지만 실제 로그는 조금씩 좋아질 수 있음
- 반대로 운 좋게 한두 번 맞는 장면만 보고 잘 되고 있다고 착각할 수도 있음

실무적으로는 아래가 낫다.

1. headless로 학습
2. 일정 step마다 checkpoint 저장
3. checkpoint를 따로 render
4. TensorBoard와 CSV로 수치 확인

## 8. 왜 viewer 키보드가 이상하게 느껴지나

이건 네가 느낀 게 맞다. 이유는 구조가 다르기 때문이다.

### 8.1 standalone viewer와 passive viewer는 다르다

`python -m mujoco.viewer`는 MuJoCo의 standalone viewer다.

특징:
- MuJoCo가 내부 루프를 관리한다.
- 기본 keybinding은 MuJoCo 쪽 규칙을 따른다.

반면 `launch_passive`는

- 네 Python 스크립트가 step을 돌린다.
- viewer는 sync될 때만 상태를 반영한다.
- macOS에서는 반드시 `mjpython`이 필요하다.

즉 둘은 겉보기는 비슷해도 이벤트 처리 방식이 다르다.

### 8.2 현재 저장소 `viewer.py`는 custom key_callback을 등록하지 않는다

현재 `pingpong_rl/src/pingpong_rl/viewer.py`를 보면 `launch_passive(..., key_callback=...)`를 쓰지 않는다.

즉 현재 구조에서 믿을 수 있는 것은 거의 아래뿐이다.

- viewer 기본 GUI 조작
- `viewer.sync()`로 반영되는 상태 변화
- 내부 `run` 플래그를 읽어서 pause 상태를 감지하는 현재 구현

그래서 `delete/backspace` 같은 키가 네 기대대로 동작하지 않는 것은 이상한 일이 아니다. 현재 스크립트가 그 키를 처리하도록 구현되어 있지 않기 때문이다.

### 8.3 macOS에서 더 헷갈리는 이유

macOS는 passive viewer에서 렌더링 main thread 제약이 있어서 `mjpython`이 필요하다. 이 플랫폼 제약 때문에

- `python`으로는 되는데 `mjpython`에서 다르게 느껴지거나
- 반대로 passive viewer는 `mjpython`이어야만 안정적인데
- standalone viewer는 `python -m mujoco.viewer`로도 뜨는

식으로 경험이 섞여서 더 헷갈리기 쉽다.

정리하면:
- scene inspection: `python -m mujoco.viewer`
- passive scripted loop: `mjpython script.py`
- headless training: 일반 `python`

으로 나눠 생각하면 된다.

## 9. 현재 저장소 구조는 나쁘지 않다

현재 구조:

```text
pingpong_rl/
  assets/
  scripts/
  src/pingpong_rl/
    controllers/
    envs/
    training/
    utils/
  tests/
```

이 구조는 계속 써도 된다.

왜 괜찮나:
- assets 분리
- scripts 분리
- library code는 `src/pingpong_rl` 아래에 모임
- env / controller / training 관심사가 나뉨

즉 지금 막 바꿔야 할 것은 패키지 구조가 아니라 `task 정의`다.

## 10. 다만 upward bounce 목표에 맞게 추가하면 좋은 파일들

현재 구조를 유지하면서 아래 파일을 추가하는 방향이 좋다.

추천 추가 파일:

- `pingpong_rl/src/pingpong_rl/controllers/scripted_bounce_controller.py`
  - 고정 drop에서 최소한 한 번 위로 보내는 baseline

- `pingpong_rl/src/pingpong_rl/envs/bounce_task_env.py`
  - single-bounce / multi-bounce 목표에 맞춘 env

- `pingpong_rl/src/pingpong_rl/training/train_sac.py`
  - SAC baseline 학습 진입점

- `pingpong_rl/src/pingpong_rl/training/eval.py`
  - checkpoint 평가 공통 함수

- `pingpong_rl/scripts/run_train_sac.py`
  - CLI 실행용 학습 스크립트

- `pingpong_rl/scripts/run_eval_checkpoint.py`
  - 특정 체크포인트 render/eval

- `pingpong_rl/tests/test_bounce_task_env.py`
  - reward, termination, observation contract 테스트

즉 지금 구조를 버릴 필요는 없고, `탁구 경기`가 아니라 `upward bounce`에 맞는 env와 baseline을 별도로 명시하면 된다.

## 11. 어떤 프레임워크와 라이브러리가 필요한가

현재 목표 기준 필수는 아래 정도면 충분하다.

### 11.1 필수

- `mujoco`
  - physics, viewer

- `gymnasium`
  - env interface

- `stable-baselines3`
  - PPO, SAC, TD3 등 baseline 학습

- `numpy`
  - 수치 처리

- `tensorboard`
  - 학습 로그 확인

### 11.2 있으면 좋은 것

- `sb3-contrib`
  - TQC 같은 추가 알고리즘

- `matplotlib`
  - 간단한 로그 시각화

- `pandas`
  - CSV 분석

### 11.3 지금 당장 필요 없는 것

- ROS2 통합
- 카메라 perception stack
- JAX/MJX
- imitation learning 프레임워크
- 대규모 experiment tracker

즉 네가 지금 감을 잡고 첫 결과를 내는 데는 MuJoCo + Gymnasium + SB3면 충분하다.

## 12. 알고리즘은 무엇부터 볼까

네 문제는 continuous control이다. 선택지는 크게 두 개다.

### 12.1 PPO

장점:
- 이미 저장소에 baseline이 있다.
- 구현과 로그가 단순하다.
- 병렬 env와 잘 맞는다.

단점:
- sample efficiency가 낮다.
- contact task에서 step 예산이 많이 든다.

적합한 경우:
- env contract 검증
- 첫 smoke baseline

### 12.2 SAC

장점:
- continuous control에서 강한 baseline이다.
- sample efficiency가 PPO보다 보통 낫다.
- 단일 env에서도 의미 있는 결과를 보기가 더 쉽다.

단점:
- replay buffer와 off-policy 특성을 이해해야 한다.
- reward 설계가 이상하면 역시 잘 안 된다.

적합한 경우:
- 네 upward bounce처럼 연속 action과 shaped reward가 있는 문제

### 12.3 그래서 무엇부터?

선택지 비교:

옵션 A: PPO 먼저
- 장점: 현재 저장소와 가장 자연스럽게 이어진다.
- 단점: 학습이 느리거나 신호가 약할 수 있다.

옵션 B: SAC 바로 시도
- 장점: 문제 특성상 더 잘 맞을 가능성이 있다.
- 단점: 현재 스크립트와는 조금 더 떨어져 있다.

현재 목표가 `감 잡기`라면,
- PPO로 env와 logging smoke run을 먼저 확인하고
- 본 학습은 SAC로 옮기는 흐름

이 가장 현실적이다.

## 13. observation, action, reward는 어떻게 잡아야 하나

### 13.1 현재 action은 괜찮은 출발점이다

현재 action:
- `delta xyz` on `racket_center`

이건 시작점으로 적절하다.

지금 바로 넣지 말아도 되는 것:
- full orientation action
- raw joint torque

나중에 성능이 막히면 그때 아래를 검토한다.

- orientation 1축 추가
- racket velocity target 추가

### 13.2 observation은 약간 더 보강할 수 있다

현재 observation:
- joint positions
- joint velocities
- racket position
- target position
- ball position
- ball velocity

upward bounce 기준 다음 후보:
- racket linear velocity
- racket orientation 또는 paddle normal
- last contact 이후 경과 step
- ball-to-racket relative position
- ball-to-racket relative velocity

중요한 원칙:
- 관측은 목표와 직접 관련된 정보 위주로
- 너무 빨리 고차원으로 키우지 않기

### 13.3 reward는 단계별로 바꾸는 것이 좋다

처음부터 `keep bouncing forever` reward만 두면 너무 어렵다.

권장 단계:

1. single-bounce reward
   - contact bonus
   - contact 직후 `ball_velocity_z` 보상
   - floor penalty

2. improved single-bounce reward
   - 최고 높이 보상
   - XY drift penalty
   - contact 타이밍 안정성 보상

3. multi-bounce reward
   - 연속 contact count 보상
   - 두 번째, 세 번째 contact에 추가 가중치

핵심은 `첫 성공을 빠르게 만든 뒤`, 그 다음에 `반복 유지`로 넘어가는 것이다.

## 14. 왜 scripted baseline이 꼭 필요한가

이건 RL보다 먼저 해야 한다.

scripted baseline이란:
- 공이 특정 패턴으로 떨어질 때
- 공이 떨어질 예상 지점으로 라켓을 보내고
- 적절한 순간에 위로 들어올리는 hand-designed controller

이것이 필요한 이유:

1. 물리 파라미터가 말이 되는지 알 수 있다.
2. 라켓 배치가 맞는지 확인할 수 있다.
3. reward가 목표 행동을 잘 포착하는지 볼 수 있다.
4. RL이 실패했을 때 원인 분리가 가능하다.

이건 `치트`가 아니다. 오히려 RL 프로젝트를 성립시키는 기본 안전장치다.

네가 느낀 `뭔가 안 되는 거 같은데 감이 안 잡힘`은 보통 scripted baseline이 없을 때 생긴다.

## 15. 실제 진행 순서

이 순서로 가는 것이 가장 안정적이다.

### 단계 0. 아주 간단한 RL 감 익히기

MuJoCo 과제 전에 아래 둘 중 하나를 먼저 해보면 좋다.

- `Pendulum-v1` + SB3 SAC/PPO
- SB3 custom env tutorial 따라하기

이 단계의 목적은 로봇이 아니라 아래 API를 익히는 것이다.

- `env.reset()`
- `env.step()`
- `model.learn()`
- checkpoint 저장
- TensorBoard 보기

### 단계 1. 현재 env를 랜덤 액션으로 점검

해야 할 것:
- random action rollout
- observation shape 확인
- termination, truncation, failure reason 확인
- NaN 여부 확인

이 단계에서 `check_env`도 같이 보는 편이 좋다.

### 단계 2. scripted single-bounce baseline 만들기

목표:
- 아주 좁은 초기 분포에서 최소한 1회 upward bounce 만들기

성공 기준 예시:
- 10 episode 중 7 episode 이상에서 첫 contact 후 `ball_velocity_z > threshold`

### 단계 3. RL로 single-bounce 풀기

목표:
- baseline보다 덜 brittle한 정책 얻기
- 약간의 XY noise와 velocity noise에 견디기

### 단계 4. reward를 multi-bounce 쪽으로 확장

목표:
- 연속 2회, 3회, 5회 바운스

### 단계 5. curriculum 확대

목표:
- 시작 높이 다양화
- XY offset 확대
- 약한 수평 속도 추가

즉 처음부터 `계속 하늘 방향으로 치기` 전체를 한 번에 풀지 말고,

- 1회 성공
- 반복 성공
- 다양한 초기조건

순서로 늘리는 것이 맞다.

## 16. 현재 저장소에서 바로 써먹을 수 있는 실행 루프

### 16.1 설치 확인

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/mujoco
python -m pip install -e .
```

### 16.2 scene과 제어 확인

```bash
mjpython pingpong_rl/scripts/run_viewer.py --mode passive --demo-controller ee --ee-axis z --demo-amplitude 0.03 --demo-frequency 0.5
```

### 16.3 bounce baseline 확인

```bash
python pingpong_rl/scripts/run_bounce_baseline.py --episodes 5 --max-steps 1200
```

### 16.4 PPO smoke run

```bash
python pingpong_rl/scripts/run_ppo_baseline.py --total-timesteps 10000 --max-episode-steps 300 --ball-height 0.22 --device cpu --run-name ppo_smoke_cpu
```

또는

```bash
python pingpong_rl/scripts/run_ppo_baseline.py --total-timesteps 10000 --max-episode-steps 300 --ball-height 0.22 --device mps --run-name ppo_smoke_mps
```

### 16.5 학습 결과 render

```bash
mjpython pingpong_rl/scripts/run_ppo_render.py --model-path docs/etc/ppo_runs/ppo_smoke_cpu/ppo_smoke_cpu_model.zip --episodes 3
```

### 16.6 TensorBoard 보기

```bash
tensorboard --logdir /Users/pilt/project-collection/ros2/mujoco/pingpong_rl/docs/etc/ppo_runs
```

## 17. 자주 생기는 오해 정리

### 17.1 `mjpython`으로 MuJoCo를 켜야 학습도 되는가

아니다.

- passive viewer를 띄우는 스크립트는 macOS에서 `mjpython`이 필요하다.
- headless training은 일반 `python`으로 돌려도 된다.

즉 `mjpython`은 학습 자체의 필수 조건이 아니라 `macOS GUI 렌더링 제약` 때문에 필요한 것이다.

### 17.2 viewer를 보면서 학습해야 하나

아니다. 오히려 기본값으로는 하지 않는 편이 낫다.

### 17.3 제어기를 먼저 만드는 건 RL 포기가 아닌가

아니다.

제어기를 먼저 만든다는 것은
- 문제를 잘게 쪼개고
- 정책이 배워야 할 고수준 결정을 남기겠다는 뜻

이다. 이건 오히려 정상적인 RL 설계다.

### 17.4 지금 패키지 구조를 갈아엎어야 하나

아니다. 현재 구조는 유지해도 된다. 필요한 것은 새로운 task env와 baseline controller를 추가하는 것이다.

## 18. 이 문서 기준의 권장 선택지

지금 바로 선택해야 할 것들은 아래다.

### 선택지 1. 알고리즘

옵션 A: PPO 먼저 유지
- 장점: 현재 저장소와 가장 자연스럽다.
- 단점: sample efficiency가 아쉽다.

옵션 B: SAC 추가
- 장점: continuous bounce task에 더 잘 맞을 가능성이 높다.
- 단점: 새 학습 스크립트를 추가해야 한다.

### 선택지 2. device

옵션 A: CPU
- 장점: 단순하고 재현성이 좋다.

옵션 B: MPS
- 장점: policy update가 빨라질 수 있다.
- 단점: env stepping이 CPU라 전체 체감은 제한적일 수 있다.

### 선택지 3. 첫 목표

옵션 A: single-bounce 성공률
- 장점: 가장 현실적이다.

옵션 B: 2~3회 연속 bounce
- 장점: 프로젝트다운 데모가 빨리 보인다.
- 단점: 난이도가 바로 올라간다.

현재 시점의 가장 안정적인 조합은 아래다.

- 첫 목표는 single-bounce
- 알고리즘은 PPO smoke 후 SAC 비교
- device는 CPU와 MPS를 둘 다 짧게 비교

## 19. 최종 정리

네가 지금 막막한 이유는 이상한 것이 아니라, `viewer`, `제어`, `env`, `학습 알고리즘`, `물리 튜닝`이 머릿속에서 한 덩어리로 섞여 있기 때문이다.

분리해서 보면 훨씬 단순하다.

- MuJoCo는 physics와 viewer
- controller는 라켓 중심을 움직이는 저수준 레이어
- env는 RL 문제 정의
- SB3는 학습기
- viewer는 학습기와 분리된 확인 도구

그리고 가장 중요한 점은 이것이다.

`upward bounce RL은 “규칙만 주고 알아서 해”보다, “좁은 초기 분포 + 적당한 제어 인터페이스 + scripted baseline + 단계적 reward”로 가야 실제로 풀린다.`

즉 지금 네가 해야 할 다음 일은 더 많은 알고리즘을 찾는 것이 아니라,

- single-bounce 기준을 고정하고
- scripted baseline을 만들고
- PPO smoke와 SAC 후보를 비교할 수 있게 만드는 것

이다.