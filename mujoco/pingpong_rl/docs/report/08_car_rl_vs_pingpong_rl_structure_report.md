# car_rl와 pingpong_rl 비교 정리

## 1. 이 문서의 목적

이 문서는 `TODO2.md`에 적힌 세 가지 질문에 한 번에 답하기 위해 만들었다.

질문은 크게 세 가지였다.

- `car_rl/test.py`의 `--headless`와 학습 중 평가의 관계
- `car_rl/report/02_car_env_mujoco_gymnasium.md`에서 말하는 "이 환경은 `mj_step()` 중심이 아니다"의 의미
- `car_rl` 프로젝트와 `pingpong_rl` 프로젝트의 구조 차이

결론부터 말하면:

- `car_rl/test.py`는 "학습이 끝난 모델을 별도로 평가하거나 눈으로 확인하는 스크립트"다.
- `car_rl`의 차량은 MuJoCo actuator 물리로 굴리는 차가 아니라, Python이 계산한 차량 운동학을 MuJoCo 상태에 반영하는 방식이다.
- `car_rl`은 학습 개념을 익히기 쉬운 토이 프로젝트이고, `pingpong_rl`은 시뮬레이터, 제어기, Gym 래퍼, PPO 로깅까지 분리한 실험용 프로젝트에 가깝다.

## 2. `car_rl/test.py`의 `--headless`는 무엇인가

`car_rl/test.py`는 이미 저장된 PPO 모델을 불러와서 평가하는 스크립트다.

핵심 분기점은 이것이다.

```python
viewer = None
if not args.headless:
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
```

즉:

- `--headless`를 주면 viewer 없이 평가만 한다.
- `--headless`를 주지 않으면 같은 평가를 하되 MuJoCo viewer에 그 과정을 보여준다.

여기서 중요한 점은 `headless` 여부가 "평가를 하느냐 마느냐"를 바꾸는 것이 아니라, "평가를 화면에 보이게 하느냐"만 바꾼다는 점이다.

## 3. 그런데 평가는 학습할 때 이미 하지 않나

그렇다. `car_rl/train.py`는 학습 중 평가를 이미 수행한다.

`train.py` 안에는 `EvalCallback`이 있다.

```python
callback = EvalCallback(
    eval_env,
    best_model_save_path=str(args.log_dir / "best_model"),
    log_path=str(args.log_dir / "eval"),
    eval_freq=args.eval_freq,
    n_eval_episodes=args.eval_episodes,
    deterministic=True,
    render=False,
)
```

이 말은 곧:

- 학습 도중에 일정 주기마다 별도 평가 환경에서 평가를 한다.
- 기본값으로는 `10,000` step마다 평가한다.
- 평가할 때마다 `10` episode를 돌린다.
- 가장 성능이 좋은 모델은 `runs/best_model/best_model.zip`에 저장된다.

그런데도 `test.py`가 따로 필요한 이유는 역할이 다르기 때문이다.

학습 중 평가의 역할:

- 현재 정책이 좋아지고 있는지 자동으로 측정
- best model 저장
- 학습 루프 안에서 빠르게 성능 추적

학습 후 `test.py`의 역할:

- 저장된 모델을 사람이 다시 확인
- 원하는 횟수만큼 반복 평가
- viewer로 실제 행동을 눈으로 관찰
- 특정 목표 좌표와 yaw를 강제로 넣어서 정책 반응 확인

즉 `train.py`의 평가는 자동 점검이고, `test.py`는 사후 확인용 수동 평가다.

추가로 하나 더 짚어야 할 점이 있다.

- `train.py`는 학습 끝에 `car_ppo_nav.zip`을 저장한다.
- `EvalCallback`은 별도로 가장 좋았던 모델을 `runs/best_model/best_model.zip`에 저장한다.

그래서 `test.py` 기본 실행은 "최종 모델"을 보는 것이지, 자동으로 "best model"을 보는 것은 아니다. best model을 보고 싶으면 `--model-path`를 따로 넘겨야 한다.

## 4. PPO 과정에서 알아야 할 최소 용어

질문에 나온 `epoch`, `n_step`, `timestamp` 비슷한 용어를 현재 `car_rl/train.py` 기준으로 정리하면 아래와 같다.

### 4.1 `total_timesteps`

```python
model.learn(total_timesteps=args.timesteps, callback=callback)
```

이 값은 학습 전체에서 환경과 상호작용하는 총 step 수다.

현재 기본값은 `120_000`이다.

쉽게 말하면:

- 에이전트가 `action`을 내고
- 환경이 `obs, reward, terminated, truncated, info`를 돌려주는
- 이 한 번의 상호작용을 `1 timestep`으로 보면 된다.

### 4.2 `n_steps`

```python
n_steps=1024
```

PPO는 매 step마다 바로 가중치를 업데이트하지 않는다. 먼저 일정 길이의 rollout을 모은 다음, 그 묶음을 가지고 업데이트한다.

여기서는:

- `1024` step을 먼저 모으고
- 그 뒤에 PPO 업데이트를 수행한다.

즉 `n_steps`는 "한 번의 PPO 업데이트 전에 모으는 데이터 길이"라고 보면 된다.

### 4.3 `batch_size`

```python
batch_size=256
```

모아둔 rollout 전체를 한 번에 학습하지 않고, 더 작은 minibatch로 쪼개서 gradient update를 한다.

여기서는 `1024` step으로 모은 데이터를 `256` 크기 묶음으로 잘라서 사용한다.

### 4.4 `epoch`

이 스크립트에서는 `n_epochs`를 직접 지정하지 않았지만, PPO 내부에는 "같은 rollout 데이터를 몇 번 반복해서 학습에 사용할지"를 정하는 epoch 개념이 있다.

즉 epoch는 보통 이런 뜻이다.

- `n_steps`만큼 데이터를 모은다.
- 그 데이터를 minibatch로 나눈다.
- 그 minibatch 전체를 여러 번 반복해서 본다.

그 "여러 번 반복"이 epoch다.

현재 `car_rl/train.py`에서는 이 값을 직접 바꾸지 않았으므로, SB3의 PPO 기본값을 따른다.

### 4.5 `eval_freq`

```python
parser.add_argument("--eval-freq", type=int, default=10_000)
```

학습 중 자동 평가를 몇 step마다 할지 정하는 값이다.

### 4.6 `eval_episodes`

```python
parser.add_argument("--eval-episodes", type=int, default=10)
```

평가를 돌릴 때 몇 개 episode를 평균낼지 정한다.

### 4.7 `seed`

```python
parser.add_argument("--seed", type=int, default=7)
```

초기 랜덤성을 재현 가능하게 맞추기 위한 값이다.

### 4.8 `learning_rate`, `gamma`, `gae_lambda`, `ent_coef`

이 값들은 PPO 내부 학습 성질을 조절한다.

- `learning_rate`: 가중치를 얼마나 빠르게 바꿀지
- `gamma`: 먼 미래 보상을 얼마나 중요하게 볼지
- `gae_lambda`: advantage 추정의 bias-variance 절충
- `ent_coef`: 행동 다양성을 얼마나 유지할지

지금 단계에서는 이것들을 전부 외울 필요는 없고, 우선은 아래 순서로 이해하면 충분하다.

1. 환경이 `step()`을 통해 데이터를 만든다.
2. PPO가 `n_steps`만큼 데이터를 모은다.
3. 모은 데이터를 여러 minibatch로 나눠 여러 epoch 학습한다.
4. `eval_freq` 주기마다 평가를 따로 돌린다.
5. 전체 `total_timesteps`를 채우면 학습이 끝난다.

### 4.9 `timestamp`는 무엇인가

현재 `car_rl/train.py`나 `car_rl/test.py`에는 PPO 하이퍼파라미터로서의 `timestamp`는 없다.

만약 TensorBoard 이벤트 파일 이름이나 로그 파일 이름에서 timestamp처럼 보이는 숫자를 봤다면, 그것은 보통:

- 파일 생성 시각
- 실행 구분용 식별자

에 가깝다. PPO의 핵심 학습 단위는 timestamp가 아니라 `timestep`이다.

## 5. `이 환경은 mj_step() 중심이 아니다`의 정확한 뜻

`car_rl/report/02_car_env_mujoco_gymnasium.md`에서 말한 핵심은 이거다.

- 이 차는 MuJoCo의 actuator에 torque나 force를 넣고 물리엔진이 차를 굴리게 하는 구조가 아니다.
- 대신 Python 코드가 차량 운동학을 먼저 계산하고, 그 결과를 MuJoCo 상태에 써 넣는다.

실제로 `car_env.py`의 `step()`은 아래 순서로 돌아간다.

1. throttle과 steering command를 받는다.
2. speed와 steering angle을 Python에서 업데이트한다.
3. `x`, `y`, `yaw`를 bicycle model 비슷한 식으로 Python에서 계산한다.
4. `_sync_pose()`가 이 값을 `self.data.qpos`, `self.data.qvel`에 직접 반영한다.
5. 마지막에 `mujoco.mj_forward()`를 호출해 장면 파생 상태를 다시 계산한다.

중요한 점은 `car_env.py`의 환경 step 안에서 `mujoco.mj_step()`을 호출하지 않는다는 것이다.

즉 질문을 조금 직설적으로 바꾸면, 네 해석이 거의 맞다.

- "MuJoCo 엔진으로 차를 밀어서 움직인다"기보다
- "Python이 차 위치와 방향을 계산해서 MuJoCo에 반영한다"

가 더 정확하다.

다만 완전히 MuJoCo가 필요 없는 것은 아니다.

- MuJoCo XML로 차와 goal 장면을 정의하고
- viewer에 그 상태를 보여주고
- body, geom, mocap pose 같은 파생 상태를 일관되게 계산하는 데 MuJoCo를 사용한다.

즉 `car_rl`의 MuJoCo는 "주된 동역학 엔진"이라기보다 "scene container + 상태 동기화 + 시각화 도구"에 더 가깝다.

## 6. `car_rl`과 `pingpong_rl`의 가장 큰 차이

두 프로젝트는 겉으로 둘 다 MuJoCo + RL처럼 보이지만, 실제 목적과 추상화 수준이 꽤 다르다.

| 항목 | `car_rl` | `pingpong_rl` |
| --- | --- | --- |
| 목표 | 2D 차량을 목표 위치와 yaw로 보내기 | 로봇팔 라켓으로 공을 쳐 올리기 |
| 성격 | 학습용 토이 프로젝트 | 실험용 연구 프로젝트 |
| 물리 사용 방식 | Python이 차량 운동학 계산 후 MuJoCo 상태 동기화 | MuJoCo를 실제 physics stepping에 사용 |
| action 의미 | throttle, steering 2차원 | EE delta 3차원, 이후 joint target으로 변환 |
| 구조 | 파일이 적고 평평함 | 시뮬레이터, 제어기, env, wrapper, training이 분리됨 |
| PPO 로깅 | 기본 `EvalCallback` 중심 | CSV, contact trace, summary JSON, TensorBoard까지 별도 관리 |
| 진입점 | `main.py`, `train.py`, `test.py` | viewer, baseline, rollout analysis, PPO train, PPO render로 세분화 |
| 패키징 | 로컬 스크립트 중심 | `pyproject.toml` 기반 패키지 구조 |

## 7. 왜 `pingpong_rl`이 더 덕지덕지 붙어 보이나

이건 단순히 정리가 안 돼서라기보다, 풀려는 문제가 더 어렵고 실험 관찰 요구가 많아서다.

`pingpong_rl`은 최소한 아래 레이어를 분리해 두고 있다.

### 7.1 시뮬레이터 레이어

`pingpong_rl/src/pingpong_rl/envs/pingpong_env.py`의 `PingPongSim`이 담당한다.

여기서는:

- scene 로드
- ball spawn/reset
- contact 검사
- `mujoco.mj_step()` 반복 수행
- 실패 조건 판정

을 맡는다.

즉 MuJoCo 장면 자체를 다루는 가장 아래 레이어다.

### 7.2 제어기 레이어

`pingpong_rl/src/pingpong_rl/controllers/ee_pose_controller.py`의 `RacketCartesianController`가 있다.

이 레이어는 RL action을 바로 joint 값으로 쓰지 않고,

- 라켓 중심 목표 위치를 관리하고
- Jacobian 기반으로 joint target을 계산한다.

즉 RL이 Franka 7개 관절을 직접 다루지 않게 난이도를 낮춘다.

### 7.3 태스크 레이어

`pingpong_rl/src/pingpong_rl/envs/ee_delta_env.py`의 `PingPongEEDeltaEnv`가 있다.

여기서는:

- observation 구성
- reward 계산
- success/failure 판정
- time limit 처리
- contact 정보 로깅용 info 구성

을 맡는다.

즉 "탁구 과제 규칙"은 여기 있다.

### 7.4 Gym 래퍼 레이어

`pingpong_rl/src/pingpong_rl/envs/ee_delta_gym_env.py`는 위 환경을 Gymnasium 형태에 맞춘다.

이 레이어는 SB3 PPO와 바로 붙이기 위한 어댑터다.

### 7.5 학습 로깅 레이어

`pingpong_rl/src/pingpong_rl/training/ppo_logging.py`는 단순 reward 출력이 아니라:

- episode CSV
- step CSV
- contact CSV
- summary JSON
- TensorBoard

를 함께 저장한다.

이건 `car_rl`보다 훨씬 실험 기록 중심이라는 뜻이다.

## 8. 반대로 `car_rl`은 왜 단순해 보이나

`car_rl`은 의도적으로 단순하다.

핵심이 거의 한 파일 `car_env.py`에 모여 있다.

- 상태 정의
- action 해석
- 목표 샘플링
- reward
- 종료 조건
- MuJoCo 동기화

이 전부 같은 환경 안에 있다.

그리고 실행 경로도 단순하다.

- `main.py`: scripted demo
- `train.py`: PPO 학습
- `test.py`: 저장 모델 평가

즉 `car_rl`은 구조적으로 "개념을 빨리 파악하기 쉬운 예제"에 가깝고, `pingpong_rl`은 "실험을 계속 쌓아가며 분석하기 위한 기반"에 가깝다.

## 9. 두 프로젝트를 한 문장씩 요약하면

- `car_rl`: PPO와 Gymnasium, MuJoCo 연결을 이해하기 위한 작고 직접적인 프로젝트
- `pingpong_rl`: MuJoCo 물리, 로봇 제어, RL 환경 래핑, 학습 로그 분석까지 분리해 둔 실험 프로젝트

그래서 `pingpong_rl`이 더 복잡해 보이는 것은 자연스러운 일이다. 이 복잡함의 상당 부분은 "장난감 예제"에서 "실험 플랫폼"으로 넘어가면서 생긴 레이어다.

## 10. 지금 기준에서의 판단

현재 저장소만 놓고 보면, `pingpong_rl`이 조금 무겁게 느껴지는 판단은 타당하다. 특히 아래 부분은 입문자 입장에서 한 번에 보기 부담스럽다.

- `src` 패키지 구조
- controller 레이어
- Gym wrapper 레이어
- PPO 전용 logging callback
- 분석용 스크립트와 산출물 디렉토리

하지만 그 대부분은 "쓸데없는 장식"이라기보다, 공 contact와 실패 원인을 계속 추적해야 하는 탁구 과제 특성 때문에 붙은 것이다.

즉 정리하면:

- `car_rl`은 배우기 쉬운 대신 확장성이 낮다.
- `pingpong_rl`은 이해 비용이 큰 대신 실험과 분석에는 유리하다.

네가 지금 `pingpong_rl`을 읽기 어렵게 느끼는 핵심 이유는 코드 품질 문제 하나라기보다, 프로젝트 목적이 이미 `car_rl`보다 한 단계 더 연구형으로 올라가 있기 때문이다.

## 11. 마지막 결론

질문 세 개에 대한 짧은 답만 다시 적으면 아래와 같다.

1. `car_rl/test.py`에서 `--headless`는 viewer 없이 평가하는 옵션이고, 옵션이 없으면 같은 평가를 화면으로 보는 것이다.
2. 학습 중 평가는 `train.py`의 `EvalCallback`이 이미 수행하지만, `test.py`는 저장 모델을 따로 확인하고 시각화하기 위해 존재한다.
3. `car_rl`은 Python이 차량 운동학을 계산해서 MuJoCo 상태를 갱신하는 단순 토이 프로젝트이고, `pingpong_rl`은 MuJoCo physics, 제어기, Gym 래퍼, PPO 로깅이 층으로 나뉜 실험 프로젝트다.
