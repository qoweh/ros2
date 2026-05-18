# pingpong_rl 빠른 안내

## 먼저 이렇게 이해하면 된다

이 디렉토리에는 크게 세 종류의 코드가 있다.

- 물리 장면과 MuJoCo 시뮬레이터 코드
- 제어기와 RL 환경 코드
- 바로 실행하는 스크립트

현재 목표는 `탁구 경기`가 아니다. 로봇팔이 라켓으로 공을 바닥에 떨어뜨리지 않고 계속 튕기도록 만드는 `keep-up` 과제가 현재 기준 목표다.

이 순서대로 읽으면 훨씬 덜 헷갈린다.

## 가장 간단한 구분

`run_bounce_baseline.py`와 `run_ppo_baseline.py`는 역할이 완전히 다르다.

- `run_bounce_baseline.py`: 학습하지 않는다. 공을 떨어뜨리고 MuJoCo를 step 하면서 공이 라켓에 먼저 맞는지, 바닥에 먼저 닿는지를 보는 물리 sanity check다.
- `run_keepup_baseline.py`: 학습하지 않는다. 현재 공 위치와 속도로부터 라켓 목표점을 계산해서, 공을 계속 살려 두는 heuristic baseline을 실행한다.
- `run_ppo_baseline.py`: 학습한다. EE delta 환경을 Gymnasium 형식으로 감싸서 PPO 정책을 학습한다.

짧게 말하면:

- `bounce_baseline` = 물리/장면 점검용
- `keepup_baseline` = 학습 없는 heuristic keep-up 기준선
- `ppo_baseline` = RL 학습 실행용

## `ppo_smoke`는 무엇인가

`ppo_smoke`는 다른 알고리즘이 아니다.

같은 PPO 학습 스크립트를 아주 짧게 돌린 smoke test run 이름일 뿐이다. 목적은 아래와 같다.

- 학습이 실행되는지 확인
- 모델과 로그가 저장되는지 확인
- 긴 baseline 학습 전에 wiring check 수행

즉:

- `ppo_active_hit` = 현재 기본 학습 run. 라켓의 upward velocity/acceleration이 있는 contact만 active hit로 보상한다.
- `ppo_smoke` = 아주 짧은 확인용 run

## 권장 읽기 순서

### 1. 장면과 물리

- `assets/scene.xml`: MuJoCo 장면 정의
- `src/pingpong_rl/envs/pingpong_env.py`: 장면을 로드하고, 공을 reset/spawn 하고, MuJoCo를 step 하고, contact와 failure reason을 판정한다.

### 2. 저수준 제어 보조 코드

- `src/pingpong_rl/controllers/joint_controller.py`: viewer 데모에서 쓰는 간단한 joint target 보관기
- `src/pingpong_rl/controllers/ee_pose_controller.py`: 라켓 중심의 목표 위치 변화를 site Jacobian을 이용해 joint target으로 바꾼다.

### 3. RL 태스크 정의

- `src/pingpong_rl/envs/ee_delta_env.py`: observation, reward, 성공/실패 조건, time limit 규칙을 정의한다.
- `src/pingpong_rl/envs/ee_delta_gym_env.py`: Stable-Baselines3 PPO가 붙을 수 있게 Gymnasium 래퍼를 제공한다.

### 4. 학습 로그

- `src/pingpong_rl/training/ppo_logging.py`: episode, step, contact CSV와 summary JSON, TensorBoard 로그를 저장한다.

### 5. 실행 스크립트

- `scripts/run_viewer.py`: viewer를 열고 hold/joint/EE 데모를 실행한다.
- `scripts/run_bounce_baseline.py`: 학습 없는 bounce sanity check를 실행한다.
- `scripts/run_keepup_baseline.py`: heuristic keep-up baseline을 실행한다.
- `scripts/run_ppo_baseline.py`: PPO 학습을 실행한다.
- `scripts/run_ppo_render.py`: 저장된 PPO 모델을 viewer에서 재생한다.
- `scripts/run_ee_rollout_analysis.py`: constant action rollout을 돌리고 CSV/JSON 분석 파일을 만든다.

### 6. 테스트

- `tests/test_scene_load.py`: scene load, bounce, controller, env contract를 확인하는 집중 테스트다.

## 보통 실제로 실행하는 파일

저장소 루트에서 아래 명령만 기억하면 된다.

```bash
mjpython pingpong_rl/scripts/run_viewer.py
python pingpong_rl/scripts/run_bounce_baseline.py
python pingpong_rl/scripts/run_keepup_baseline.py
python pingpong_rl/scripts/run_ppo_baseline.py
mjpython pingpong_rl/scripts/run_ppo_render.py
python -m unittest discover -s pingpong_rl/tests -p 'test_scene_load.py'
```

## 왜 args가 많아 보이나

실험용 프로젝트라서 CLI args는 남겨두고 있다. 다만 기본값을 넣어 두었기 때문에 평소에는 인자를 추가하지 않고 실행해도 된다.

인자를 바꾸는 경우는 보통 아래 정도다.

- 짧은 smoke 학습을 돌릴 때
- 공 시작 높이를 바꿔볼 때
- PPO rollout 길이를 바꿔볼 때
- 기본 모델이 아닌 다른 저장 모델을 렌더링할 때
- 학습 장치를 명시적으로 바꾸고 싶을 때

현재 `run_ppo_baseline.py`의 기본 `--device`는 `cpu`다. 필요하면 다른 장치를 직접 지정할 수 있지만, 현재 기본 워크플로는 그냥 CPU 기준으로 보면 된다.

```bash
python pingpong_rl/scripts/run_ppo_baseline.py --device cpu
```

## `python -m mujoco.viewer`로 보면 왜 자세가 다르나

`python -m mujoco.viewer --mjcf=...`는 MuJoCo 기본 viewer를 직접 여는 것이다. 이 경로는 프로젝트 코드의 reset 로직을 거치지 않는다.

반면 `mjpython pingpong_rl/scripts/run_viewer.py`는 프로젝트 viewer 경로를 타서 아래 작업을 먼저 수행한다.

- `PingPongSim.reset()` 호출
- model의 `home` keyframe 적용
- `ctrl`에 home target 적용
- 공을 라켓 위로 다시 spawn

그래서 raw viewer에서는 로봇이 기본 XML 상태대로, 바닥에 수직으로 펴진 것처럼 보일 수 있다. 프로젝트 viewer에서는 home 자세가 먼저 적용돼서 네가 기대한 자세로 보인다.

즉:

- `python -m mujoco.viewer` = MuJoCo 기본 viewer, 프로젝트 reset 미적용
- `mjpython pingpong_rl/scripts/run_viewer.py` = 프로젝트 viewer, home 자세와 공 reset 적용

## 학습할 때도 raw viewer 환경을 쓰나

아니다. 학습은 raw viewer를 쓰지 않는다.

학습 경로는 아래다.

- `scripts/run_ppo_baseline.py`
- `PingPongEEDeltaGymEnv`
- `PingPongEEDeltaEnv.reset()`
- `PingPongSim.reset()`

즉 학습은 viewer 없이 headless로 진행되고, episode가 시작될 때마다 프로젝트 reset 로직을 정상적으로 탄다. 그래서 네가 `python -m mujoco.viewer`에서 본 raw 자세가 학습 환경 그대로라고 보면 안 된다.

## 가장 단순한 사용 순서

아래 순서로 보면 가장 덜 헷갈린다.

1. viewer를 실행해서 장면이 살아 있는지 확인한다.
2. `run_bounce_baseline.py`로 공과 라켓의 기본 접촉이 이상하지 않은지 확인한다.
3. `run_keepup_baseline.py`로 학습 없는 기준선이 어느 정도 공을 살려 두는지 본다.
4. `run_ppo_baseline.py`로 학습한다.
5. `run_ppo_render.py`로 저장된 모델을 본다.
6. `docs/etc/ppo_runs/ppo_active_hit/ppo_active_hit_training_summary.json`을 열어 결과를 확인한다.
