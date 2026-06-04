# keep1_v31 넓은 시작 위치와 keep2 모델 계획

작성일: 2026-06-04

## 1. 시작 위치 랜덤화 확장

현재 발표/웹서비스용 1공 모델은 `keep1_v30`이다.

v30은 `reset_xy_range=0.10m`에서 안정적으로 동작했다.

100 episode 분석:

- mean useful bounces: `30.56`
- max useful bounces: `53`
- 30+ rate: `71%`
- time_limit: `70/100`
- ball_out_of_bounds: `8/100`
- next-intercept reachable rate: `91.4%`

따라서 다음 실험은 구조를 바꾸지 않고 시작 위치만 더 넓힌다.

새 preset:

- `contact_frame_self_rally_v31_keep1_wider_xy`

새 config:

- `configs/keep1_v31.json`

v31 설정:

- base model: `artifacts/ppo_runs/keep1_v30/keep1_v30_model.zip`
- action mode: v30과 같은 15D
- reset XY curriculum: `0.10m -> 0.14m`
- reset velocity: v30 유지
- spin: `0`
- target XY clamp: `±0.14m`
- total timesteps: `1M`

의도:

- v29처럼 action dimension, spin, 초기 속도를 한 번에 바꾸지 않는다.
- v30의 안정적인 keep-up 능력을 보존하면서 눈에 보이는 시작 위치 분포만 넓힌다.
- v31이 크게 무너지면 다음은 `0.10m -> 0.12m`로 줄여서 다시 학습한다.

## 2. v31 학습 명령

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py --config-file configs/keep1_v31.json
```

학습 후 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v31/keep1_v31_model.zip \
  --episodes 100 \
  --seed 231 \
  --episode-step-limit 1800 \
  --analysis-name keep1_v31_eval100
```

viewer:

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/keep1_v31/keep1_v31_model.zip \
  --episodes 100 \
  --max-episode-steps 1800
```

## 3. 2공 모델은 별도 모델이 필요한가?

필요하다.

현재 `keep1` 모델은 observation, reward, termination, planner가 모두 "공 하나"를 전제로 한다. 공 2개를 넣으면 단순히 같은 모델을 재사용하기 어렵다.

2공 keep-up을 학습하려면 최소한 아래가 바뀌어야 한다.

- MuJoCo scene/body: ball body 2개
- simulation wrapper: `ball_position`, `ball_velocity`를 2개 관리
- observation: 공 2개의 위치/속도와 active mask
- planner: 다음에 칠 공을 선택하는 scheduling logic
- reward: 두 공 각각의 useful bounce, out-of-bounds, floor/body contact
- termination: 한 공만 실패해도 종료할지, 두 공 모두 실패해야 종료할지 정의
- model: `keep2_v1`처럼 별도 PPO 모델

웹 UI에서는 아래처럼 나누는 것이 안전하다.

- checkbox off: `keep1_v30` 또는 `keep1_v31`
- checkbox on: 나중에 학습할 `keep2_v1`

한 모델로 1공/2공을 모두 처리하려면 처음부터 `max_balls=2` observation과 mask를 넣은 통합 환경으로 다시 설계해야 한다. 현재 졸업작품 발표 안정성 기준에서는 모델 분리가 낫다.

