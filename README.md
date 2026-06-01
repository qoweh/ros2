# ROS2 Graduation Project Workspace

이 repository에는 ROS2/로봇팔 실험 관련 작업물이 들어 있다. 현재 진행 중인 강화학습 프로젝트는 `graduation-prj/pingpong_rl2`다.

- `graduation-prj/pingpong_rl2`: 현재 작업 중인 MuJoCo 기반 탁구공 keep-up/self-rally 강화학습 프로젝트
- `graduation-prj/pingpong_rl`: 이전 버전 프로젝트. 현재 기준 문서/실험은 `pingpong_rl2`를 우선한다.

## 목표

로봇팔 끝에 탁구채를 붙인 상태에서 탁구공을 계속 위로 올려치는 강화학습 정책을 만든다. 목표는 상대 코트로 보내는 탁구가 아니라, 공이 다시 라켓 근처로 적절한 높이와 시간 뒤에 떨어지도록 만드는 self-rally/keep-up이다.

## 기본 환경

권장 환경은 conda `mujoco_env`다.

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

conda create -n mujoco_env python=3.10
conda activate mujoco_env

python -m pip install -U pip
python -m pip install -e .
python -m pip install torch tensorboard pandas matplotlib pytest
```

`pingpong_rl2/pyproject.toml`의 핵심 Python dependency:

- `gymnasium`
- `mujoco`
- `numpy`
- `stable-baselines3`

Stable-Baselines3가 PyTorch를 사용하므로 `torch`가 필요하다. 환경에 따라 SB3 설치 과정에서 같이 설치되지만, 명시적으로 설치해두는 편이 편하다.

## MuJoCo 확인

Python package `mujoco`를 사용한다.

```bash
conda activate mujoco_env
python -c "import mujoco; print(mujoco.__version__)"
```

macOS에서 MuJoCo viewer를 띄울 때는 보통 `mjpython`을 사용한다.

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/<run>/<run>_model.zip \
  --episodes 20
```

사용자 로컬 환경에 `mujoco_activate` helper가 있으면 그것을 써도 된다. 다른 사람이 새로 세팅할 때는 위 conda/pip 흐름을 기준으로 한다.

## 자주 쓰는 명령

작업 디렉토리:

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
conda activate mujoco_env
```

테스트:

```bash
PYTHONPATH=src python -m unittest \
  tests/test_scene_load.py \
  tests/test_keepup_env.py \
  tests/test_vector_env.py \
  tests/test_ppo_runs.py \
  tests/test_keepup_contract_features.py
```

현재 self-rally 후보 학습:

```bash
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v3 \
  --reset-model \
  --total-timesteps 2000000
```

현재 self-rally preset은 기본 checkpoint를 만들지 않는다. 마지막 모델은 아래처럼 생긴다.

```text
artifacts/ppo_runs/pmk_cf_self_rally_v3/pmk_cf_self_rally_v3_model.zip
```

중간 checkpoint/best model이 꼭 필요할 때만 학습 명령에 추가한다.

```bash
--checkpoint-interval 50000 --checkpoint-eval-episodes 20
```

학습 결과 시각화:

```bash
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v3/pmk_cf_self_rally_v3_model.zip \
  --episodes 100
```

rebound/contact 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v3/pmk_cf_self_rally_v3_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v3_quality100
```

TensorBoard:

```bash
tensorboard --logdir artifacts/ppo_runs/<run>/tb
```

`tb/PPO_*/events.out.tfevents...` 파일은 TensorBoard binary event log다. 텍스트로 읽는 파일이 아니며, 삭제해도 모델 실행에는 영향이 없다.

## 주요 파일 역할

- `graduation-prj/pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
  - MuJoCo keep-up 환경, reward/success contract, contact/rebound 진단값, planner/primitive 로직의 중심 파일
- `graduation-prj/pingpong_rl2/src/pingpong_rl2/envs/gym_env.py`
  - Stable-Baselines3/Gymnasium용 wrapper
- `graduation-prj/pingpong_rl2/src/pingpong_rl2/training/vector_env.py`
  - `SyncVectorEnv`/`AsyncVectorEnv`를 SB3 `VecEnv`처럼 쓰기 위한 adapter
- `graduation-prj/pingpong_rl2/scripts/run_ppo_learning.py`
  - PPO 학습 entrypoint, preset/CLI 옵션, checkpoint/final evaluation 처리
- `graduation-prj/pingpong_rl2/scripts/run_viewer.py`
  - 학습된 모델을 MuJoCo viewer로 확인
- `graduation-prj/pingpong_rl2/scripts/run_ppo_rebound_analysis.py`
  - contact 이후 공 궤적, next intercept, useful contact 품질 분석
- `graduation-prj/pingpong_rl2/scripts/run_heuristic_keepup_diagnostic.py`
  - heuristic/primitive smoke diagnostic
- `graduation-prj/pingpong_rl2/artifacts/`
  - 학습 모델, 분석 CSV/JSON, TensorBoard log 등 생성 산출물

## 문서

보고서 인덱스:

- [graduation-prj/pingpong_rl2/docs/report/00_index.md](graduation-prj/pingpong_rl2/docs/report/00_index.md)

현재 self-rally 설계와 진단에서 특히 중요한 문서:

- [24_self_rally_planner_primitive_report.md](graduation-prj/pingpong_rl2/docs/report/24_self_rally_planner_primitive_report.md)
- [25_cleanup_and_self_rally_status.md](graduation-prj/pingpong_rl2/docs/report/25_cleanup_and_self_rally_status.md)
- [26_learning_runtime_parallel_and_v2_diagnosis.md](graduation-prj/pingpong_rl2/docs/report/26_learning_runtime_parallel_and_v2_diagnosis.md)
- [27_self_rally_execution_stabilization_report.md](graduation-prj/pingpong_rl2/docs/report/27_self_rally_execution_stabilization_report.md)

## 현재 중요한 판단 기준

학습 로그의 `ep_rew_mean`만 보지 말고 아래 지표를 우선 확인한다.

- `mean_useful_bounces`
- `two_or_more_useful_bounce_rate`
- `three_or_more_useful_bounce_rate`
- `useful_contact_rate`
- `next_intercept_reachable_rate`
- `useful_contact_mean_next_intercept_xy_error`
- `useful_contact_mean_ball_lateral_speed`
- `robot_body_contact_rate`
- `ball_out_of_bounds_rate`

현재 병목은 reward 숫자만의 문제가 아니라, 라켓이 제때 안전한 contact 자세에 들어가고 팔 몸체가 공 경로를 막지 않는 low-level execution 문제다. 최신 self-rally preset은 `body clearance`, `recovery posture`, `strike-window stabilization`을 포함한다.
