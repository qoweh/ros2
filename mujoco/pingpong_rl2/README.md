# pingpong_rl2

`pingpong_rl2`는 MuJoCo에서 Franka Panda 로봇팔이 라켓으로 탁구공을 계속 받아 올리는 keep-up/self-rally 강화학습 패키지다. 현재 저장소의 핵심 실험 코드는 이 디렉터리에 있다.

## 현재 기준

- 권장 환경: conda `mujoco_env`
- 현재 발표/웹 기준 모델: `artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip`
- 후속 polish 후보: `artifacts/ppo_runs/keep1_v40_17d_v39_polish/keep1_v40_17d_v39_polish_model.zip`
- 주력 preset: `contact_frame_self_rally_v32_17d_v30_transfer`
- 주력 action mode: `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual`
- 기본 scene: `assets/scene.xml`

## 구조

| 경로 | 역할 |
| --- | --- |
| `src/pingpong_rl2/envs/` | MuJoCo sim wrapper, keep-up MDP, Gymnasium wrapper |
| `src/pingpong_rl2/controllers/` | Cartesian pose controller와 heuristic baseline |
| `src/pingpong_rl2/training/` | PPO preset/config, vector env, bootstrap/evaluation helpers |
| `src/pingpong_rl2/utils/` | artifact/model/path 해석 |
| `scripts/` | 학습, 평가, rebound 분석, viewer, sanity/benchmark entrypoint |
| `configs/` | 긴 학습 command를 줄이기 위한 JSON config |
| `artifacts/ppo_runs/` | 모델 zip, monitor CSV, training summary, 분석 결과 |
| `docs/report/` | v1~v54 실험 보고서 |
| `docs/rl_presentation_pack/` | 발표용 데이터, 그래프, 슬라이드 outline |

## 설치와 검증

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
conda activate mujoco_env
python -m pip install -e .
```

테스트:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest discover -s tests
```

Python compile check:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m compileall -q src scripts tests
```

## 자주 쓰는 실행

v39 모델 viewer:

```bash
PYTHONPATH=src mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --episodes 5
```

v39 rebound 분석:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --episodes 20 \
  --episode-step-limit 7200 \
  --analysis-name keep1_v39_readme_eval20
```

v40 polish 후보 확인:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v40_17d_v39_polish/keep1_v40_17d_v39_polish_model.zip \
  --episodes 20 \
  --episode-step-limit 7200 \
  --analysis-name keep1_v40_readme_eval20
```

새 실험을 만들 때는 `configs/keep1_v32_17d_transfer.json`를 기반으로 run version과 resume model을 명시한다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/keep1_v32_17d_transfer.json \
  --run-version v41_experiment_name \
  --resume-from artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --total-timesteps 300000
```

짧은 wiring check:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/keep1_v32_17d_transfer.json \
  --run-version smoke_check \
  --smoke \
  --reset-model \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0
```

## 읽을 문서

- `docs/report/00_index.md`: 전체 실험 보고서 인덱스
- `docs/report/54_v32_17d_transfer_finetune_report.md`: 17D action transfer 기준점
- `docs/model_evolution_to_v39.md`: v39까지 모델 진화 흐름
- `docs/script_entrypoints/README.md`: 실행 파일별 역할
- `docs/rl_presentation_pack/README.md`: 발표 자료 패키지

## 웹 프로젝트와의 관계

웹 시연 저장소 `/Users/pilt/project-collection/pingpong`는 이 패키지를 vendoring해서 사용한다. 웹 런타임에서는 모델 이름을 `keep_v39_17d`로 짧게 복사해 두지만, 원본 학습 계보는 이 디렉터리의 `keep1_v39_17d_mid_curriculum_fixed`이다.
