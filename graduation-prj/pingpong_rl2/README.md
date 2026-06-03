# pingpong_rl2

`pingpong_rl2`는 기존 `pingpong_rl`을 덮어쓰지 않고, keep-up 문제를 최소 구조로 다시 검증하기 위한 분리 프로젝트다.

## 설계 원칙

- env-side assist는 실험으로만 넣고, 현재 winner만 preset으로 고정한다.
- heuristic keep-up policy는 기본 학습 경로에 직접 섞지 않되, 구조 gate가 통과한 뒤에는 actor bootstrap warm-start 용도로만 제한적으로 사용한다.
- 현재 주력 action은 `position_contact_frame_velocity_tilt_lateral_apex_residual` 15D residual이다.
- 현재 주력 preset은 `contact_frame_self_rally_v25_long_horizon_30_bounce`다.
- 학습 실행은 긴 CLI 대신 `configs/*.json` 설정파일과 `--set KEY=VALUE` override를 우선 사용한다.

## 디렉터리

- `assets/`: MuJoCo scene와 Franka asset
- `docs/analysis/`: 기존 실패 원인 분석 문서
- `src/pingpong_rl2/`: 최소 패키지
- `scripts/`: 학습, 평가, viewer, benchmark 진입점
- `artifacts/ppo_runs/`: 학습 결과물

## 주요 스크립트

현재 v25 재현 학습:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json
```

새 실험은 `run_version`만 바꿔서 실행한다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json \
  --run-version v26
```

드문 일회성 override는 개별 CLI 인자보다 `--set`을 사용한다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json \
  --run-version smoke_check \
  --smoke \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0
```

분석과 viewer:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name pmk_cf_self_rally \
  --run-version v25 \
  --episodes 100 \
  --seed 251 \
  --analysis-name pmk_cf_self_rally_v25_final_contact_diagnosis

PYTHONPATH=src conda run -n mujoco_env python scripts/run_viewer.py \
  --run-name pmk_cf_self_rally \
  --run-version v25 \
  --best-model \
  --episodes 5
```

보조 도구:

- `scripts/run_material_sanity.py`: MuJoCo 공/라켓 물성 sanity check
- `scripts/run_contact_feasibility_map.py`: scripted contact upper-bound/feasibility sweep
- `scripts/run_heuristic_keepup_diagnostic.py`: PPO 없이 heuristic baseline 확인
- `scripts/benchmark_vector_env.py`: vector env throughput 확인

학습 재개 규칙:

- 기본적으로 `<run-name>_model.zip`가 이미 있으면 같은 이름으로 이어서 학습한다.
- 새 이름을 주고 그 이름의 체크포인트가 아직 없으면 새 모델이 생성된다.
- 같은 이름으로 처음부터 다시 시작하려면 `--reset-model`을 사용한다.
- 다른 체크포인트를 이어받으려면 `--resume-from <zip>`을 사용한다.

## 현재 판단

- 현재 발표 후보는 `pmk_cf_self_rally_v25`다.
- 100 episode rebound analysis 기준 mean useful bounce `28.51`, max `51`, `30+ useful bounce rate=0.61`이다.
- `time_limit` episode는 평균 useful `37.37`로, 실패라기보다 episode horizon 끝까지 버틴 성공 케이스에 가깝다.
- 남은 개선 포인트는 초중반 `low_apex_contact`와 긴 episode 말기의 드문 `ball_out_of_bounds`다.
- 시연은 final model과 `pmk_cf_self_rally_v25_best_model.zip`을 viewer로 비교해 더 안정적인 쪽을 쓰는 것이 좋다.
