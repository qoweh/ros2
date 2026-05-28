# pingpong_rl2

`pingpong_rl2`는 기존 `pingpong_rl`을 덮어쓰지 않고, keep-up 문제를 최소 구조로 다시 검증하기 위한 분리 프로젝트다.

## 설계 원칙

- env-side assist는 실험으로만 넣고, 현재 winner만 preset으로 고정한다.
- heuristic keep-up policy를 학습 경로에 섞지 않는다.
- curriculum을 기본 꺼 둔다.
- 현재 주력 action은 `position_strike`다.
- reward는 strike-window alignment, useful contact, apex quality, failure penalty만 남긴다.

## 디렉터리

- `assets/`: MuJoCo scene와 Franka asset
- `docs/analysis/`: 기존 실패 원인 분석 문서
- `src/pingpong_rl2/`: 최소 패키지
- `scripts/`: 학습, 평가, viewer, benchmark 진입점
- `artifacts/ppo_runs/`: 학습 결과물

## 주요 스크립트

- `python pingpong_rl2/scripts/run_bounce_sanity.py`
- `python pingpong_rl2/scripts/run_ppo_learning.py --preset final_candidate --run-name <name> --run-version <version> --reset-model`
- `python pingpong_rl2/scripts/run_ppo_evaluation.py --model-path <zip>`
- `python pingpong_rl2/scripts/run_ppo_rebound_analysis.py --model-path <zip> --episodes 50 --analysis-name <name>`
- `python pingpong_rl2/scripts/run_viewer.py --mode zero_action`
- `python pingpong_rl2/scripts/benchmark_vector_env.py --n-envs 4`

학습 재개 규칙:

- 기본적으로 `<run-name>_model.zip`가 이미 있으면 같은 이름으로 이어서 학습한다.
- 새 이름을 주고 그 이름의 체크포인트가 아직 없으면 새 모델이 생성된다.
- 같은 이름으로 처음부터 다시 시작하려면 `--reset-model`을 사용한다.
- 다른 체크포인트를 이어받으려면 `--resume-from <zip>`을 사용한다.

## 현재 baseline 범위

현재 active candidate는 `position_strike` 기반 control 설정이다.

- `strike_tilt_ramp_pitch=-0.03`
- `strike_tilt_ramp_xy_tolerance=0.04`
- `post_contact_return_assist_weight=0.5`
- `post_contact_return_max_intercept_time=0.6`
- velocity-domain observation은 기본으로 넣지 않는다.
- reward-side inward-return shaping은 기본 세트에 넣지 않는다.
