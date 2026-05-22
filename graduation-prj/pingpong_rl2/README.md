# pingpong_rl2

`pingpong_rl2`는 기존 `pingpong_rl`을 덮어쓰지 않고, keep-up 문제를 최소 구조로 다시 검증하기 위한 분리 프로젝트다.

## 설계 원칙

- env 내부 tracking assist를 기본 사용하지 않는다.
- heuristic keep-up policy를 학습 경로에 섞지 않는다.
- curriculum을 기본 꺼 둔다.
- 기본 action은 `position-only EE delta`다.
- reward는 strike-window alignment, useful contact, apex quality, failure penalty만 남긴다.

## 디렉터리

- `assets/`: MuJoCo scene와 Franka asset
- `docs/analysis/`: 기존 실패 원인 분석 문서
- `src/pingpong_rl2/`: 최소 패키지
- `scripts/`: 학습, 평가, viewer, benchmark 진입점
- `artifacts/ppo_runs/`: 학습 결과물

## 주요 스크립트

- `python pingpong_rl2/scripts/run_bounce_sanity.py`
- `python pingpong_rl2/scripts/run_ppo_learning.py --smoke`
- `python pingpong_rl2/scripts/run_ppo_evaluation.py --model-path <zip>`
- `python pingpong_rl2/scripts/run_viewer.py --mode zero_action`
- `python pingpong_rl2/scripts/benchmark_vector_env.py --n-envs 4`

## 현재 baseline 범위

현재 baseline은 “assist 없이 racket-first contact와 반복 bounce가 다시 생기는가”만 본다. tilt, curriculum, rebound-direction shaping, single-bounce-out penalty는 기본 세트에 넣지 않았다.