# pingpong_rl2

`pingpong_rl2`는 기존 `pingpong_rl`을 덮어쓰지 않고, keep-up 문제를 최소 구조로 다시 검증하기 위한 분리 프로젝트다.

## 설계 원칙

- env-side assist는 실험으로만 넣고, 현재 winner만 preset으로 고정한다.
- heuristic keep-up policy는 기본 학습 경로에 직접 섞지 않되, 구조 gate가 통과한 뒤에는 actor bootstrap warm-start 용도로만 제한적으로 사용한다.
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
- `python pingpong_rl2/scripts/run_heuristic_keepup_diagnostic.py --episodes 20 --analysis-name <name>`
- `python pingpong_rl2/scripts/run_ppo_learning.py --preset final_candidate --run-name <name> --run-version <version> --reset-model`
- `python pingpong_rl2/scripts/run_ppo_learning.py --preset phase_contract_candidate --run-name <name> --run-version <version> --reset-model`
- `python pingpong_rl2/scripts/run_ppo_learning.py --preset followup_strike_candidate --run-name <name> --run-version <version> --reset-model`
- `python pingpong_rl2/scripts/run_ppo_learning.py --preset followup_strike_candidate --run-name <name> --run-version <version> --reset-model --bootstrap-heuristic-episodes 40 --bootstrap-epochs 5 --bootstrap-max-samples 4000`
- `python pingpong_rl2/scripts/run_ppo_evaluation.py --model-path <zip>`
- `python pingpong_rl2/scripts/run_ppo_rebound_analysis.py --model-path <zip> --episodes 50 --analysis-name <name>`
- `python pingpong_rl2/scripts/run_viewer.py --run-name clean_tnp_return_assist --run-version v1 --best-model --episodes 5`
- `python pingpong_rl2/scripts/run_viewer.py --mode heuristic --episodes 3`
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

## keep-up rethink follow-up

- `phase_contract_candidate` preset은 기존 `final_candidate` control stack 위에 `phase/contact/next-intercept` observation과 `next_intercept_reachable_bonus_weight=0.2`를 얹는 새 실험 preset이다.
- `followup_strike_candidate` preset은 위 contract에 더해 첫 useful bounce 이후 inward follow-up tilt contract를 유지해서 second-strike geometry를 직접 바꾸는 preset이다.
- `run_heuristic_keepup_diagnostic.py`는 PPO 없이 scripted diagnostic baseline을 돌려서 현재 환경/제어가 반복 keep-up을 허용하는지 먼저 확인한다.
- `run_viewer.py --mode heuristic`는 같은 baseline을 MuJoCo viewer에서 바로 재생한다.
- `run_ppo_learning.py`의 `--bootstrap-*` 옵션은 heuristic rollout 중 useful bounce가 나온 episode만 모아 actor를 supervised warm-start한 뒤 PPO를 시작한다. 현재 결과상 이 경로는 first useful bounce 안정화에는 유효하지만, second-strike quality는 follow-up checkpoint selection과 함께 봐야 한다.
- `run_ppo_learning.py`는 추가로 `--bootstrap-sample-mode`와 `--bootstrap-followup-*` 실험 옵션을 지원한다. 이들은 post-success / multi-bounce heuristic sample만 따로 bootstrap하는 연구용 경로이며, 현재 50-episode 기준으로는 기본 bootstrap보다 우위가 확인되지 않아 기본값으로 승격하지 않는다.

## 현재 판단

- centered upward useful second strike를 여는 핵심 구조 변경은 `followup_strike_target_tilt=(-0.03, 0.0)` 기반 follow-up strike contract다.
- heuristic bootstrap은 PPO가 이 구조를 더 빨리 배우게 해 준다.
- 현재 가장 목표지향적인 training schedule은 `followup_strike_bootstrap_v1_best_model.zip` 같은 plain-bootstrap best checkpoint에서 시작해, 같은 `followup_strike_candidate` contract 아래 PPO를 이어학습하는 staged resume 경로다.
- 이 staged 방향의 현재 기준 run은 `followup_bootstrap_resume_contract_v1_best`이며, 50-episode 기준 `two+ rate`를 유지하면서 contract-only run보다 `mean useful bounces`, `one+ rate`, useful-contact reachable rate, easy-next-ball score가 모두 좋아졌다.
