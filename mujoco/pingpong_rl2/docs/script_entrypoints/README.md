# Script Entrypoints

이 디렉터리는 `pingpong_rl2/scripts/`와 발표 자료 생성 스크립트를 실행 파일 단위로 설명한다. 현재 기준 핵심 모델은 `keep1_v39_17d_mid_curriculum_fixed`이고, 후속 확인 대상은 `keep1_v40_17d_v39_polish`다.

## 전체 흐름

1. `scripts/run_ppo_learning.py`
   - config/preset/CLI override를 합쳐 환경 kwargs를 만든다.
   - `PingPongKeepUpGymEnv`를 vector env로 감싼다.
   - 새 PPO를 만들거나 checkpoint를 resume한다.
   - 모델 zip, monitor CSV, training summary JSON을 저장한다.

2. `scripts/run_ppo_evaluation.py`
   - 저장 모델과 training summary에서 환경 설정을 복원한다.
   - headless episode 평가 결과를 JSON으로 남긴다.

3. `scripts/run_ppo_rebound_analysis.py`
   - contact event, outgoing velocity, apex, next intercept, failure mode를 CSV/JSON으로 분석한다.

4. `scripts/run_viewer.py`
   - MuJoCo viewer에서 policy 또는 heuristic 동작을 직접 확인한다.

5. `scripts/run_heuristic_keepup_diagnostic.py`, `scripts/run_contact_feasibility_map.py`
   - PPO 없이 hand-coded policy와 contact feasibility upper bound를 확인한다.

6. `scripts/run_bounce_sanity.py`, `scripts/run_material_sanity.py`, `scripts/benchmark_vector_env.py`, `scripts/expand_ppo_action_space.py`
   - scene 물리, vector env 처리량, action space 확장 transfer를 확인한다.

7. `docs/rl_presentation_pack/scripts/generate_visuals.py`
   - 발표용 그래프와 CSV를 재생성한다.

## 문서 목록

| 실행 파일 | 문서 | 용도 |
| --- | --- | --- |
| `scripts/run_ppo_learning.py` | `run_ppo_learning.md` | PPO 학습 |
| `scripts/run_ppo_evaluation.py` | `run_ppo_evaluation.md` | 저장 모델 평가 |
| `scripts/run_ppo_rebound_analysis.py` | `run_ppo_rebound_analysis.md` | contact/rebound 분석 |
| `scripts/run_heuristic_keepup_diagnostic.py` | `run_heuristic_keepup_diagnostic.md` | heuristic baseline 진단 |
| `scripts/run_contact_feasibility_map.py` | `run_contact_feasibility_map.md` | feasibility grid sweep |
| `scripts/run_viewer.py` | `run_viewer.md` | viewer 실행 |
| `scripts/expand_ppo_action_space.py` | `expand_ppo_action_space.md` | 15D to 17D transfer |
| `scripts/run_bounce_sanity.py` | `run_bounce_sanity.md` | 낙하/접촉 sanity |
| `scripts/run_material_sanity.py` | `run_material_sanity.md` | material/restitution sanity |
| `scripts/benchmark_vector_env.py` | `benchmark_vector_env.md` | vector env benchmark |
| `docs/rl_presentation_pack/scripts/generate_visuals.py` | `generate_visuals.md` | 발표 그래프 생성 |

## 현재 결론

`HeuristicKeepUpPolicy`는 PPO policy가 아니라 hand-coded teacher/baseline이다. v39/v40 학습 경로에서는 bootstrap이 꺼져 있으며, 최종 모델의 성능 판단은 저장된 PPO 모델과 rebound/evaluation 결과를 기준으로 한다.
