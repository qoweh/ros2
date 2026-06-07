# pingpong_rl2 실행 시작점 문서

이 디렉토리는 터미널에서 직접 실행하는 Python 파일을 기준으로 정리한다. 범위는 `pingpong_rl2`이며 `pingpong_rl3`는 제외한다. 최종 발표 기준 run은 `keep1_v39_17d_mid_curriculum_fixed`이다.

## 전체 실행 흐름

1. 학습: `scripts/run_ppo_learning.py`
   - CLI/config/preset을 읽고 환경 kwargs를 만든다.
   - `PingPongKeepUpGymEnv`를 병렬 vector env로 감싼다.
   - 새 PPO를 만들거나 기존 checkpoint를 resume한다.
   - 선택적으로 heuristic bootstrap을 한 뒤 PPO 학습을 실행한다.
   - 모델, monitor CSV, training summary JSON을 저장한다.

2. 평가: `scripts/run_ppo_evaluation.py`
   - 저장된 모델과 training summary에서 환경 설정을 복원한다.
   - headless episode를 반복 실행한다.
   - return, contact, useful bounce, stable cycle, failure reason을 JSON으로 출력한다.

3. 반동/contact 분석: `scripts/run_ppo_rebound_analysis.py`
   - 평가보다 더 자세히 contact event별 물리량을 CSV로 남긴다.
   - outgoing velocity, apex error, next intercept, action dimension별 사용량, failure mode를 분석한다.

4. heuristic/진단: `run_heuristic_keepup_diagnostic.py`, `run_contact_feasibility_map.py`, `run_viewer.py`
   - PPO 없이 hand-coded policy가 어느 정도 가능한지 확인하거나, viewer에서 직접 본다.
   - 학습 중 사용되는 경우는 `run_ppo_learning.py`의 bootstrap 조건이 켜져 있을 때뿐이다.

5. 물리 sanity/유틸리티: `run_bounce_sanity.py`, `run_material_sanity.py`, `benchmark_vector_env.py`, `expand_ppo_action_space.py`
   - MuJoCo XML/반발, vector env 처리량, action space 확장 transfer checkpoint를 확인한다.

6. 발표 자료 시각화: `docs/rl_presentation_pack/scripts/generate_visuals.py`
   - 저장된 summary/analysis CSV를 읽어 발표용 PNG와 CSV를 재생성한다.

## 파일별 문서

| 실행 파일 | 문서 | 핵심 용도 |
| --- | --- | --- |
| `scripts/run_ppo_learning.py` | [run_ppo_learning.md](run_ppo_learning.md) | PPO 학습 entrypoint |
| `scripts/run_ppo_evaluation.py` | [run_ppo_evaluation.md](run_ppo_evaluation.md) | 저장 모델 headless 평가 |
| `scripts/run_ppo_rebound_analysis.py` | [run_ppo_rebound_analysis.md](run_ppo_rebound_analysis.md) | contact/rebound 상세 분석 |
| `scripts/run_heuristic_keepup_diagnostic.py` | [run_heuristic_keepup_diagnostic.md](run_heuristic_keepup_diagnostic.md) | heuristic baseline 진단 |
| `scripts/run_contact_feasibility_map.py` | [run_contact_feasibility_map.md](run_contact_feasibility_map.md) | heuristic parameter grid sweep |
| `scripts/run_viewer.py` | [run_viewer.md](run_viewer.md) | MuJoCo viewer 데모 |
| `scripts/expand_ppo_action_space.py` | [expand_ppo_action_space.md](expand_ppo_action_space.md) | action dimension 확장 transfer |
| `scripts/run_bounce_sanity.py` | [run_bounce_sanity.md](run_bounce_sanity.md) | 기본 낙하/접촉 sanity |
| `scripts/run_material_sanity.py` | [run_material_sanity.md](run_material_sanity.md) | XML material/restitution sanity |
| `scripts/benchmark_vector_env.py` | [benchmark_vector_env.md](benchmark_vector_env.md) | vector env step 처리량 측정 |
| `docs/rl_presentation_pack/scripts/generate_visuals.py` | [generate_visuals.md](generate_visuals.md) | 발표용 그래프 생성 |

## heuristic_keepup.py 결론

자세한 내용은 [heuristic_keepup_usage.md](heuristic_keepup_usage.md)에 따로 정리했다.
프로젝트 전체 과거 모델 audit은 [../heuristic_bootstrap_audit.md](../heuristic_bootstrap_audit.md)에 있다.

- `HeuristicKeepUpPolicy`는 PPO가 아니라 손으로 만든 scripted controller다.
- 학습에서 직접 쓰이는 경로는 `run_ppo_learning.py`의 optional bootstrap뿐이다.
- v39 최종 run 자체에서는 사용되지 않았다. v39 summary 기준 `training_mode=resume`, `starting_model_path=keep1_v36_17d_balanced_xyz012_model.zip`, `bootstrap_heuristic_episodes=0`, `bootstrap_epochs=0`, `bootstrap=null`이다.
