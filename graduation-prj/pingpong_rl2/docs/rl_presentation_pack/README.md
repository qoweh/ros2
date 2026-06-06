# RL 발표 정리 패키지

작성일: 2026-06-05

이 디렉터리는 `docs/report/`의 실험 기록과 `artifacts/ppo_runs/`의 summary/CSV/log를 발표용으로 다시 묶은 자료다. 목적은 “탁구공을 계속 올려치는 모델”을 단순 시연으로만 보이지 않게 하고, 강화학습 세팅을 어떻게 설계하고 검증했는지 설명할 수 있게 만드는 것이다.

추천해서 먼저 볼 순서는 README -> 05_slide_outline -> 04_visualization_catalog입니다. 발표 흐름을 잡고, 그 다음 세부 근거를 00~07에서 가져오면 좋습니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| [00_pre_v25_trial_history.md](00_pre_v25_trial_history.md) | git 기록으로 확인한 v25 이전 시행착오 |
| [01_experiment_story.md](01_experiment_story.md) | v25 이후 시행착오와 최종 스토리라인 |
| [02_training_setup_and_troubleshooting.md](02_training_setup_and_troubleshooting.md) | 학습이 잘 안 되던 이유, 로그/명령어/평가 세팅 정리 |
| [03_action_observation_validation.md](03_action_observation_validation.md) | 17D action, 55D observation, action 사용량/ablation 검토 |
| [04_visualization_catalog.md](04_visualization_catalog.md) | 생성한 시각화 자료별 발표 포인트 |
| [05_slide_outline.md](05_slide_outline.md) | 실제 발표 슬라이드 구성 후보 |
| [06_source_data.md](06_source_data.md) | 수치가 나온 원본 report/artifacts 색인 |
| [07_v35_training_review_and_next_plan.md](07_v35_training_review_and_next_plan.md) | v35 학습 완료 검토와 v36 개선 방향 |
| [08_v36_wider_domain_review.md](08_v36_wider_domain_review.md) | v36 학습 완료 검토와 넓은 위치/속도 영역 개선안 |
| [scripts/generate_visuals.py](scripts/generate_visuals.py) | artifacts에서 CSV/PNG를 재생성하는 스크립트 |

## 생성된 자료

| 파일 | 핵심 메시지 |
| --- | --- |
| [assets/01_version_timeline_metrics.png](assets/01_version_timeline_metrics.png) | horizon, reset, action dimension 변화가 성능에 어떤 영향을 줬는지 |
| [assets/02_failure_modes_by_version.png](assets/02_failure_modes_by_version.png) | 버전이 올라가며 실패 모드가 어떻게 바뀌었는지 |
| [assets/03_long_horizon_target_hits.png](assets/03_long_horizon_target_hits.png) | v34/v35 장기 랠리 목표 비교와 v35 trade-off |
| [assets/04_apex_height_distribution.png](assets/04_apex_height_distribution.png) | low-apex 기준을 낮추기보다 grace를 늘린 이유 |
| [assets/05_action_usage_17d.png](assets/05_action_usage_17d.png) | 17D action 중 어떤 축을 크게/작게 쓰는지 |
| [assets/06_action_ablation_mean_useful.png](assets/06_action_ablation_mean_useful.png) | 약하게 보이는 축도 제거하면 성능이 떨어진다는 검증 |
| [assets/07_observation_action_diagram.png](assets/07_observation_action_diagram.png) | 55D observation과 17D action의 관계 |
| [assets/08_monitor_training_curves.png](assets/08_monitor_training_curves.png) | 터미널 로그가 조용해도 monitor log로 학습 진행을 확인할 수 있음 |

## 재생성 명령

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
PYTHONPATH=src python docs/rl_presentation_pack/scripts/generate_visuals.py
```

현재 스크립트는 아래 자료를 읽는다.

- `docs/report/46~54`의 결론과 대응되는 artifacts
- `artifacts/ppo_runs/keep1_v26`, `keep1_v30`, `keep1_v31`, `keep1_v32_17d`, `keep1_v33_17d_perf`, `keep1_v34_17d_long_xyz012`, `keep1_v35_17d_strong_axis_stable`
- v25는 `_legacy_models/pmk_cf_self_rally_v25` training summary
