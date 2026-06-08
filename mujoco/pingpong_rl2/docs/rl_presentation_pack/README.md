# RL Presentation Pack

이 디렉터리는 `pingpong_rl2`의 실험 보고서, artifacts, 분석 CSV를 발표용으로 다시 묶은 자료다. 단순히 viewer 시연만 보여주는 대신, 15D에서 17D로 넘어간 이유와 v39 모델을 선택한 근거를 설명하는 데 초점을 둔다.

## 읽는 순서

1. `05_slide_outline.md`: 발표 흐름
2. `04_visualization_catalog.md`: 그림별 핵심 메시지
3. `01_experiment_story.md`: v25 이후 실험 전개
4. `03_action_observation_validation.md`: 55D observation, 17D action 검증
5. `06_source_data.md`: 수치 출처와 재생성 기준

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `00_pre_v25_trial_history.md` | v25 이전 시행착오 |
| `01_experiment_story.md` | v25 이후 v39까지의 실험 스토리 |
| `02_training_setup_and_troubleshooting.md` | 학습/로그/평가 세팅과 문제 해결 |
| `03_action_observation_validation.md` | 17D action과 observation 검증 |
| `04_visualization_catalog.md` | 생성된 PNG별 발표 포인트 |
| `05_slide_outline.md` | 슬라이드 구성 후보 |
| `06_source_data.md` | 원본 report/artifact 색인 |
| `07_v35_training_review_and_next_plan.md` | v35 검토와 다음 방향 |
| `08_v36_wider_domain_review.md` | v36 넓은 reset 영역 검토 |
| `scripts/generate_visuals.py` | 발표용 CSV/PNG 재생성 |

## 생성물

| 파일 | 핵심 메시지 |
| --- | --- |
| `assets/01_version_timeline_metrics.png` | version, horizon, reset, action dimension 변화 |
| `assets/02_failure_modes_by_version.png` | 실패 모드 변화 |
| `assets/03_long_horizon_target_hits.png` | 장기 랠리 목표 달성 비교 |
| `assets/04_apex_height_distribution.png` | low-apex grace 설계 근거 |
| `assets/05_action_usage_17d.png` | 17D action 사용량 |
| `assets/06_action_ablation_mean_useful.png` | action ablation 결과 |
| `assets/07_observation_action_diagram.png` | observation/action 관계 |
| `assets/08_monitor_training_curves.png` | monitor log 기반 학습 진행 확인 |

## 재생성

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python docs/rl_presentation_pack/scripts/generate_visuals.py
```

현재 자료는 `artifacts/ppo_runs/keep1_v26`부터 `keep1_v40_17d_v39_polish`까지의 결과와 `docs/report/46~54`의 결론을 함께 참고한다. 웹 시연용 모델은 sibling 저장소에서 `keep_v39_17d` 이름으로 제공된다.
