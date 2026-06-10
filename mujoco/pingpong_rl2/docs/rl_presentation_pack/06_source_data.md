# 사용한 원본 자료와 재현 경로

이 문서는 발표 자료의 수치가 어디에서 왔는지 빠르게 확인하기 위한 색인이다.

## 기존 보고서

| 주제 | 문서 |
| --- | --- |
| v25 이전 시행착오 git 기록 | [00_pre_v25_trial_history.md](00_pre_v25_trial_history.md) |
| 실험 설계 체크리스트 | [../report/06_learning_design_checklist.md](../report/06_learning_design_checklist.md) |
| contact feasibility와 scripted controller 한계 | [../report/15_contact_feasibility_map_report.md](../report/15_contact_feasibility_map_report.md) |
| 학습 런타임, PPO 로그, v2 실패 진단 | [../report/26_learning_runtime_parallel_and_v2_diagnosis.md](../report/26_learning_runtime_parallel_and_v2_diagnosis.md) |
| action ownership와 residual action 설계 | [../report/36_rl_action_ownership_and_8d_residual_plan.md](../report/36_rl_action_ownership_and_8d_residual_plan.md) |
| v25 horizon/30-bounce 기준 | [../report/46_v23_v24_review_and_v25_30_bounce_horizon.md](../report/46_v23_v24_review_and_v25_30_bounce_horizon.md) |
| preset/config 실행 구조 | [../report/47_run_ppo_learning_preset_config_reference.md](../report/47_run_ppo_learning_preset_config_reference.md) |
| v26 unlimited/broad XYZ reset | [../report/48_v26_unlimited_broad_xyz_reset.md](../report/48_v26_unlimited_broad_xyz_reset.md) |
| 17D tracking residual, spin, 2공 계획 | [../report/49_racket_center_tracking_spin_and_two_ball_plan.md](../report/49_racket_center_tracking_spin_and_two_ball_plan.md) |
| v28/v29 staged distribution | [../report/50_v28_tracking_spin_analysis_and_v29_staged_distribution.md](../report/50_v28_tracking_spin_analysis_and_v29_staged_distribution.md) |
| v30 안정 모델 정리 | [../report/52_v30_review_and_short_model_names.md](../report/52_v30_review_and_short_model_names.md) |
| v31 wider XY와 keep2 계획 | [../report/53_keep1_v31_wider_xy_and_keep2_model_plan.md](../report/53_keep1_v31_wider_xy_and_keep2_model_plan.md) |
| v32 17D transfer/fine-tune | [../report/54_v32_17d_transfer_finetune_report.md](../report/54_v32_17d_transfer_finetune_report.md) |
| v35 완료 검토와 v36 개선안 | [07_v35_training_review_and_next_plan.md](07_v35_training_review_and_next_plan.md) |

## artifacts summary

| 모델 | 원본 summary |
| --- | --- |
| v25 | `artifacts/ppo_runs/_legacy_models/pmk_cf_self_rally_v25/pmk_cf_self_rally_v25_training_summary.json` |
| v26 | `artifacts/ppo_runs/keep1_v26/keep1_v26_training_summary.json` |
| v30 | `artifacts/ppo_runs/keep1_v30/keep1_v30_training_summary.json` |
| v31 | `artifacts/ppo_runs/keep1_v31/keep1_v31_training_summary.json` |
| v32 17D | `artifacts/ppo_runs/keep1_v32_17d/keep1_v32_17d_training_summary.json` |
| v33 17D | `artifacts/ppo_runs/keep1_v33_17d_perf/keep1_v33_17d_perf_training_summary.json` |
| v34 17D | `artifacts/ppo_runs/keep1_v34_17d_long_xyz012/keep1_v34_17d_long_xyz012_training_summary.json` |
| v35 17D | `artifacts/ppo_runs/keep1_v35_17d_strong_axis_stable/keep1_v35_17d_strong_axis_stable_training_summary.json` |

주의:

- v33 training summary는 사용자가 중간에 0-step summary repair를 실행한 흔적 때문에 `completed_timesteps=0`으로 보일 수 있다.
- v33 성능 판단은 training summary보다 analysis CSV/summary와 monitor files를 우선했다.
- v35는 학습 완료 후 training summary와 7200-step long analysis가 생성되어 정량 비교에 포함했다.

## 분석 CSV/JSON

| 목적 | 파일 |
| --- | --- |
| v32 long eval20 | `artifacts/ppo_runs/keep1_v32_17d/analysis/keep1_v32_17d_long7200_eval20_summary.json` |
| v33 long eval20 | `artifacts/ppo_runs/keep1_v33_17d_perf/analysis/keep1_v33_17d_perf_long7200_eval20_summary.json` |
| v34 long eval20 | `artifacts/ppo_runs/keep1_v34_17d_long_xyz012/analysis/keep1_v34_17d_long_xyz012_long7200_eval20_summary.json` |
| v35 long eval20 | `artifacts/ppo_runs/keep1_v35_17d_strong_axis_stable/analysis/keep1_v35_17d_strong_axis_stable_long7200_eval20_summary.json` |
| v34 action usage | `artifacts/ppo_runs/keep1_v34_17d_long_xyz012/analysis/keep1_v34_17d_long_xyz012_long7200_eval20_contacts.csv` |
| v34 1800-step eval100 | `artifacts/ppo_runs/keep1_v34_17d_long_xyz012/analysis/keep1_v33_17d_long_xyz012_eval100_summary.json` |

## 생성된 요약 CSV

| 파일 | 내용 |
| --- | --- |
| [data/version_metrics.csv](data/version_metrics.csv) | v25/v26/v30/v31/v32/v34/v35 training summary 지표 |
| [data/long_horizon_metrics.csv](data/long_horizon_metrics.csv) | v32/v33/v34/v35 7200-step long eval 비교 |
| [data/action_usage_v34.csv](data/action_usage_v34.csv) | v34 17D action mean/std/사용률 |
| [data/action_ablation_v34.csv](data/action_ablation_v34.csv) | action mask ablation 결과 |

## 재생성

```bash
cd mujoco/pingpong_rl2
PYTHONPATH=src python docs/rl_presentation_pack/scripts/generate_visuals.py
```

스크립트가 생성하는 산출물:

- `assets/*.png`
- `data/*.csv`

시각화나 표가 최신 모델과 달라졌다면 이 스크립트를 다시 실행하고, `01_experiment_story.md`와 `03_action_observation_validation.md`의 본문 수치를 함께 갱신한다.
