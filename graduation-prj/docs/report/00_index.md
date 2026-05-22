# docs/report 인덱스

`docs/report`는 보고서용으로 재사용 가능한 정리본을 모아두는 곳이다.

## 1. 기존 보고서 초안

- `01_mujoco_scene_setup_report.md`
- `02_ee_viewer_demo_report.md`
- `03_ball_bounce_and_ee_delta_env_report.md`
- `04_reward_logging_and_rollout_analysis_report.md`
- `05_preppo_distribution_and_ppo_logging_report.md`
- `06_competitive_robot_table_tennis_paper_report.md`
- `07_takeaways_from_competitive_table_tennis_for_bounce_rl.md`
- `08_car_rl_vs_pingpong_rl_structure_report.md`
- `09_pingpong_pending_tasks_audit.md`

## 2. 현재 기준 정리본

- `10_pingpong_rl_status_report.md`
  - 최근 코드 수정과 물리/보상/커리큘럼 변경을 반영한 현재 상태 요약
- `11_keepup_stability_review.md`
  - stable repeated bouncing 기준으로 task/action/reward/reset/bottleneck을 다시 검토한 보고서
- `12_keepup_v3_followup_and_control_assist_report.md`
  - `ppo_keepup_v3` 40k 재학습 분석, reward-only rebound shaping 실패, control-side tracking assist 비교 결과
- `13_v7_check_position_tilt_and_curriculum_report.md`
  - `ppo_keepup_v7` 확인, `position_tilt` action 실험, reward/policy/curriculum 역할과 현재 판단 정리
