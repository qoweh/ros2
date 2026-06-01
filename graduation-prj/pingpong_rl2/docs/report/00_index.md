# pingpong_rl2 docs/report 인덱스

`pingpong_rl2/docs/report`는 `pingpong_rl2` 전용 정리본과 실험 보고서를 모아두는 곳이다.

## 1. 현재 기준 정리본

- `01_pingpong_rl2_minimal_keepup_status_report.md`
  - `pingpong_rl2` 최소 keep-up 재설계, 학습 상태, 현재 병목, 검증 결과를 한 번에 정리한 상태 보고서
- `02_pingpong_rl2_position_tilt_and_rebound_report.md`
  - versioned run-name 실험 표면, `position_tilt` branch, rebound analysis 도구와 최신 결과를 정리한 보고서
- `03_pingpong_rl2_position_tilt_chatter_fix_report.md`
  - `position_tilt` chatter exploit 진단, anti-chatter reward/control 수정, staged tilt profile A/B 결과를 정리한 보고서
- `04_pingpong_rl2_inward_tilt_direction_report.md`
  - outward rebound 편향 진단, inward-only pitch clamp 및 initial tilt bias A/B, 그리고 왜 아직 tilt 사용이 살아나지 않았는지 정리한 보고서
- `05_project_completion_plan.md`
  - 현재 실패 원인을 기준으로 프로젝트를 어떤 단계와 성공 기준으로 끝낼지 정리한 완성 로드맵
- `06_learning_design_checklist.md`
  - 매 실험 전후로 확인할 지표, 실패 모드별 처방, observation/reward/training 설계 체크리스트
- `07_reward_policy_cleanup_plan.md`
  - baseline/preset/reward 정리, projected apex target 비교 도구, 다음 clean ablation 명령과 1M 진행 조건을 정리한 정비 계획
- `08_easy_next_ball_completion_plan.md`
  - easy-next-ball metric 추가, active candidate 정합성 확인, metric-only 분석 결과와 reward 승격 보류 이유를 정리한 계획
- `09_keepup_task_rethink_plan.md`
  - keep-up을 phase로 다시 나누고 observation/reward/controller contract를 재정의한 방향 재검토 문서
- `10_keepup_phase_contract_implementation_report.md`
  - phase/contact/next-intercept observation, small event reward, heuristic diagnostic baseline 구현 결과를 정리한 보고서
- `11_keepup_heuristic_variant_gate_report.md`
  - heuristic pitch/contact-direction variant 3개 비교, 100-episode gate 결과, PPO/bootstrapping 보류 판단을 정리한 보고서
- `12_followup_strike_bootstrap_report.md`
  - follow-up strike contract PPO의 50-episode 비교, heuristic bootstrap warm-start 구현, 그리고 second-strike 품질 관점의 현재 결론을 정리한 보고서
- `13_direct_trajectory_objective_report.md`
  - direct outgoing-trajectory objective 실험과 왜 그것만으로는 구조 gate를 통과하지 못했는지 정리한 보고서
- `14_contact_trace_sanity_report.md`
  - contact trace와 outgoing velocity source 해석, MuJoCo normal 부호, post-contact velocity selection sanity를 정리한 보고서
- `15_contact_feasibility_map_report.md`
  - scripted contact feasibility grid sweep과 현재 `position_strike` ceiling이 왜 `3+`를 못 넘는지 정리한 보고서
- `16_contact_upper_bound_report.md`
  - desired-outgoing contact oracle로 geometry ceiling 여부를 분리하고, geometry가 아니라 controller/action abstraction이 병목이라는 점을 증명한 보고서
- `17_contact_primitive_training_plan.md`
  - contact primitive branch의 구현 순서, gating criteria, 현재까지의 scaffold 상태를 정리한 계획 문서
- `18_tilt_primitive_scripted_feasibility_report.md`
  - `position_strike_tilt` tilt-only residual sweep 결과와 왜 tilt-only primitive가 불충분한지 정리한 보고서
- `19_non_tilt_contact_residual_report.md`
  - phase-aware non-tilt residual, explicit lift scaffold, strike x sweep까지 포함한 최신 scripted primitive 결과를 정리한 보고서
- `20_state_dependent_contact_point_and_timing_report.md`
  - state-dependent strike XY correction gain과 late strike z pulse 실험, 그리고 왜 다음 branch가 더 직접적인 contact-point / impact-time primitive여야 하는지 정리한 보고서
- `21_contact_frame_primitive_report.md`
  - `position_contact_frame` primitive, desired outgoing velocity, trajectory tilt, follow-through, bootstrap/strict success 변화를 길게 누적 정리한 보고서
- `22_sweet_spot_start_range_and_next_work.md`
  - sweet spot-only reset 범위 계산, `reset_xy_range=0.028` 근거, 그리고 값 sweep을 멈추기 위한 다음 작업 순서를 정리한 보고서
- `23_sweet_spot_completion_status_report.md`
  - 최신 모델/strict sweet spot 평가, pitch-roll/no-rise-chase/nullspace 실험 해석, 최종 목표를 위해 필요한 planned contact-time strike primitive 방향을 정리한 보고서
- `24_self_rally_planner_primitive_report.md`
  - DeepMind/table-tennis 연구에서 참고한 계층적 skill 구조를 keep-up에 맞게 적용하고, fixed planner / primitive base strike / residual-only RL / strict reward-success 구현을 정리한 보고서
