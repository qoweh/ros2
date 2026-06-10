# Script Entrypoints

이 디렉터리는 `pingpong_rl2/scripts/`와 발표 자료 생성 스크립트를 실행 파일 단위로 설명한다. 최종 발표 기준 핵심 모델은 `keep1_v39_17d_mid_curriculum_fixed`다.

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

## 읽는 순서

코드를 처음 따라갈 때는 `run_ppo_learning.md`를 먼저 읽는다. 이 문서가 CLI 설정, preset/config, vector env, PPO, heuristic bootstrap, 실제 `env.step()` 아래의 Cartesian/Jacobian controller 흐름까지 가장 넓게 설명한다.

그 다음에는 `run_ppo_evaluation.md`와 `run_ppo_rebound_analysis.md`를 같이 읽는다. evaluation은 저장된 policy를 headless로 재생해 episode-level 성능을 요약하고, rebound analysis는 contact event마다 물리량과 action 사용량을 CSV로 남긴다.

viewer나 sanity 파일은 그 뒤에 보면 된다. `run_viewer.md`는 같은 policy를 눈으로 확인하는 실행 흐름이고, `run_bounce_sanity.md`, `run_material_sanity.md`, `benchmark_vector_env.md`는 학습 결과가 아니라 물리/인프라 검증 파일이다.

## 공통 실행 계층

PPO policy를 실제로 실행하는 파일들은 결국 같은 환경 step 구조를 탄다.

```text
terminal entrypoint
  -> PingPongKeepUpGymEnv
  -> PingPongKeepUpEnv.step(action)
  -> action_mode별 residual 해석
  -> RacketCartesianController.compute_joint_targets()
  -> PingPongSim.step_with_contact_trace()
  -> MuJoCo actuator ctrl[:7]
  -> mujoco.mj_step()
```

반대로 `expand_ppo_action_space.py`, `benchmark_vector_env.py`, `run_bounce_sanity.py`, `run_material_sanity.py`, `generate_visuals.py`는 목적이 다르다. 이들은 policy rollout을 학습하거나 평가하는 파일이 아니라, transfer checkpoint 생성, vector env 처리량 측정, 순수 물리 sanity check, 발표 그래프 생성처럼 보조 작업을 담당한다.

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

`HeuristicKeepUpPolicy`는 PPO policy가 아니라 hand-coded teacher/baseline이다. v39 학습 경로에서는 bootstrap이 꺼져 있으며, 최종 모델의 성능 판단은 저장된 PPO 모델과 rebound/evaluation 결과를 기준으로 한다.
