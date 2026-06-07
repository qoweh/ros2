# Heuristic, Bootstrap, Assist 사용 이력 감사

작성 기준: 2026-06-07 로컬 저장소 기준.  
범위: `pingpong_rl`, `pingpong_rl2`, `pingpong_rl3`, `car_rl` 코드와 저장된 모델 summary/artifact.

## 결론

질문에 대한 짧은 답은 이렇다.

- `pingpong_rl2`의 일부 예전 모델은 `HeuristicKeepUpPolicy`를 actor bootstrap/warm-start로 실제 사용했다.
- 최종 발표 기준 모델인 `keep1_v39_17d_mid_curriculum_fixed` 자체는 heuristic bootstrap을 사용하지 않았다.
- `pingpong_rl` 초기 모델은 `HeuristicKeepUpPolicy` bootstrap은 없지만, PPO 환경 내부에 `tracking_assist_weight`가 섞인 모델들이 있어 “정책 단독 행동”이라고 말하면 안 되는 구간이 있다.
- `pingpong_rl3`에는 현재 구현/저장 artifact 기준 heuristic/bootstrap/teacher distillation이 없다. 문서상 “나중에 distillation 가능”이라는 아이디어만 있다.
- `car_rl`에는 scripted demo controller가 있지만, pingpong 모델 학습과는 별개이고 PPO 학습에 bootstrap teacher로 쓰인 흔적은 없다.

## 판정 기준

이 문서에서는 다음처럼 구분한다.

| 분류 | 의미 | 사용으로 세는가 |
| --- | --- | --- |
| actual heuristic bootstrap | training summary의 `bootstrap` 객체가 `null`이 아니고 actor pretraining 기록이 있음 | 예 |
| attempted but no accepted sample | `bootstrap` 객체는 있으나 accepted sample/loss가 없음 | 호출은 됐지만 학습 효과는 없음 |
| latent bootstrap config | `bootstrap_heuristic_episodes > 0` 같은 설정은 있으나 `bootstrap = null` | 아니오 |
| diagnostic/scripted baseline | heuristic script를 별도 실행해 baseline/feasibility를 확인 | 학습 사용 아님 |
| env internal assist | policy action 뒤에 환경/controller가 analytic target을 섞음 | bootstrap은 아니지만 발표에서 밝혀야 함 |

`run_ppo_learning.py`의 실제 조건은 다음이다.

```python
starting_model_path is None
and args.bootstrap_heuristic_episodes > 0
and args.bootstrap_epochs > 0
```

근거:

- [run_ppo_learning.py](../scripts/run_ppo_learning.py:162)
- [bootstrap.py](../src/pingpong_rl2/training/bootstrap.py:12)
- [HeuristicKeepUpPolicy](../src/pingpong_rl2/controllers/heuristic_keepup.py:49)

## 조사 범위

저장된 summary 기준으로는 다음을 확인했다.

| 프로젝트 | 확인한 summary |
| --- | ---: |
| `pingpong_rl` | `*_training_summary.json` 21개 |
| `pingpong_rl2` | `*_training_summary.json` 45개 |
| `pingpong_rl3` | `training_config.json` 3개, analysis `summary.json` 2개 |

추가로 코드 검색에서 `heuristic`, `bootstrap`, `teacher`, `distill`, `scripted`, `tracking_assist` 키워드를 확인했다.

## pingpong_rl: heuristic baseline은 있었지만 PPO bootstrap은 아님

`pingpong_rl`에는 초기 keep-up heuristic controller가 있다.

- [keepup_heuristic.py](../../pingpong_rl/src/pingpong_rl/controllers/keepup_heuristic.py:99)
- [run_keepup_baseline.py](../../pingpong_rl/scripts/run_keepup_baseline.py:14)

하지만 이 controller는 `run_keepup_baseline.py`에서 별도 scripted baseline으로 실행된다. `pingpong_rl/scripts/run_ppo_baseline.py`는 `KeepUpHeuristicController`를 import하지 않는다.

따라서 `pingpong_rl`의 PPO 모델들은 `rl2`처럼 heuristic rollout dataset으로 actor를 MSE 사전학습한 모델은 아니다.

### 중요한 예외: tracking assist

`pingpong_rl`에는 PPO action 이후 환경 내부에서 keep-up target을 섞는 assist가 있었다.

근거:

- [ee_delta_env.py](../../pingpong_rl/src/pingpong_rl/envs/ee_delta_env.py:602): policy가 낸 controller target과 tracking assist target을 blend한다.
- [ee_delta_env.py](../../pingpong_rl/src/pingpong_rl/envs/ee_delta_env.py:1026): `tracking_assist_weight <= 0`이면 assist를 끈다.
- [ee_delta_env.py](../../pingpong_rl/src/pingpong_rl/envs/ee_delta_env.py:1116): tilt tracking assist도 따로 있다.
- [curriculum.py](../../pingpong_rl/src/pingpong_rl/training/curriculum.py:18): curriculum stage 이름은 `bootstrap`이지만, 이것은 heuristic actor bootstrap이 아니라 env 난이도/assist stage다.

저장 summary에서 확인된 값:

| run | curriculum | explicit tracking assist |
| --- | --- | ---: |
| `ppo_keepup_v1` ~ `ppo_keepup_v5`, `ppo_smoke` | `keepup_v1` | summary에는 직접 값 없음 |
| `ppo_keepup_v6_trackassist40k` ~ `ppo_keepup_v17` | `keepup_v1` | `tracking_assist_weight = 0.2` |
| `ppo_keepup_v8_tilt_smoke` | `keepup_v1` | `tracking_assist_weight = 0.2` |

주의할 점:

- `curriculum=keepup_v1` 자체도 stage 시작 시 `tracking_assist_weight`를 `0.45 -> 0.30 -> 0.15`로 바꾼다.
- 그래서 `pingpong_rl` keep-up 계열은 “heuristic bootstrap은 아니다”라고 답하되, “초기 환경에는 analytic tracking assist가 정책 행동에 섞였다”고 같이 설명해야 한다.

## pingpong_rl2: 실제 heuristic bootstrap 사용 모델

`pingpong_rl2`에서 actual heuristic bootstrap이 확인된 모델은 다음이다. 기준은 training summary의 `bootstrap` 객체가 존재하는지다.

| run | mode | action mode | timesteps | base accepted episodes | base samples | follow-up |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `pmk_cf_self_rally_v13` | new | `position_contact_frame` | 2,000,000 | 18 | 2,711 | yes |
| `pmk_cf_self_rally_v14` | new | `position_contact_frame` | 2,000,000 | 11 | 3,000 | yes |
| `pmk_cf_self_rally_v15` | new | `position_contact_frame` | 1,000,000 | 4 | 700 | yes |
| `pmk_cf_self_rally_v16` | new | `position_contact_frame_velocity_residual` | 1,000,000 | 4 | 700 | yes |
| `pmk_cf_self_rally_v17` | new | `position_contact_frame_velocity_tilt_residual` | 1,500,000 | 12 | 3,000 | yes |
| `pmk_cf_self_rally_v18` | new | `position_contact_frame_velocity_tilt_lateral_residual` | 1,000,000 | 12 | 3,000 | yes |
| `pmk_cf_self_rally_v21` | new | `position_contact_frame_velocity_tilt_lateral_apex_residual` | 1,000,000 | 14 | 3,000 | yes |
| `pmk_cf_self_rally_v24` | new | `position_contact_frame_velocity_tilt_lateral_apex_residual` | 1,000,000 | 12 | 3,000 | yes |
| `pmk_cf_self_rally_v28_racket_tracking_spin` | new | `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual` | 2,000,000 | 9 | 3,000 | yes |
| `pmk_cf_self_rally_v29_racket_tracking_staged_distribution` | new | `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual` | 2,000,000 | 7 | 3,000 | yes |

근거 summary:

- [v13 summary](../artifacts/ppo_runs/rl2_legacy_models/pmk_cf_self_rally_v13/pmk_cf_self_rally_v13_training_summary.json:1)
- [v28 tracking spin summary](../artifacts/ppo_runs/rl2_legacy_models/pmk_cf_self_rally_v28_racket_tracking_spin/pmk_cf_self_rally_v28_racket_tracking_spin_training_summary.json:1)
- [v29 staged summary](../artifacts/ppo_runs/rl2_legacy_models/pmk_cf_self_rally_v29_racket_tracking_staged_distribution/pmk_cf_self_rally_v29_racket_tracking_staged_distribution_training_summary.json:1)

추가로 `pmk_cf_self_rally_v28_robot_base_disk_smoke`는 `bootstrap` 객체가 있지만 `base accepted episodes=0`, `base samples=0`, `loss=None`이다. 즉 bootstrap 호출은 시도됐지만 actor pretraining 효과는 없었다.

## pingpong_rl2: 설정은 있었지만 실제 bootstrap으로 세지 않는 모델

아래 모델들은 config에 bootstrap 값이 남아 있지만 summary의 `bootstrap`은 `null`이다. 그래서 “heuristic bootstrap을 실제 사용했다”고 말하면 안 된다.

| run | mode | bootstrap config | starting model |
| --- | --- | --- | --- |
| `pmk_cf_self_rally_v26` / `keep1_v26` | resume | `80 / 20 / 10` | summary상 `None` |
| `pmk_cf_self_rally_v30_v26_wider_xy_stability` / `keep1_v30` | resume | `80 / 20 / 10` | `pmk_cf_self_rally_v26_model.zip` |
| `keep1_v31` | resume | `80 / 20 / 10` | `keep1_v30_model.zip` |
| `pmk_cf_self_rally_v27_fast` | resume | `80 / 20 / 10` | summary상 `None` |
| `pmk_cf_self_rally_v28_robot_base_disk` | resume | `80 / 20 / 10` | `pmk_cf_self_rally_v27_fast_best_model.zip` |

해석:

- 이 계열은 preset/config에 bootstrap 값이 남아 있었지만 summary에 `bootstrap=null`이다.
- `run_ppo_learning.py` 조건상 resume이면 bootstrap 블록을 타지 않는다.
- 발표에서는 “config에 값은 있었지만 실제 bootstrap 실행 기록은 없다”고 답하는 편이 정확하다.

## 최종 keep1 계열과 v39

최종 발표 기준으로 봐야 할 `keep1_v39_17d_mid_curriculum_fixed`는 heuristic bootstrap을 사용하지 않았다.

v39 summary 근거:

- [v39 training summary](../artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json:1)

핵심 값:

```text
training_mode = resume
starting_model_path = .../keep1_v36_17d_balanced_xyz012_model.zip
bootstrap_heuristic_episodes = 0
bootstrap_epochs = 0
bootstrap_followup_epochs = 0
bootstrap = null
```

저장된 keep1 lineage:

```text
keep1_v39
  <- keep1_v36
      <- keep1_v34
          <- keep1_v33 / keep1_v32
              <- keep1_v32_17d_init
                  <- keep1_v30
                      <- v26 계열
```

확인된 summary 기준으로 `keep1_v32`부터 `keep1_v40`까지는 `bootstrap_heuristic_episodes=0`, `bootstrap_epochs=0`, `bootstrap=null`이다.

주의:

- `keep1_v32_17d_init`은 [expand_ppo_action_space.py](../scripts/expand_ppo_action_space.py:1)로 만든 action-space transfer checkpoint다.
- 이것은 heuristic이 아니라 기존 PPO action head를 새 17D action space에 맞춰 복사/확장한 것이다.

## pingpong_rl3: 구현/저장 artifact 기준 heuristic 없음

`pingpong_rl3`의 학습 스크립트는 config를 읽고 PPO를 새로 만들거나 resume한다.

- [train.py](../../pingpong_rl3/scripts/train.py:65): resume 시 PPO load
- [train.py](../../pingpong_rl3/scripts/train.py:105): `model.learn(...)`

`pingpong_rl3/artifacts/ppo_runs` 아래 `training_config.json` 3개와 analysis `summary.json` 2개에서 `heuristic`, `bootstrap`, `teacher`, `distill`, `scripted` 키워드는 발견되지 않았다.

문서에는 teacher/distillation 아이디어가 한 번 나온다.

- [keep2_v2 report](../../pingpong_rl3/docs/report/04_keep2_v2_review_rl2_transfer_and_lift_recovery.md:54)

하지만 이 문장은 “나중에 rl2 policy를 teacher로 둔 distillation script를 만들 수 있다”는 가능성 설명이고, 현재 구현이나 artifact에는 없다.

## car_rl: project-wide scripted demo는 있지만 pingpong heuristic과 별개

`car_rl`에는 사람이 짠 `scripted_controller`가 있다.

- [car_rl/main.py](../../car_rl/main.py:23)

하지만 PPO 학습은 별도 `train.py`에서 진행된다.

- [car_rl/train.py](../../car_rl/train.py:59)

따라서 project 전체 관점에서는 scripted demo가 존재하지만, pingpong heuristic bootstrap과 같은 “학습 전 teacher warm-start” 근거는 아니다.

## 발표 때 특히 조심할 디테일

1. `bootstrap`이라는 단어가 두 종류다.
   - `pingpong_rl` curriculum stage 이름 `bootstrap`: 쉬운 초기 환경/assist stage.
   - `pingpong_rl2` heuristic bootstrap: heuristic rollout dataset으로 actor를 MSE 사전학습.

2. `config.bootstrap_heuristic_episodes > 0`만 보고 사용했다고 말하면 안 된다.
   - 실제 사용 증거는 summary의 `bootstrap` 객체다.
   - `bootstrap=null`이면 사용 기록으로 세지 않는다.

3. `bootstrap` 객체가 있어도 accepted sample이 0이면 학습 효과가 없다.
   - `pmk_cf_self_rally_v28_robot_base_disk_smoke`가 이 경우다.

4. `run_heuristic_keepup_diagnostic.py`, `run_contact_feasibility_map.py`, `run_viewer.py --mode heuristic`는 diagnostic/demo다.
   - PPO training으로 섞인 것은 아니다.

5. `post_contact_return_assist_weight`는 거의 모든 `pingpong_rl2` self-rally config에 남아 있다.
   - 이것은 `HeuristicKeepUpPolicy`가 아니라 환경/controller의 analytic return assist다.
   - “정책이 모든 것을 직접 제어했다”보다는 “contact-frame primitive와 residual policy를 같이 설계했다”가 정확하다.

6. v39는 heuristic bootstrap을 쓰지 않았지만, analytic contact-frame primitive 위에서 residual action을 학습한 모델이다.
   - 즉 “heuristic teacher로 학습했다”는 말은 틀리고, “구조화된 controller/primitive 위에서 PPO residual을 학습했다”가 맞다.

7. v28/v29 구간에는 tracking residual 17D action mode와 heuristic bootstrap padding/수정 이슈가 있었다.
   - [v28/v29 report](report/50_v28_tracking_spin_analysis_and_v29_staged_distribution.md:67)
   - 이 구간은 “heuristic bootstrap이 새 action 축을 어떻게 채웠는가”라는 질문이 들어올 수 있다.

## 답변 템플릿

발표 중 질문이 들어오면 이렇게 답하면 된다.

> 예전에는 사용한 적이 있습니다. 다만 최종 v39 학습에는 쓰지 않았습니다. `pingpong_rl2` legacy v13, v14, v15, v16, v17, v18, v21, v24, v28 tracking spin, v29 staged distribution 모델은 summary에 `bootstrap` 객체가 남아 있어서 heuristic rollout으로 actor를 사전학습한 근거가 있습니다. 반면 v39는 summary에 `training_mode=resume`, `bootstrap_heuristic_episodes=0`, `bootstrap_epochs=0`, `bootstrap=null`이라 heuristic bootstrap을 직접 사용하지 않았습니다. 또 초기 `pingpong_rl`에는 actor bootstrap은 없지만 `tracking_assist_weight`라는 환경 내부 assist가 있어서, 그 구간은 정책 단독 행동이 아니라 analytic assist가 섞인 PPO라고 설명해야 합니다.

## 재현용 확인 명령

```bash
rg -n "HeuristicKeepUpPolicy|collect_heuristic_bootstrap_dataset|bootstrap_actor_from_dataset" pingpong_rl2 -g '*.py'
rg -n "tracking_assist_weight|KeepUpHeuristicController" pingpong_rl -g '*.py'
rg -n "heuristic|bootstrap|teacher|distill|scripted" pingpong_rl pingpong_rl2 pingpong_rl3 car_rl -g '*.py' -g '*.json' -g '*.md'
find pingpong_rl pingpong_rl2 pingpong_rl3 car_rl -type f \( -name '*training_summary.json' -o -name 'training_config.json' -o -name 'summary.json' \)
```
