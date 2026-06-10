# pingpong_rl 초기 모델부터 keep1_v39까지의 전체 흐름

작성 기준: 2026-06-07 로컬 저장소와 저장 artifact 기준.
여기서 `pingpong_rl 초기 모델`은 저장소 디렉토리상 `pingpong_rl`을 말한다. 최종 발표 기준 모델은 `pingpong_rl2/artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed`다. 사용자 지시 기준으로 `pingpong_rl3`와 `keep1_v40` 이후 artifact는 제외했다.

서술형으로 "왜 그 다음 단계로 넘어갔는지"까지 읽고 싶으면 [model_training_process_story.md](../../../model_training_process_story.md)를 본다.

## 한 줄 요약

처음에는 end-effector delta를 PPO가 직접 밀어 보는 단순 keep-up 문제였다. 이후 실패 원인을 보상 부족이 아니라 contact/control feasibility와 action ownership 문제로 다시 정의했고, contact-frame primitive 위에 PPO residual action을 얹는 구조로 바꾸었다. heuristic은 초기에 baseline, gate, actor bootstrap으로 탐색 공간을 잡아주는 역할을 했지만, 최종 v39 학습 자체에는 직접 들어가지 않았다.

## 큰 흐름

```text
pingpong_rl 초기 EE-delta PPO
  -> tracking assist와 heuristic baseline으로 keep-up 가능성 확인
  -> pingpong_rl2에서 contact trace, rebound analysis, heuristic gate 추가
  -> position_strike / tilt / follow-up strike 실험
  -> contact-frame primitive 도입
  -> heuristic bootstrap으로 actor warm-start 실험
  -> velocity, tilt, lateral, apex residual action 확장
  -> v25 long-horizon 30-bounce 목표
  -> v26 broad XYZ reset와 unlimited horizon
  -> v30/v31 reset 일반화 조정
  -> v32 15D -> 17D action-space transfer
  -> v34/v36 17D 안정 계열
  -> v39 mid curriculum fixed 최종 모델
```

## 1. pingpong_rl 초기 모델: EE delta 직접 제어

초기 `pingpong_rl`은 Panda racket의 end-effector target을 PPO action으로 조금씩 움직이는 방식이었다.

핵심 구조:

- action은 주로 EE position delta였다.
- 이후 `position_tilt` action이 추가되어 tilt도 일부 열렸다.
- `curriculum=keepup_v1`으로 쉬운 reset에서 넓은 reset으로 이동했다.
- `tracking_assist_weight`가 있으면 policy target과 analytic tracking target을 섞었다.

근거:

- [run_ppo_baseline.py](../../pingpong_rl/scripts/run_ppo_baseline.py:287)
- [ee_delta_env.py](../../pingpong_rl/src/pingpong_rl/envs/ee_delta_env.py:602)
- [curriculum.py](../../pingpong_rl/src/pingpong_rl/training/curriculum.py:18)
- [heuristic audit](heuristic_bootstrap_audit.md)

대표 저장 summary:

| run | timesteps | action | curriculum | assist | heuristic bootstrap |
| --- | ---: | --- | --- | ---: | --- |
| `ppo_keepup_v1` | 2,000,000 | position 계열 | `keepup_v1` | curriculum stage 내부 | 없음 |
| `ppo_keepup_v6_trackassist40k` | 40,000 | position 계열 | `keepup_v1` | `tracking_assist_weight=0.2` | 없음 |
| `ppo_keepup_v8_tilt40k` | 40,000 | `position_tilt` | `keepup_v1` | `tracking_assist_weight=0.2` | 없음 |
| `ppo_keepup_v17` | 2,000,000 | `position_tilt` | `keepup_v1` | `tracking_assist_weight=0.2` | 없음 |

해석:

- 이 단계는 “PPO가 라켓을 움직여 공을 다시 띄울 수 있는가”를 확인한 시기다.
- 하지만 contact 순간의 outgoing velocity, 다음 intercept, 낮은 apex 회복 같은 핵심을 PPO가 직접 소유하지 못했다.
- tracking assist는 유용했지만, policy action 뒤에 analytic target이 섞이므로 최종적으로 “정책이 혼자 다 했다”고 설명하기 어렵다.

## 2. pingpong_rl2 재설계: 보상보다 contact/control feasibility

`pingpong_rl2`에서는 task를 다시 쪼갰다.

중요한 변화:

- contact trace와 rebound analysis를 추가했다.
- next intercept, contact context, task phase를 observation으로 넣기 시작했다.
- `run_heuristic_keepup_diagnostic.py`, `run_contact_feasibility_map.py`로 PPO 전에 scripted feasibility를 확인했다.
- 단순 reward shaping보다 “라켓이 실제로 원하는 반동을 만들 수 있는가”를 먼저 봤다.

근거:

- [PRESENTATION_PREP.md](../../PRESENTATION_PREP.md:65)
- [contact feasibility report](report/15_contact_feasibility_map_report.md:167)
- [contact feasibility conclusion](report/15_contact_feasibility_map_report.md:193)

핵심 교훈:

- scripted feasibility가 반복적으로 `3+` useful bounce gate를 못 넘었다.
- 그래서 병목은 “PPO를 더 오래 돌리면 된다”가 아니라 “contact/control surface를 바꿔야 한다” 쪽으로 해석됐다.

## 3. heuristic의 첫 번째 역할: baseline과 gate

초기 heuristic은 좋은 최종 정책이라기보다 계측 도구였다.

역할:

1. 환경이 물리적으로 가능한지 확인
2. action mode가 제대로 작동하는지 smoke test
3. PPO 전에 scripted gate를 넘는 primitive인지 확인
4. 실패 원인이 reward인지, controller/action 구조인지 구분

근거:

- [run_heuristic_keepup_diagnostic.md](script_entrypoints/run_heuristic_keepup_diagnostic.md)
- [run_contact_feasibility_map.md](script_entrypoints/run_contact_feasibility_map.md)
- [heuristic audit](heuristic_bootstrap_audit.md)

중요한 점:

- heuristic은 장기 랠리를 해결한 만능 policy가 아니었다.
- 오히려 heuristic도 `2` 근처에서 막히는 것을 보고 action/control 구조를 더 바꿔야 한다는 결론이 나왔다.

## 4. contact-frame primitive: zero action이 더 이상 빈 action이 아님

큰 전환점은 contact-frame primitive다.

이 구조에서는 PPO가 관절이나 EE 목표를 전부 처음부터 만들지 않는다. 기본 contact-frame strike가 있고, PPO는 그 위에 residual을 얹는다.

근거:

- [contact-frame primitive report](report/21_contact_frame_primitive_report.md:130)
- [zero residual interpretation](report/21_contact_frame_primitive_report.md:154)

핵심 문장:

> deterministic zero residual is no longer an empty action. It is the centered scripted contact-frame strike, and PPO learns residuals around it.

발표용 해석:

- 최종 모델은 순수 black-box joint policy가 아니다.
- 사람이 설계한 contact-frame primitive가 기본 타격 좌표계와 목표를 제공한다.
- PPO는 위치, tilt, outgoing velocity, apex/timing, tracking residual을 조정한다.

## 5. pingpong_rl2 v1~v24: 실패 원인을 쪼개며 구조를 키운 순서

`pingpong_rl2`의 초중반 흐름은 “한 번에 PPO를 성공시키기”가 아니라, 실패 원인을 나눠 보고 그에 맞춰 action space와 primitive를 계속 바꾸는 과정이었다.

| 구간 | 무엇을 바꿨나 | 왜 바꿨나 | 발표 해석 |
| --- | --- | --- | --- |
| minimal keep-up / position tilt | `position_tilt`, rebound analysis, chatter fix, inward tilt 방향 점검 | 단순 EE delta로는 rebound 방향과 타격 자세를 안정적으로 만들기 어려웠다. | “PPO action을 조금 넓히는 것만으로는 부족했다.” |
| phase contract / easy-next-ball | phase, contact, next-intercept observation과 event metric을 정리 | 성공/실패를 한 번의 contact가 아니라 다음 공을 칠 수 있는 상태로 봐야 했다. | “무엇을 보상하고 무엇을 성공으로 볼지 다시 정의했다.” |
| heuristic gate / follow-up strike | heuristic diagnostic, follow-up strike, bootstrap warm-start | PPO 전에 scripted primitive가 최소 gate를 넘는지 확인해야 했다. | “heuristic은 최종 정답보다 진단기와 초기 교사에 가까웠다.” |
| contact trace / feasibility map | contact trace, outgoing velocity sanity, feasibility grid, oracle 비교 | geometry 자체가 불가능한지, controller/action abstraction이 문제인지 분리했다. | “병목은 물리가 아니라 control surface였다.” |
| contact-frame primitive | `position_contact_frame`, desired outgoing velocity, zero residual strike | random action이 아니라 의미 있는 기본 타격 위에서 residual을 학습하게 했다. | “zero action이 빈 action이 아니라 기본 타격이 되었다.” |
| v13~v24 self-rally | low-apex recovery, lateral stability, 8D -> 11D -> 13D -> 15D residual, outward timing guard | 낮은 apex 루프, lateral out, timing/velocity 한계를 하나씩 줄였다. | “v25의 30-bounce horizon으로 넘어가기 위한 기반을 만든 시기다.” |

근거:

- [report index](report/00_index.md:10)
- [contact trace sanity](report/14_contact_trace_sanity_report.md:1)
- [contact feasibility map](report/15_contact_feasibility_map_report.md:1)
- [contact-frame primitive report](report/21_contact_frame_primitive_report.md:1)
- [v13 low-apex recovery](report/34_v13_fast_episode_low_apex_recovery_fix.md:1)
- [v18 lateral residual](report/39_v17_action_scale_and_v18_lateral_residual.md:1)
- [v24 to v25 horizon](report/46_v23_v24_review_and_v25_30_bounce_horizon.md:1)

## 6. heuristic의 두 번째 역할: actor bootstrap

`pingpong_rl2` 일부 legacy 모델에서는 `HeuristicKeepUpPolicy`가 실제 학습 전 actor bootstrap으로 쓰였다.

작동 방식:

1. heuristic이 env를 플레이한다.
2. observation/action sample을 모은다.
3. PPO actor가 heuristic action을 MSE로 모방하도록 사전학습한다.
4. 그 다음 PPO reinforcement learning을 진행한다.

근거:

- [bootstrap.py](../src/pingpong_rl2/training/bootstrap.py:12)
- [bootstrap_actor_from_dataset](../src/pingpong_rl2/training/bootstrap.py:145)
- [run_ppo_learning.py](../scripts/run_ppo_learning.py:162)

실제로 bootstrap summary가 남은 모델:

| run | action mode | mean useful | max useful | bootstrap evidence |
| --- | --- | ---: | ---: | --- |
| `pmk_cf_self_rally_v13` | `position_contact_frame` | 0.34 | 2 | base 18 episodes, 2,711 samples |
| `pmk_cf_self_rally_v18` | velocity/tilt/lateral residual | 2.89 | 9 | base 12 episodes, 3,000 samples |
| `pmk_cf_self_rally_v21` | apex residual | 1.53 | 7 | base 14 episodes, 3,000 samples |
| `pmk_cf_self_rally_v24` | outward timing guard | 9.21 | 23 | base 12 episodes, 3,000 samples |
| `pmk_cf_self_rally_v28_racket_tracking_spin` | 17D tracking residual | artifact summary에 bootstrap 존재 |
| `pmk_cf_self_rally_v29_racket_tracking_staged_distribution` | 17D tracking residual | artifact summary에 bootstrap 존재 |

자세한 목록은 [heuristic_bootstrap_audit.md](heuristic_bootstrap_audit.md)에 정리했다.

## 7. heuristic이 없었다면 초반 학습은 어려웠을까?

정직한 답은 “그랬을 가능성이 높지만, 불가능했다고 단정할 수는 없다”다.

어려웠을 가능성이 높은 이유:

- 공을 맞히는 contact event 자체가 sparse하다.
- useful bounce는 단순 contact보다 더 어렵다. 방향, 높이, 다음 intercept까지 맞아야 한다.
- 초기 policy가 random이면 대부분 floor contact, ball out, low apex로 끝난다.
- contact 이후 좋은 상태인 `post_success` 구간을 random exploration만으로 충분히 모으기 어렵다.
- heuristic bootstrap은 이 rare state/action을 먼저 보여줘 actor를 이상한 초기 분포에서 꺼내는 역할을 했다.

근거:

- v13은 bootstrap을 썼지만 여전히 mean useful 0.34, max 2라서 task가 매우 어려웠다.
- v18, v24처럼 action ownership이 넓어진 뒤 bootstrap과 PPO가 결합되면서 max useful이 9, 23까지 올라갔다.
- 보고서에는 zero-timestep bootstrapped model이 짧은 PPO continuation보다 더 안정적인 경우도 기록되어 있다.
  - [contact-frame report](report/21_contact_frame_primitive_report.md:898)
  - [contact-frame report](report/21_contact_frame_primitive_report.md:1075)

하지만 단정하면 안 되는 이유:

- heuristic 자체가 장기 랠리를 해결하지 못했다.
- PPO continuation이 bootstrap skill을 망가뜨린 기록도 있다.
- v25 이후 성능 향상은 bootstrap 하나보다 horizon, checkpoint 기준, low-apex recovery, action expansion, reset curriculum이 함께 만든 결과다.
- v32 이후 최종 keep1 계열은 bootstrap을 꺼 둔 상태로 checkpoint transfer/fine-tuning을 했다.

발표용 답:

> 초반에는 heuristic이 없었다면 학습이 훨씬 어려웠을 가능성이 큽니다. 특히 공을 맞힌 뒤 다시 칠 수 있는 상태를 random exploration으로 충분히 모으기 어려웠기 때문입니다. 다만 heuristic이 최종 해답은 아니었고, 장기 랠리는 contact-frame primitive, residual action 확장, low-apex recovery, 긴 horizon 평가 기준이 같이 들어오면서 가능해졌습니다.

## 8. 거푸집 비유는 맞는가?

상당히 맞다. 다만 두 종류의 거푸집을 구분해야 한다.

### 임시 거푸집: heuristic bootstrap

heuristic bootstrap은 건물 공사에서 콘크리트가 굳기 전 형태를 잡아주는 거푸집에 가깝다.

- 초기에 actor가 말도 안 되는 action 분포로 시작하지 않게 한다.
- contact 이후 유효한 행동 예시를 제공한다.
- PPO가 어느 쪽으로 탐색해야 하는지 초기 방향을 준다.
- 나중에 v39 학습에서는 제거됐다.

즉 “처음에는 형태를 잡아주고, 나중에는 없어지는 보조 구조”라는 점에서 거푸집 비유가 좋다.

### 남아 있는 구조물: contact-frame primitive

반면 contact-frame primitive는 단순 거푸집이 아니다. v39에도 남아 있는 구조다.

- action mode: `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual`
- 기본 타격 좌표계, desired outgoing velocity, apex/timing target은 계속 존재한다.
- PPO는 이 구조 위에서 residual을 학습한다.

그래서 더 정확한 비유는 다음과 같다.

> heuristic bootstrap은 임시 거푸집이고, contact-frame primitive는 건물의 골조다. v39에서는 거푸집은 떼어냈지만, 골조 위에서 더 정교한 residual policy를 학습한 것이다.

## 9. v25부터 v39까지의 성능 계보

v25 이후는 “처음부터 새로 배우기”보다 안정된 policy를 보존하면서 과제를 넓히는 흐름이다.

| 모델 | 핵심 변화 | train eval mean useful | max | 30+ rate | bootstrap |
| --- | --- | ---: | ---: | ---: | --- |
| `v25` | v23 안정 policy 유지, horizon 1800, 30-bounce 기준 | 30.41 | 48 | 0.65 | null |
| `v26` | unlimited horizon, broad XYZ reset | 49.90 | 82 | 0.733 | null |
| `v30` | v26에서 reset XY 0.10으로 확장 | 59.19 | 88 | 0.787 | null |
| `v31` | 더 넓은 XY 시도, 성능 하락 | 35.47 | 95 | 0.47 | null |
| `v32 17D` | v30 15D policy를 17D로 확장 후 fine-tune | 49.67 | 96 | 0.66 | null |
| `v34 17D` | long horizon, stronger 17D 계열 | 101.67 | 175 | 0.79 | null |
| `v36 17D` | balanced XYZ | 106.86 | 180 | 0.87 | null |
| `v39 final` | v36 기반 mid curriculum fixed | 119.52 | 181 | 0.83 | null |

중요한 연결:

- [v25 report](report/46_v23_v24_review_and_v25_30_bounce_horizon.md:1)
- [v26 report](report/48_v26_unlimited_broad_xyz_reset.md:1)
- [v32 report](report/54_v32_17d_transfer_finetune_report.md:1)
- [v39 summary](../artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json:1)

## 10. v39 최종 결과

v39는 v36 checkpoint에서 이어 학습했다.

```text
training_mode = resume
starting_model_path = keep1_v36_17d_balanced_xyz012_model.zip
total_timesteps = 700,000
bootstrap_heuristic_episodes = 0
bootstrap_epochs = 0
bootstrap = null
```

train eval 100 episode:

| metric | value |
| --- | ---: |
| mean useful bounces | 119.52 |
| max useful bounces | 181 |
| 30+ useful rate | 0.83 |
| low_apex_contact failures | 1 |

long eval 20 episode, 7200-step:

| metric | value |
| --- | ---: |
| mean useful bounces | 130.95 |
| max useful bounces | 182 |
| 30+ useful rate | 0.90 |

근거:

- [v39 training summary](../artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json:1)
- [v39 long eval summary](../artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/analysis/keep1_v39_oldbase_long7200_eval20_summary.json:1)
- [PRESENTATION_PREP.md](../../PRESENTATION_PREP.md:259)

## 11. 발표용 스토리

이 프로젝트의 스토리는 “PPO를 오래 돌려서 해결했다”가 아니다.

발표에서는 이렇게 말하는 편이 정확하다.

> 처음에는 end-effector delta를 PPO가 직접 제어하는 단순 문제로 시작했습니다. 하지만 공을 한 번 맞히는 것과 반복적으로 살릴 수 있는 반동을 만드는 것은 달랐습니다. 그래서 contact trace와 heuristic diagnostic으로 원인을 분해했고, 단순 reward보다 contact/control feasibility가 병목이라는 결론을 냈습니다. 이후 contact-frame primitive를 만들고, PPO는 그 위에서 residual action을 학습하도록 바꾸었습니다. 초반에는 heuristic bootstrap이 거푸집처럼 actor의 초기 형태를 잡아줬지만, 최종 v39는 bootstrap 없이 v36 checkpoint에서 fine-tuning된 모델입니다. 최종적으로는 17D residual action과 mid curriculum을 통해 long eval에서 평균 130.95 useful bounces까지 도달했습니다.
