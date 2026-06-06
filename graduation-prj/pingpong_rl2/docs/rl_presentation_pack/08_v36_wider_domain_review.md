# v36 학습 완료 검토와 넓은 영역 개선안

작성일: 2026-06-05

## 한 줄 요약

`keep1_v36_17d_balanced_xyz012`는 training summary 기준으로는 v34/v35보다 좋아졌지만, 같은 seed의 7200-step long eval에서는 v34보다 낮다. 대신 `xy=0.14`, `velocity_xy=0.05`, `velocity_z=[-0.16,0.04]`를 동시에 준 broad stress 조건에서는 v34보다 좋아졌다. 즉 v36은 넓은 영역으로 갈 가능성은 보였지만, 기본 분포 장기 안정성을 일부 잃었다.

## 학습 완료 확인

원본 파일:

- model: `artifacts/ppo_runs/keep1_v36_17d_balanced_xyz012/keep1_v36_17d_balanced_xyz012_model.zip`
- training summary: `artifacts/ppo_runs/keep1_v36_17d_balanced_xyz012/keep1_v36_17d_balanced_xyz012_training_summary.json`
- long analysis: `artifacts/ppo_runs/keep1_v36_17d_balanced_xyz012/analysis/keep1_v36_17d_balanced_xyz012_long7200_eval20_summary.json`

학습 조건:

| 항목 | 값 |
| --- | ---: |
| resume source | `keep1_v34_17d_long_xyz012_model.zip` |
| completed timesteps | `800,000` |
| reset XY | `0.12m` |
| reset height | `[0.22, 0.52]m` |
| reset velocity XY | `0.035` |
| reset velocity Z | `[-0.12, 0.04]` |
| learning rate | `2e-6` |
| n epochs | `1` |
| clip range | `0.02` |

중요한 점:

- v36은 학습 분포 자체가 v34보다 넓어진 모델은 아니다.
- v34와 같은 `xy=0.12`, `velocity_xy=0.035`, `velocity_z=[-0.12,0.04]`에서 balanced fine-tune한 모델이다.
- 따라서 "더 넓은 영역을 커버한다"는 주장은 별도 stress eval로만 말해야 한다.

## 기본 long eval 비교

7200-step / 20 episodes / seed 231 기준:

| 모델 | mean contacts | mean useful | max contacts | max useful | 300/100 | 400/150 | 주요 실패 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v34 | `318.55` | `116.05` | `426` | `170` | `13/20` | `9/20` | ball-out `5`, body `3`, speed `1` |
| v35 | `278.05` | `102.30` | `414` | `169` | `11/20` | `4/20` | ball-out `6`, body `1`, speed `2` |
| v36 | `255.90` | `94.45` | `444` | `165` | `9/20` | `6/20` | ball-out `10`, body `1`, speed `1` |

해석:

- v36의 peak contact는 `444`로 가장 높다.
- 하지만 평균 contact/useful과 `300/100` 달성률은 v34보다 낮다.
- 결정적인 문제는 `ball_out_of_bounds`가 `5 -> 10`으로 늘어난 것이다.

training summary만 보면 v36이 좋아 보인다.

| 모델 | mean useful | max useful | 30+ rate |
| --- | ---: | ---: | ---: |
| v34 | `101.67` | `175` | `0.79` |
| v35 | `104.80` | `171` | `0.82` |
| v36 | `106.86` | `180` | `0.87` |

하지만 최종 선택은 training summary 하나로 하지 말고, long eval과 stress eval을 같이 봐야 한다.

## 넓은 위치/속도 stress eval

3600-step / 12 episodes / seed 331 기준:

| 조건 | 모델 | mean contacts | mean useful | max useful | 30+ | 70+ | time limit | ball out | body | speed | low-apex |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `xy=0.14` | v34 | `140.42` | `50.58` | `81` | `8/12` | `7/12` | `8` | `1` | `2` | `1` | `0` |
| `xy=0.14` | v36 | `96.25` | `36.17` | `90` | `5/12` | `5/12` | `5` | `5` | `2` | `0` | `0` |
| `vxy=0.05`, `vz=[-0.16,0.04]` | v34 | `125.92` | `44.33` | `82` | `7/12` | `5/12` | `6` | `5` | `1` | `0` | `0` |
| `vxy=0.05`, `vz=[-0.16,0.04]` | v36 | `121.75` | `45.00` | `88` | `7/12` | `5/12` | `6` | `3` | `3` | `0` | `0` |
| `xy=0.14`, `vxy=0.05`, `vz=[-0.16,0.04]` | v34 | `88.08` | `31.08` | `83` | `5/12` | `4/12` | `4` | `5` | `1` | `1` | `1` |
| `xy=0.14`, `vxy=0.05`, `vz=[-0.16,0.04]` | v36 | `129.50` | `47.92` | `89` | `8/12` | `6/12` | `7` | `4` | `1` | `0` | `0` |

해석:

- 위치만 넓힌 `xy=0.14`에서는 v36이 v34보다 나쁘다.
- 속도만 넓힌 조건에서는 거의 비슷하다.
- 위치와 속도를 동시에 넓힌 broad 조건에서는 v36이 v34보다 좋다.

따라서 v36은 "기본 분포에서는 장기 안정성이 떨어졌지만, 더 어려운 broad 조건에서는 일부 일반화가 좋아졌다"라고 보는 것이 맞다.

## action usage 관찰

v36은 v34/v35보다 일부 축을 다르게 쓴다.

| action | v34 mean abs | v35 mean abs | v36 mean abs | 해석 |
| --- | ---: | ---: | ---: | --- |
| `strike_plane_z` | `0.01624` | `0.01839` | `0.02302` | v36에서 거의 limit의 `92%`까지 사용 |
| `outgoing_x` | `0.19324` | `0.18729` | `0.20141` | 다음 궤적 조절 축이 더 강해짐 |
| `tracking_vx` | `0.00771` | `0.00098` | `0.01074` | v35에서 죽었던 tracking x가 살아남 |
| `tracking_vy` | `0.01922` | `0.01386` | `0.01598` | v34보다는 약하지만 유지 |
| `tilt_roll` | `0.00672` | `0.00647` | `0.00280` | v36에서 크게 줄어듦 |
| `centering_tilt_scale` | `0.00777` | `0.00311` | `0.01490` | v36에서 중심 보정이 커짐 |

해석:

- v36은 넓은 조건을 다루기 위한 tracking/height/timing 축을 더 적극적으로 쓴다.
- 하지만 `strike_plane_z`가 거의 포화되어 있어, z/velocity 확장에서는 여유가 부족할 수 있다.
- action bound를 바꾸면 `action_space.high`가 바뀌어 PPO zip resume이 위험하므로, 다음 실험에서는 bounds를 유지하는 것이 안전하다.

## 다음 개선 방향

v37 목표:

- v34의 기본 long-horizon 안정성 회복
- v36의 broad stress 장점 유지
- ball-out 짧은 실패 감소
- action bound는 유지

추천 방향은 `v36 -> v37` curriculum이다. v34에서 다시 시작하면 broad 조건에서 얻은 v36의 장점이 사라질 수 있고, v36에서 바로 hard broad로만 학습하면 기본 분포를 더 잊을 수 있다. 따라서 v36에서 시작하되, 매우 낮은 learning rate로 reset distribution을 천천히 넓힌다.

권장 학습 명령:

```bash
PYTHONPATH=src python -u scripts/run_ppo_learning.py \
  --config-file configs/keep1_v32_17d_transfer.json \
  --set run_version=v37_17d_wide_curriculum_guarded \
  --set resume_from=artifacts/ppo_runs/keep1_v36_17d_balanced_xyz012/keep1_v36_17d_balanced_xyz012_model.zip \
  --set total_timesteps=1200000 \
  --set reset_xy_range=0.14 \
  --set reset_xy_curriculum_enabled=true \
  --set reset_xy_curriculum_start=0.12 \
  --set reset_xy_curriculum_end=0.14 \
  --set reset_xy_curriculum_fraction=0.90 \
  --set reset_velocity_xy_range=0.05 \
  --set reset_velocity_xy_curriculum_start=0.035 \
  --set reset_velocity_xy_curriculum_end=0.05 \
  --set reset_velocity_z_range='[-0.16,0.04]' \
  --set reset_velocity_z_curriculum_start='[-0.12,0.04]' \
  --set reset_velocity_z_curriculum_end='[-0.16,0.04]' \
  --set reset_ball_height_bounds='[0.22,0.52]' \
  --set low_apex_contact_grace_count=6 \
  --set stable_cycle_reward_cap=30 \
  --set next_intercept_xy_error_penalty_weight=1.5 \
  --set useful_contact_return_target_xy_reward_weight=0.35 \
  --set post_contact_lateral_velocity_penalty_weight=1.05 \
  --set contact_racket_outward_velocity_penalty_weight=0.95 \
  --set contact_lateral_stability_reward_weight=0.60 \
  --set easy_next_ball_reward_weight=1.05 \
  --set controller_body_clearance_gain=0.85 \
  --set controller_body_clearance_margin=0.15 \
  --set controller_body_clearance_vertical_margin=0.33 \
  --set controller_body_clearance_max_step=0.020 \
  --set learning_rate=1.5e-6 \
  --set n_epochs=1 \
  --set clip_range=0.015 \
  --set eval_episodes=100 \
  --set evaluation_step_limit=7200 \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0
```

평가 순서:

```bash
PYTHONPATH=src python -u scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v37_17d_wide_curriculum_guarded/keep1_v37_17d_wide_curriculum_guarded_model.zip \
  --episodes 20 \
  --seed 231 \
  --episode-step-limit 7200 \
  --analysis-name keep1_v37_17d_wide_curriculum_guarded_base_long7200_eval20
```

```bash
PYTHONPATH=src python -u scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v37_17d_wide_curriculum_guarded/keep1_v37_17d_wide_curriculum_guarded_model.zip \
  --episodes 12 \
  --seed 331 \
  --episode-step-limit 3600 \
  --reset-ball-height-bounds 0.22 0.52 \
  --reset-xy-range 0.14 \
  --reset-velocity-xy-range 0.05 \
  --reset-velocity-z-range -0.16 0.04 \
  --analysis-name stress_v37_xy014_vel05_z016_eval12_3600
```

성공 기준:

- 기본 long eval: v36보다 `ball_out` 감소, `300/100`은 최소 `9/20` 이상, 가능하면 v34의 `13/20`에 접근
- broad stress: v36의 `30+ 8/12`, `70+ 6/12` 이상 유지
- body contact: `1~2/20` 또는 `1/12` 근처 유지
- ball speed/low-apex: 거의 0에 가깝게 유지

## 현재 모델 선택

발표/시연 기준:

- 장기 랠리 안정성만 강조하면 v34가 아직 가장 안전하다.
- "더 넓은 위치/속도 조건으로 확장 중"이라는 연구 흐름을 보여주려면 v36 stress 결과가 좋다.
- 최종 배포 모델을 고르려면 v37을 위 조건으로 학습한 뒤, 기본 long eval과 broad stress를 모두 통과하는지 봐야 한다.
