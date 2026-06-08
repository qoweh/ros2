질문 세 개를 분리해서 보겠습니다. 먼저 지금까지의 run/config를 기준으로 confounded 결론을 다시 정리하고, 그다음 PPO 자체의 한계인지 과제 정의/실험 설계 문제인지 설명한 뒤, 마지막으로 SAC 전환 비용만 코드 기준으로 검토하겠습니다.

SAC 전환 비용은 코드에서 PPO가 얼마나 박혀 있는지 보면 바로 나옵니다. 학습 스크립트와 평가/분석 스크립트의 하드코딩 범위만 좁게 확인하겠습니다.

Searched for regex `\bPPO\b|run_ppo_|ppo_runs|make_sb3_async_vector_env|VecMonitor` (`**/pingpong_rl2/**`), 200 results

Read [](file:///Users/pilt/project-collection/ros2/mujoco/pingpong_rl2/scripts/run_ppo_learning.py#1-1), lines 1 to 420

Read [](file:///Users/pilt/project-collection/ros2/mujoco/pingpong_rl2/scripts/run_ppo_evaluation.py)

SAC 전환 비용을 과장하지 않으려면 벡터 환경과 run-path 유틸까지 봐야 합니다. off-policy 알고리즘이 지금 학습 루프에 그냥 들어가는지 그 경계만 더 확인하겠습니다.

Read [](file:///Users/pilt/project-collection/ros2/mujoco/pingpong_rl2/src/pingpong_rl2/training/vector_env.py)

Read [](file:///Users/pilt/project-collection/ros2/mujoco/pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py)

Read [](file:///Users/pilt/project-collection/ros2/mujoco/pingpong_rl2/pyproject.toml)

todo 쪽 상태는 이번 기준으로 정리 완료로 맞춰두겠습니다. 이후부터는 reward를 더 얹는 게 아니라, 어떤 결론을 버려야 하는지와 SAC 전환 비용을 분리해서 답하겠습니다.

Completed: *Assess confounded conclusions* (2/4)

**현재 정리**

이번에는 지침대로 방향을 바꿨습니다. 새 reward를 더 붙이지 않고, 먼저 어떤 결론이 믿을 수 있는지 정리했고, 그다음 analysis용 metric만 최소 추가했습니다. 실제로 바뀐 건 pingpong_sim.py, keepup_env.py, run_ppo_rebound_analysis.py, test_scene_load.py, test_keepup_env.py입니다. 학습 reward는 더 안 얹었고, unittest 46개는 통과했습니다.

지금 artifact 기준으로 믿을 수 있는 결론과 confounded 결론은 분리됩니다.

| run | 핵심 변경 | mean useful bounces | mean contact vx | useful mean vx | positive x ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| pmk_s_v1 | position_strike만 | 0.16 | +0.3353 | +0.1913 | 1.00 |
| pmk_sc_v1 | center-seeking assist 추가 | 0.18 | +0.3578 | +0.1939 | 1.00 |
| pmk_sc_v2 | center-seeking assist 변형 | 0.10 | +0.3522 | +0.1532 | 1.00 |
| pmk_tnp_v1 | timed negative pitch 추가 | 0.30 | +0.3441 | +0.1919 | 1.00 |
| pmk_tnpv_v1 | velocity observation 추가 | 0.38 | +0.3302 | +0.1346 | 1.00 |
| pmk_tnpvr_v1 | x reward 추가 | 0.38 | +0.3488 | +0.2201 | 1.00 |
| pmk_tnpv_v2 | position only 1M | 0.16 | +0.4276 | +0.3587 | 0.8843 |
| pmk_tnpvr_oneway_v1 | strike+ramp+vel obs+reward | 0.12 | +0.3147 | +0.2233 | 1.00 |

여기서 믿을 수 있는 건 이겁니다. rebound 방향이 현재 병목이라는 점, center-seeking assist 계열이 별로였다는 점, 그리고 clean chain 안에서는 position_strike → timed negative pitch → velocity-domain observation까지는 개선 경향이 있었다는 점입니다. 반대로 pmk_tnpvr_oneway_v1_training_summary.json을 pmk_tnpv_v2_training_summary.json와 직접 비교해서 “reward 때문”이라고 말하는 건 틀렸습니다. position과 position_strike, ramp, velocity observation, reward가 한꺼번에 섞여 있습니다. 또 pmk_sn_v1_training_summary.json은 artifact상 position resume 500k라서, 예전 설명처럼 fixed negative pitch 근거로 쓰기 어렵습니다.

새로 한 실행은 baseline 재분석 하나입니다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py --run-name pmk_tnpv --run-version v2 --episodes 50 --analysis-name pmk_tnpv_v2_apex_50ep
```

결과는 pmk_tnpv_v2_apex_50ep_summary.json에 있습니다. projected apex xy error는 이제 계산되지만, 현재 target인 controller anchor xy로는 아직 분별력이 약합니다. useful contact mean apex error가 0.1663으로 오히려 더 크고, out-of-bounds 대 non-out-of-bounds도 0.1229 대 0.1209로 거의 안 갈립니다. 그래서 이 metric을 지금 바로 reward로 올리는 건 아직 아닙니다.

**지금 왜 어려운가**

질문대로, 최근 분기는 덕지덕지 붙는 방향으로 가기 시작했던 게 맞습니다. 특히 x reward 분기는 control mode, tilt assist, observation, reward가 섞여서 원인 분리가 안 됐습니다. 그래서 이번에는 그걸 멈추고 “어떤 실험이 깨끗한가”부터 다시 잘랐습니다.

이 문제가 어려운 이유는 PPO라서만은 아닙니다. 더 큰 문제는 과제 자체가 두 단계로 얽혀 있기 때문입니다. 첫 번째는 공 밑으로 다시 들어가는 것, 두 번째는 contact 후 공을 다시 칠 수 있는 strike zone 쪽으로 보내는 것입니다. 지금 reward와 metric은 대체로 “위로 튀었는가” 쪽엔 맞아 있는데, “다음에 다시 칠 수 있는 공간으로 돌아오는가”를 아직 정확히 정의하지 못했습니다. 즉 부족한 건 알고리즘보다 먼저 목표 정의와 실험 분리입니다. PPO가 나빠서 못 배우는 상태라고 단정할 근거는 아직 없습니다. 지금 target이 어긋나 있으면 SAC로 바꿔도 잘못된 목표를 더 빨리 최적화할 가능성이 큽니다.

짧게 말하면 지금 부족한 건 세 가지입니다.

- 좋은 rebound를 정의하는 target이 아직 애매합니다.
- 바뀐 요소를 한 번에 너무 많이 섞은 run이 있었습니다.
- 일부 historical run 설명과 실제 artifact config가 안 맞아서 해석 신뢰도가 떨어졌습니다.

**SAC 검토**

SAC로 바꾸는 건 환경 자체를 갈아엎는 수준은 아닙니다. 현재 action space는 연속형 Box라서 알고리즘 호환성은 있습니다. 즉 keepup_env.py나 controller를 대규모로 고칠 필요는 없습니다. 다만 “코드를 거의 안 바꾸고 import만 SAC로 바꾸는 수준”도 아닙니다. 난이도는 중간 정도입니다.

실제로 PPO가 박혀 있는 부분은 꽤 분명합니다.

- run_ppo_learning.py: PPO import, PPO 생성/로드, PPO 전용 하이퍼파라미터 구조
- run_ppo_evaluation.py: PPO.load 하드코딩
- run_ppo_rebound_analysis.py: PPO.load 하드코딩
- run_viewer.py: PPO.load 하드코딩
- ppo_runs.py: run path와 summary 유틸이 PPO 이름에 묶여 있음
- vector_env.py: 이건 재사용 가능하지만, SAC에서는 n_steps 대신 buffer_size, learning_starts, train_freq, gradient_steps 같은 off-policy 하이퍼파라미터 체계로 바뀝니다

그래서 결론은 이렇습니다.

- 환경 변경량: 작음
- 학습/평가/뷰어/유틸 변경량: 중간
- 전체 체감 난이도: “많이 어렵진 않지만, 그냥 한 줄 교체는 아님”

가장 깔끔한 방식은 PPO를 덮어쓰는 게 아니라 SAC를 평행 분기로 추가하는 겁니다.

1. training summary에 algorithm 필드를 넣는다.
2. 학습은 별도 run_sac_learning.py로 추가하거나, 기존 스크립트에 algo 옵션을 넣는다.
3. 평가, rebound analysis, viewer는 summary의 algorithm을 보고 PPO 또는 SAC를 로드하게 만든다.
4. 첫 SAC 실험은 1M이 아니라 50k~100k, n_envs는 1~2부터 시작한다.

제 판단으로는, 지금 당장 SAC로 바꾸는 게 1순위는 아닙니다. 먼저 clean ablation을 다시 하는 게 맞습니다. position, position_strike, position_strike + timed negative pitch, position_strike + timed negative pitch + velocity observation 이 네 줄만 같은 timestep budget으로 다시 맞추면 PPO에서 무엇이 실제로 먹히는지부터 분리됩니다. 그다음에도 막히면 SAC 분기는 충분히 해볼 가치가 있습니다. 다만 그때도 “x velocity를 줄이는가”가 아니라 “다음 strike zone으로 돌아오는가” 기준으로 가야 합니다.

다음으로 자연스러운 선택지는 둘입니다.
1. PPO 유지 상태에서 clean ablation 4개만 50k~100k로 다시 맞춘다.
2. SAC는 코드 변경 계획만 먼저 짧게 잡고, parallel branch로 시험용 러너만 만든다.