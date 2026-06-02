# v14 lateral out-of-bounds and v15 stability gate

작성일: 2026-06-02

## 목표

최종 목표는 탁구공을 계속 위로 치되, 단순히 높게 띄우는 것이 아니라 일정한 apex 높이와 다시 받을 수 있는 XY 방향을 유지하는 self-rally다. `pmk_cf_self_rally_v14`는 v13의 빠른 `low_apex_contact` 종료를 일부 완화했지만, 대신 `ball_out_of_bounds`와 next-intercept 품질 저하가 커졌다. 이번 작업은 이 trade-off를 막기 위해 높이 보상과 다음 공 안정성을 더 강하게 묶었다.

## v14 진단

`artifacts/ppo_runs/pmk_cf_self_rally_v14/analysis/pmk_cf_self_rally_v14_contact_diagnosis_summary.json` 기준:

- `mean_useful_bounces = 0.85`, `max_useful_bounces = 6`
- failure counts: `ball_out_of_bounds=61`, `low_apex_contact=19`, `floor_contact=9`, `time_limit=8`, `ball_speed_limit=2`, `robot_body_contact=1`
- total contacts `1445`
- mean projected apex `0.205m`, median `0.174m`
- upward contacts below target apex rate `0.768`
- next-intercept reachable rate `0.488`
- mean easy-next-ball score `0.487`
- mean ball lateral speed `0.143m/s`
- mean next-intercept XY error `0.071m`

해석:

- v14는 v13보다 useful count는 나아졌지만, 다음 공을 받을 수 있는 위치로 남기는 능력이 악화됐다.
- useful contact는 평균 target pitch가 약 `-0.026`, ball lateral speed가 `0.063`, easy-next-ball score가 `0.921`이었다.
- non-useful contact는 target pitch가 약 `-0.053`, target velocity z가 더 크고, ball lateral speed가 `0.148`, next XY error가 `0.074`였다.
- 따라서 "더 강한 lift/tilt"가 아니라 "높이 보상은 다음 intercept가 쉬울 때만 유효하게 만들고, lateral stability를 직접 보상"하는 쪽이 맞다.

## residual / tilt / lift / velocity 학습 범위

현재 active preset은 `position_contact_frame`이다.

- RL action은 5차원이다.
- 앞 3개는 contact-frame target position residual이다.
- 뒤 2개는 target tilt residual이다.
- `lift`와 `velocity target`은 대부분 hand-coded primitive/feedforward에서 만든다.
- 정책은 lift/velocity를 직접 자유롭게 출력하지 않고, primitive가 만든 접촉점/기울기/목표 속도를 작은 residual로 보정한다.

즉 현재 한계는 분명하다:

- 정책이 "위로 치는 힘"을 직접 크게 학습하는 구조가 아니다.
- 잘 설계된 primitive 위에서 residual이 안정성을 보정해야 한다.
- residual/tilt headroom이 너무 크면 v14처럼 높이 회복 전에 lateral drift를 만들 수 있다.
- 따라서 이번 v15는 residual 학습 공간을 없애지 않되, tilt headroom과 reward를 next-ball stability 중심으로 재정렬했다.

## 외부 지식 검토

- DeepMind robot table tennis 연구는 low-level skill/controller와 high-level selection을 나눈 계층형 구조, task distribution/curriculum, sim-to-real gap 처리를 강조한다. 현재 `contact_frame planner + primitive + residual PPO` 방향은 이와 맞다.  
  Source: https://deepmind.google/research/publications/achieving-human-level-competitive-robot-table-tennis/
- 로봇 탁구의 model-based feedforward + RL 접근도 명시적인 물리/제어 feedforward와 학습 policy를 결합한다. 이번 수정도 pure PPO가 아니라 primitive 안정화 위에 residual을 제한하는 방향이다.  
  Source: https://link.springer.com/article/10.1007/s10514-023-10140-6
- MuJoCo `solref/solimp`는 실제 반발계수를 직접 넣는 값이 아니라 contact solver parameter다. 물성은 XML 숫자와 drop sanity를 함께 봐야 한다.  
  Source: https://mujoco.readthedocs.io/en/stable/modeling.html#restitution

## 코드 변경

### 1. apex progress gate

파일:

- `src/pingpong_rl2/envs/keepup_env.py`
- `scripts/run_ppo_learning.py`
- `scripts/run_ppo_rebound_analysis.py`

추가 env 옵션:

- `gate_contact_apex_progress_by_easy_next_ball`
- `contact_apex_progress_min_easy_next_ball_score`

동작:

- upward contact의 `contact_apex_progress_term`과 `contact_apex_recovery_progress_term`을 `easy_next_ball_score`로 scale한다.
- easy score가 설정된 floor보다 낮으면 height progress 보상을 0으로 만든다.
- 의도는 "높이는 좋아졌지만 다음 공이 못 받을 위치로 가는 접촉"을 더 이상 높이 보상으로 강화하지 않는 것이다.

### 2. lateral stability reward

추가 env 옵션:

- `contact_lateral_stability_reward_weight`
- `contact_lateral_stability_speed_tolerance`
- `contact_lateral_stability_xy_tolerance`

새 reward term:

- `contact_lateral_stability_term`

동작:

- outgoing ball lateral speed가 낮고 projected apex XY가 return target 근처일 때만 보상한다.
- v14의 `ball_out_of_bounds` 병목을 직접 겨냥한다.

### 3. residual/action logging

step info와 rebound analysis CSV에 아래를 추가했다.

- `applied_action`
- `applied_action_norm`
- `applied_action_normalized_norm`
- `applied_position_action_norm`
- `applied_tilt_action_norm`
- `applied_action_0_radial`
- `applied_action_1_tangent`
- `applied_action_2_z`
- `applied_action_3_tilt_pitch`
- `applied_action_4_tilt_roll`
- `contact_apex_progress_easy_next_ball_gate`
- `contact_lateral_stability_term`

이제 다음 분석에서 정책이 residual/tilt를 실제로 크게 쓰는지, 아니면 primitive만 따라가는지 분리할 수 있다.

### 4. v15 preset 조정

`contact_frame_self_rally_candidate` 주요 변경:

- `tilt_action_limit`: `0.006 -> 0.004`
- `target_tilt_limit`: `(0.16, 0.16) -> (0.12, 0.12)`
- `controller_max_orientation_step`: `0.24 -> 0.18`
- `contact_frame_trajectory_tilt_gain`: `1.0 -> 0.70`
- `contact_frame_trajectory_tilt_limit`: `(0.08, 0.08) -> (0.05, 0.05)`
- `contact_frame_centering_tilt_limit`: `(0.06, 0.06) -> (0.035, 0.035)`
- `contact_frame_centering_tilt_radius`: `0.10 -> 0.12`
- `next_intercept_xy_error_penalty_weight`: `1.00 -> 1.35`
- `post_contact_lateral_velocity_penalty_weight`: `0.45 -> 0.75`
- `contact_racket_lateral_velocity_penalty_weight`: `0.30 -> 0.45`
- `contact_apex_progress_reward_weight`: `0.90 -> 0.75`
- `contact_apex_recovery_progress_reward_weight`: `0.70 -> 0.45`
- `contact_lateral_stability_reward_weight = 0.45`
- `contact_lateral_stability_speed_tolerance = 0.25`
- `contact_lateral_stability_xy_tolerance = 0.08`
- `stable_contact_reward_weight`: `1.40 -> 1.60`
- `stable_cycle_reward_weight`: `0.90 -> 1.10`
- `trajectory_error_penalty_weight`: `0.55 -> 0.75`

### 5. checkpoint selection

`evaluation_sort_key()`에서 `ball_out_of_bounds` 감점을 useful-count tie-break보다 앞쪽으로 옮겼다.

이유:

- v14는 중간/초기 checkpoint가 반드시 최종 PPO보다 나쁘지 않았다.
- PPO가 더 오래 돌면서 lateral instability를 키울 수 있으므로, best model 선택 기준이 out-of-bounds를 더 일찍 싫어해야 한다.

## XML / 물성 검토

확인:

- `assets/scene.xml`
- `assets/franka/panda.xml`
- `scripts/run_material_sanity.py`

결과:

- gravity `-9.81`
- timestep `0.002`, control dt `0.02`
- ball radius `0.02m`
- ball mass `0.0027kg`
- racket body mass `0.18kg`
- default drop sanity at `drop_height=0.30`: effective normal restitution approx `0.968`

주의:

- `drop_height=0.40`에서는 같은 sanity script가 낮은 effective restitution을 보였다. 높은 impact speed/접촉 위치/solver 조건에 민감할 수 있다.
- 이번 v15에서는 물성을 바꾸지 않는다. 지금 병목은 XML 공 규격보다 policy residual/tilt가 lateral drift를 만드는 문제로 보는 것이 더 타당하다.
- 물성 A/B는 비교축이 깨지므로 v15 학습 결과를 본 뒤 별도로 다룬다.

## 검증

문법:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  scripts/run_ppo_learning.py \
  scripts/run_ppo_rebound_analysis.py
```

결과: 통과.

유닛 테스트:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest tests.test_keepup_env
```

결과: `96 tests` 통과.

확장 테스트:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest \
  tests.test_ppo_runs \
  tests.test_keepup_contract_features \
  tests.test_vector_env \
  tests.test_scene_load
```

결과: `17 tests` 통과.

smoke 학습:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally_smoke \
  --run-version v15_pretrain_check \
  --output-dir artifacts/tmp/pmk_cf_self_rally_v15_pretrain_check \
  --reset-model \
  --total-timesteps 1 \
  --smoke
```

결과:

- bootstrap, PPO smoke, checkpoint save, best-model save 모두 동작
- resolved preset에서 v15 gate/stability 옵션이 env config에 기록됨

분석 smoke:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/tmp/pmk_cf_self_rally_v15_pretrain_check/pmk_cf_self_rally_smoke_v15_pretrain_check_best_model.zip \
  --episodes 3 \
  --analysis-name pmk_cf_self_rally_v15_pretrain_check_analysis \
  --output-dir artifacts/tmp/pmk_cf_self_rally_v15_pretrain_check/analysis
```

결과:

- 새 contact summary fields 생성 확인
- `contact_apex_progress_easy_next_ball_gate`, `contact_lateral_stability_term`, action norm fields 기록 확인

## v15 학습 명령

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v15 \
  --reset-model \
  --total-timesteps 2000000
```

학습 후 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v15/pmk_cf_self_rally_v15_best_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v15_contact_diagnosis
```

## 다음 판정 기준

v15가 좋아졌다고 보려면 아래가 동시에 좋아져야 한다.

- `ball_out_of_bounds`가 v14의 `61/100`보다 뚜렷하게 감소
- next-intercept reachable rate가 v14의 `0.488`보다 증가
- mean ball lateral speed가 v14의 `0.143m/s`보다 감소
- useful bounce가 v14의 `0.85`보다 유지 또는 증가
- upward contact below target rate가 v14의 `0.768`보다 감소

만약 v15에서도 `ball_out_of_bounds`가 높으면, 다음은 단순 reward 값 조절보다 `contact_frame_velocity_target`과 tilt primitive를 분리 A/B해서 "velocity feedforward가 옆속도를 만드는지"와 "tilt residual이 옆속도를 만드는지"를 별도로 끊어야 한다.
