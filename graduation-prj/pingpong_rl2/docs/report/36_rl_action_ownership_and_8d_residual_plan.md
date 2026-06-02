# RL action ownership and 8D residual implementation

작성일: 2026-06-02

## 배경

`pmk_cf_self_rally_v15`를 1M 학습한 결과, 2M까지 더 밀어도 자연히 해결될 가능성은 낮아 보인다.

저장된 `pmk_cf_self_rally_v15_training_summary.json` 기준:

- completed timesteps: `1,000,000`
- `mean_useful_bounces = 0.05`
- `max_useful_bounces = 1`
- `ball_out_of_bounds = 87 / 100`
- `ball_speed_limit = 6 / 100`
- `floor_contact = 5 / 100`
- `low_apex_contact = 2 / 100`

분석 요지:

- v15는 low apex보다 lateral direction/out-of-bounds 문제가 더 크다.
- 단순히 2M까지 더 학습하거나 reward weight만 조금 바꾸는 것보다, 현재 어떤 행동을 RL에 맡기고 어떤 행동을 primitive에 고정했는지 다시 나누는 것이 맞다.

## v15에서 RL이 직접 배우는 것

`pmk_cf_self_rally_v15` 기준 active mode는 `position_contact_frame`이다. PPO policy action은 5차원이며, 아래만 직접 출력한다.

1. `action[0]`: contact-frame radial XY 위치 residual
   - 대략 `+-0.02m`
   - 예상 접촉점/anchor 기준 radial 방향으로 라켓 목표점을 보정한다.

2. `action[1]`: contact-frame tangent XY 위치 residual
   - 대략 `+-0.02m`
   - radial에 수직인 tangent 방향으로 라켓 목표점을 보정한다.

3. `action[2]`: z 위치 residual
   - 대략 `+-0.03m`
   - scripted lift target 위/아래로 라켓 목표 높이를 보정한다.

4. `action[3]`: pitch tilt residual
   - v15 기준 `+-0.004rad`
   - scripted tilt 위에 pitch 기울기를 보정한다.

5. `action[4]`: roll tilt residual
   - v15 기준 `+-0.004rad`
   - scripted tilt 위에 roll 기울기를 보정한다.

즉 현재 policy가 직접 배우는 것은 라켓 목표 위치/기울기 residual이다. velocity, lift amount, desired outgoing XY velocity, contact timing은 직접 action으로 나오지 않는다.

## v15에서 RL이 직접 배우지 않는 것

아래는 hand-coded primitive/feedforward/constraint로 동작한다.

1. contact-frame planner
   - 다음 descending contact time/position 예측
   - target apex z 설정
   - target xy 설정
   - strike hold 여부

2. 기본 접촉 목표 위치
   - predicted/planned contact position
   - post-contact return target
   - anchor 기준 target

3. lift 계열
   - `contact_frame_base_strike_z_boost`
   - `contact_frame_base_strike_z_offset`
   - `contact_frame_apex_lift`
   - `contact_frame_low_apex_recovery_lift`
   - `contact_frame_velocity_lead`

4. velocity target 계열
   - intercept velocity target
   - desired outgoing velocity 기반 required racket velocity
   - low-apex recovery velocity
   - velocity target max clamp

5. tilt primitive
   - trajectory tilt
   - centering tilt
   - tilt ramp
   - tilt limit/clamp

6. post-contact recovery
   - rising ball 때 return target
   - body clearance
   - nullspace posture
   - controller gains

7. reward/success/failure 판정
   - useful contact 조건
   - low-apex termination
   - next-intercept reachable
   - easy-next-ball score
   - lateral stability reward

## RL에 맡길 후보

### 1. velocity/lift residual

우선순위: 높음.

이유:

- 현재 z 위치 residual은 있지만 실제 위로 치는 속도는 primitive가 대부분 결정한다.
- 낮은 apex 회복, 과한 수직 속도, 부족한 타격 힘을 policy가 직접 보정할 수 없다.
- full velocity control은 어렵지만 `vz_scale` 같은 bounded scale은 안정적인 중간 형태다.

후보:

- `target_velocity_z_scale`
- `required_racket_velocity_z_scale`
- low-apex recovery velocity/lift scale

### 2. outgoing XY velocity residual

우선순위: 매우 높음.

이유:

- v15의 `ball_out_of_bounds=87/100`은 방향 문제가 핵심이라는 신호다.
- contact point residual만으로는 "공을 어느 XY 방향으로 보내야 하는지"를 명확히 조절하기 어렵다.
- `heading` 하나보다 `outgoing_x_residual`, `outgoing_y_residual`이 디버깅하기 쉽다.

후보:

- desired outgoing velocity x residual
- desired outgoing velocity y residual

### 3. tilt primitive scale

우선순위: 중간-높음.

이유:

- raw tilt를 크게 배우게 하면 lateral instability가 커질 수 있다.
- scripted trajectory/centering tilt의 scale만 배우게 하면 안전하다.

후보:

- trajectory tilt scale
- centering tilt scale

### 4. low-apex recovery amount

우선순위: 중간.

이유:

- 현재 low-apex recovery는 고정 gain이다.
- 낮은 공을 매번 같은 방식으로 세게 치면 lateral drift가 생길 수 있다.

후보:

- recovery lift scale
- recovery velocity scale

### 5. contact timing/intercept time offset

우선순위: 중간-낮음.

이유:

- 같은 힘/tilt라도 너무 이르거나 늦게 맞으면 방향이 망가진다.
- 하지만 contact timing을 바로 policy action으로 열면 난도가 커진다.
- velocity/lift/outgoing XY residual 다음 단계가 맞다.

후보:

- contact time offset
- strike hold duration scale

## 7D vs 8D 판단

7D 후보:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_xy_heading]
```

8D 후보:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual]
```

판단:

- 현재 문제는 heading을 조금 돌리는 수준보다 `ball_out_of_bounds`가 큰 방향 실패다.
- `outgoing_x/y_residual`은 analysis CSV에서 component별로 바로 볼 수 있다.
- x/y residual은 reward나 clipping 문제도 분리 진단하기 쉽다.

따라서 다음 구현은 8D가 더 적합하다.

## 구현 방향

기존 `position_contact_frame` 5D mode는 보존한다. 기존 v14/v15 모델과 분석 호환성을 깨지 않기 위해 새 action mode를 추가한다.

새 mode:

```text
position_contact_frame_velocity_residual
```

새 action layout:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual]
```

예상 동작:

- 앞 5개는 기존 `position_contact_frame`과 같다.
- `vz_scale`은 desired outgoing velocity z / required racket velocity z 계열을 bounded scale로 보정한다.
- `outgoing_x_residual`, `outgoing_y_residual`은 desired outgoing velocity XY target에 직접 더한다.
- 새 residual은 info/analysis CSV에 반드시 기록한다.

주의:

- full velocity action으로 가면 학습 난도가 급격히 오른다.
- 따라서 action은 residual/scale 형태로 두고, primitive와 reward gate는 유지한다.

## 2026-06-02 구현 결과

새 action mode를 추가했다.

```text
position_contact_frame_velocity_residual
```

v16용 action layout:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual]
```

구현 세부사항:

- 기존 `position_contact_frame` 5D mode는 그대로 보존했다.
- `contact_frame_self_rally_candidate` preset은 새 8D mode로 전환했다.
- `vz_scale`은 residual 값이다.
  - 실제 controller-side scale은 `max(0, 1 + action[5])`.
  - preset 기본 limit은 `+-0.35`라서 기본적으로 `0.65x ~ 1.35x` 범위다.
- `outgoing_x_residual`, `outgoing_y_residual`은 controller가 시도하는 desired outgoing velocity XY에 더한다.
  - preset 기본 limit은 각 `+-0.35 m/s`.
- reward/metric 기준이 되는 base `_desired_outgoing_velocity()`는 그대로 둔다.
  - 정책이 목표 자체를 바꾸어 보상을 속이지 않게 하기 위함이다.
  - 실제 컨트롤러/primitive가 쓰는 목표만 `_contact_frame_controller_desired_velocity()`에서 residual을 적용한다.
- trajectory tilt, apex lift, required racket velocity, velocity target, followthrough는 controller-side desired velocity를 사용하도록 바꿨다.
- info/analysis CSV에 다음 값을 추가했다.
  - `applied_action_5_vz_scale`
  - `applied_action_6_outgoing_x_residual`
  - `applied_action_7_outgoing_y_residual`
  - `contact_frame_vz_scale`
  - `contact_frame_outgoing_x_residual_action`
  - `contact_frame_outgoing_y_residual_action`
  - `contact_frame_controller_desired_velocity_*`
- heuristic bootstrap은 8D mode에서 뒤 3개 residual을 0으로 채워 동작하게 했다.

수정 파일:

- `src/pingpong_rl2/envs/keepup_env.py`
- `src/pingpong_rl2/controllers/heuristic_keepup.py`
- `src/pingpong_rl2/defaults.py`
- `src/pingpong_rl2/utils/ppo_runs.py`
- `scripts/run_ppo_learning.py`
- `scripts/run_ppo_rebound_analysis.py`
- `scripts/run_heuristic_keepup_diagnostic.py`
- `tests/test_keepup_env.py`

## v16 학습 직전 검증

검증 결과:

- `py_compile` 통과
  - env/controller/defaults/ppo_runs/training/analysis/heuristic diagnostic script
- `tests.test_keepup_env` 통과
  - `101 tests`
- `tests.test_ppo_runs tests.test_keepup_contract_features tests.test_vector_env tests.test_scene_load` 통과
  - `17 tests`
- preset/env 생성 확인 통과
  - `contact_frame_self_rally_candidate` -> `position_contact_frame_velocity_residual`
  - action shape `(8,)`
  - observation shape `(55,)`
  - heuristic action shape `(8,)`
  - velocity scale/outgoing xy limit `0.35`
- 짧은 PPO smoke run 통과
  - action mode `position_contact_frame_velocity_residual`
  - smoke model 저장 성공

주의:

- action dimension이 5D에서 8D로 바뀌었으므로 v15 checkpoint를 이어받으면 안 된다.
- v16은 새 모델로 시작해야 한다.

## 다음 학습 명령

1M 먼저 권장한다. v15처럼 2M까지 그냥 미는 것보다, 8D residual이 방향/높이 문제를 잡는지 1M에서 확인하는 것이 빠르다.

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v16 \
  --reset-model \
  --total-timesteps 1000000
```

학습 후 분석 명령:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v16/pmk_cf_self_rally_v16_best_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v16_contact_diagnosis
```

분석에서 우선 볼 것:

- `ball_out_of_bounds`
- `mean_useful_bounces`, `max_useful_bounces`
- `stable_cycle_count`
- `applied_action_5_vz_scale`
- `applied_action_6_outgoing_x_residual`
- `applied_action_7_outgoing_y_residual`
- `outgoing_velocity_xy_error`
- `next_intercept_reachable`
- projected apex height / low-apex contact 비율

## 2026-06-03 v17 ownership update

v17에서 새 action mode를 추가했다.

```text
position_contact_frame_velocity_tilt_residual
```

v17 action layout:

```text
[radial, tangent, z, pitch, roll, vz_scale, outgoing_x_residual, outgoing_y_residual, racket_vz_residual, trajectory_tilt_scale, centering_tilt_scale]
```

v17에서 RL이 직접 배우는 것:

1. `radial`, `tangent`, `z`
   - 기존 contact-frame 목표 위치 residual.
2. `pitch`, `roll`
   - 기존 tilt residual.
3. `vz_scale`
   - desired outgoing velocity z 계열을 `max(0, 1 + action[5])` 형태로 스케일한다.
4. `outgoing_x_residual`, `outgoing_y_residual`
   - controller-side desired outgoing XY velocity에 직접 더한다.
5. `racket_vz_residual`
   - `_contact_frame_velocity_target()`의 z target velocity에 직접 더한다.
   - v16에서 `actual racket velocity z`가 낮고, `vz_scale`이 충분히 쓰이지 않은 문제를 겨냥한다.
6. `trajectory_tilt_scale`, `centering_tilt_scale`
   - scripted trajectory/centering tilt primitive를 `max(0, 1 + residual)`로 스케일한다.
   - raw tilt를 크게 여는 대신, 기존 안정화 primitive의 강도를 policy가 조절하게 한다.

v17에서도 아직 RL이 직접 배우지 않는 것:

- contact planner의 target time/position 계산
- target apex와 return target 생성
- low-apex recovery primitive의 기본 gain/max
- controller gain, max velocity/orientation step
- body clearance/nullspace posture
- useful/stable cycle 판정과 reward gate

따라서 v17은 full controller learning이 아니라, hand-coded juggling primitive 위에 velocity execution과 tilt 강도 조절 권한을 더 주는 구조다. v16에서 포착된 병목이 "목표 outgoing velocity는 있는데 실제 racket velocity와 타격 준비 시간이 부족함"이었기 때문에, 우선순위는 더 타당하다.
