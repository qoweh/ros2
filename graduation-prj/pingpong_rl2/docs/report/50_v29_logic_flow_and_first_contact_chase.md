# v29 logic flow and first-contact chase residual

Date: 2026-06-04

## 1. 현재 전체 로직 흐름

학습/실행 시 한 step은 아래 흐름으로 돈다.

1. `reset()`
   - 공 시작 높이, XY offset, 초기 속도를 샘플링한다.
   - v28부터는 `reset_xy_origin=robot_base`를 쓸 수 있어서, 탁구채 중심이 아니라 로봇 베이스 중심 기준으로 공을 떨어뜨릴 수 있다.

2. `observation()`
   - joint position/velocity, racket position/velocity, target position, ball position/velocity, ball-relative position, predicted intercept 등을 관측으로 만든다.
   - preset에 따라 phase, contact context, next intercept, desired outgoing velocity, velocity-domain observation이 추가된다.

3. PPO policy가 action을 출력한다.
   - action은 env의 `action_low/action_high`로 clip된다.
   - contact-frame 계열에서는 이 action이 절대 joint command가 아니라 여러 residual로 해석된다.

4. contact-frame planner가 접촉 후보를 계산한다.
   - 공이 하강 중이면 ballistic intercept time/contact position을 계산한다.
   - 목표 apex와 다음 target XY를 정하고, 원하는 outgoing ball velocity를 계산한다.

5. target position/tilt/velocity를 만든다.
   - target position: 예측 접촉점 + contact point residual + z/lift/feedforward.
   - target tilt: scripted/base tilt + trajectory tilt + centering tilt + RL tilt residual/scale.
   - target velocity: intercept velocity + required racket velocity + RL velocity residual + lateral brake + low-apex recovery velocity.

6. guard와 controller가 적용된다.
   - body-safe target, pre-contact XY/Z guard, body-clearance reference가 적용된다.
   - `RacketCartesianController`가 target position/tilt/velocity를 joint target으로 바꾼다.

7. MuJoCo step과 contact trace를 계산한다.
   - racket contact, robot body contact, floor contact, ball bounds, speed limit 등을 감지한다.

8. success/failure/reward를 계산한다.
   - useful bounce는 upward velocity, racket upward velocity, contact centering, apex window, next-intercept/easy-next-ball 조건을 통과해야 한다.
   - failure reason은 `ball_out_of_bounds`, `robot_body_contact`, `floor_contact`, `ball_speed_limit`, `low_apex_contact` 등으로 기록된다.

9. PPO 학습 loop가 reward와 transition으로 policy/value를 업데이트한다.
   - 학습 종료 후 final model과 training summary만 저장한다. checkpoint 코드는 제거된 상태다.

## 2. RL이 담당하는 부분

v28 이전/현재 15D 계열에서 RL이 직접 내던 action은 아래 residual들이다.

`position_contact_frame_velocity_tilt_lateral_apex_residual` 15D:

| 축 | 의미 |
| --- | --- |
| 0 `radial_contact_pos` | 접촉점 기준 radial 위치 residual |
| 1 `tangent_contact_pos` | 접촉점 기준 tangent 위치 residual |
| 2 `z_contact_pos` | 타격 z 위치 residual |
| 3 `pitch_tilt` | pitch tilt residual |
| 4 `roll_tilt` | roll tilt residual |
| 5 `vz_scale` | desired outgoing z velocity scale residual |
| 6 `outgoing_x_residual` | desired outgoing x velocity residual |
| 7 `outgoing_y_residual` | desired outgoing y velocity residual |
| 8 `racket_vz_residual` | racket target z velocity residual |
| 9 `tilt_scale_pitch` | trajectory tilt scale residual |
| 10 `tilt_scale_roll` | centering tilt scale residual |
| 11 `racket_vx_residual` | racket target x velocity residual |
| 12 `racket_vy_residual` | racket target y velocity residual |
| 13 `target_apex_z_residual` | target apex z residual |
| 14 `strike_plane_z_residual` | strike plane z residual |

반대로 RL이 직접 담당하지 않던 부분:

- ballistic intercept 계산
- desired outgoing velocity 기본값 계산
- base strike lift/feedforward
- low-apex recovery lift/velocity
- trajectory tilt와 centering tilt의 기본 계산식
- lateral brake
- body clearance/nullspace
- target guard와 IK/controller
- MuJoCo contact physics
- reset distribution 자체

따라서 v28의 "무작위 공을 잘 못 따라감"은 엄밀히 말하면 RL이 절대 target/joint를 직접 제어하지 못해서라기보다, hand-coded planner/controller가 넓은 robot-base disk를 커버하지 못하고 RL residual도 첫 접촉 획득을 직접 담당하기에는 작았기 때문이다.

## 3. v28 분석에서 본 병목

v28 robot-base disk를 100 episode, max 150 step으로 평가했을 때:

- mean useful bounce: `0.25`
- max useful bounce: `4`
- `ball_out_of_bounds`: `62/100`
- `robot_body_contact`: `14/100`
- `ball_speed_limit`: `9/100`
- `floor_contact`: `7/100`
- contact rate: 약 `14%`

분포별로 보면 `+x` 방향에서만 상대적으로 접촉이 나오고, `+y`, `-x`, `-y`는 거의 접촉하지 못했다. 즉 지금 단계에서 전체 360도 robot-base disk를 한 번에 안정화하기보다는, 로봇팔이 실제로 닿는 전방 sector에서 첫 접촉 skill을 먼저 학습시키는 편이 더 타당하다.

## 4. 이번 v29 변경

### 4.1 19D action mode 추가

새 action mode:

```text
position_contact_frame_velocity_tilt_lateral_apex_chase_residual
```

기존 15D에 아래 4축을 추가했다.

| 추가 축 | bound | 의미 |
| --- | ---: | --- |
| 15 `chase_vx_residual` | `+-0.80m/s` | 첫 접촉 전 racket target x velocity residual |
| 16 `chase_vy_residual` | `+-0.80m/s` | 첫 접촉 전 racket target y velocity residual |
| 17 `contact_x_residual` | `+-0.12m` | 첫 접촉 전 planned contact x residual |
| 18 `contact_y_residual` | `+-0.12m` | 첫 접촉 전 planned contact y residual |

이 4축은 `contact_count <= 0`이고 공이 하강 중일 때만 effective하게 적용된다. 첫 접촉 이후에는 기존 15D residual 구조가 self-rally 품질을 담당한다.

### 4.2 robot-base sector/annulus reset 추가

추가 env 인자:

- `reset_xy_min_radius`
- `reset_xy_angle_bounds_degrees`

disk sampling에서 최소 반경과 각도 sector를 줄 수 있다. v29 preset은 아래처럼 시작한다.

- origin: robot base `(0, 0)`
- radius: `0.35m ~ 0.68m`
- angle: `-60deg ~ +60deg`
- curriculum: `0.40m -> 0.68m`

이 값은 전체 360도가 아니라 v28에서 접촉 가능성이 확인된 전방 workspace를 먼저 배우기 위한 설정이다.

### 4.3 first-contact reward 추가

추가 reward:

- `first_contact_chase_term`
  - 첫 접촉 전 racket XY가 실제 예측 접촉점에 가까워질수록 dense reward를 준다.
  - RL이 contact residual로 reward를 속이지 않도록, reward 기준은 "보정 전 실제 planner/predicted contact position"이다.

- `first_contact_reach_term`
  - episode의 첫 racket contact를 만들면 보상을 준다.
  - useful bounce 이전에도 "일단 닿기"를 학습시키려는 용도다.

### 4.4 config/preset 추가

추가 preset:

```text
contact_frame_self_rally_v29_first_contact_chase_sector
```

추가 config:

```text
configs/pmk_cf_self_rally_v29_first_contact_chase_sector.json
```

v29는 action dimension이 `15D -> 19D`로 바뀌었기 때문에 v28 모델에서 resume하지 않는다. config에 `reset_model: true`를 넣었다.

## 5. 검증

정적 컴파일:

```bash
python3 -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  src/pingpong_rl2/controllers/heuristic_keepup.py \
  src/pingpong_rl2/defaults.py \
  src/pingpong_rl2/utils/ppo_runs.py \
  scripts/run_ppo_learning.py \
  scripts/run_viewer.py \
  scripts/run_ppo_evaluation.py \
  scripts/run_ppo_rebound_analysis.py \
  scripts/run_heuristic_keepup_diagnostic.py
```

통과.

Unit test:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest discover -s tests
```

결과:

```text
Ran 149 tests in 5.381s
OK
```

v29 config/env 생성 smoke:

```text
preset=contact_frame_self_rally_v29_first_contact_chase_sector
tilt_profile=early
action_mode=position_contact_frame_velocity_tilt_lateral_apex_chase_residual
action_size=19
reset_xy_min_radius=0.35
reset_xy_angle_bounds_degrees=[-60.0, 60.0]
```

## 6. v29 학습 명령

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v29_first_contact_chase_sector.json
```

학습 후 모델 위치:

```text
artifacts/ppo_runs/pmk_cf_self_rally_v29_first_contact_chase_sector/pmk_cf_self_rally_v29_first_contact_chase_sector_model.zip
```

config의 `run_name=pmk_cf_self_rally`, `run_version=v29_first_contact_chase_sector` 기준 실제 run name은:

```text
pmk_cf_self_rally_v29_first_contact_chase_sector
```

## 7. v29 이후 확인할 것

학습이 끝나면 먼저 아래를 본다.

- 첫 접촉률이 v28의 약 `14%`보다 올랐는가
- `ball_out_of_bounds`가 줄었는가
- 새 4축 중 `chase_vx/vy`, `contact_x/y`가 실제로 쓰이는가
- 전방 sector 안에서 useful bounce가 회복되는가
- sector를 `-75~75`, `-90~90`, 이후 `full 360`으로 넓힐 수 있는가

전체 360도는 v29 한 번으로 완성되기보다, reachable workspace를 점점 넓히는 curriculum이 더 현실적이다. 그래도 이번 v29는 단순 값 조절이 아니라, 첫 접촉 획득 자체를 RL이 보정할 수 있게 action abstraction을 확장한 변경이다.
