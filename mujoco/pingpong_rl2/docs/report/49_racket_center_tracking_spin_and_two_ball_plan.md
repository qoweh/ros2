# 49. Racket-centered tracking residual, spin reset, and two-ball plan

## 1. 현재 v25/v26에서 hand-coded인 부분

v25/v26 계열의 self-rally는 RL이 로봇 관절을 직접 제어하지 않는다. 큰 흐름은 아래와 같다.

- planner/controller가 공의 하강 교차점과 racket target position/velocity를 계산한다.
- controller가 target position/tilt/velocity를 관절 target으로 바꾼다.
- RL은 그 위에 residual action을 더한다.

현재 15D `position_contact_frame_velocity_tilt_lateral_apex_residual`에서 RL이 직접 내는 값:

```text
[radial, tangent, z,
 pitch, roll,
 vz_scale, outgoing_x, outgoing_y,
 racket_vz,
 trajectory_tilt_scale, centering_tilt_scale,
 racket_x_velocity, racket_y_velocity,
 target_apex_z, strike_plane_z]
```

아직 hand-coded로 남아 있는 핵심:

- predicted/contact intercept 계산
- contact target XY 선택
- intercept velocity target
- lateral brake velocity
- trajectory/centering tilt 계산
- low-apex recovery lift/velocity
- Cartesian controller와 body-clearance guard

이 전체를 RL로 대체하면 action/observation/credit assignment가 너무 커져 현재 안정 keep-up 성능을 잃을 가능성이 크다. 그래서 이번 변경은 "추적 제어 전체"가 아니라, 넓은 racket-centered 초기 위치에서 병목이 되는 pre-contact XY 추적 속도만 RL residual로 열었다.

## 2. 이번 구현

새 action mode:

```text
position_contact_frame_velocity_tilt_lateral_apex_tracking_residual
```

기존 15D 뒤에 2축을 추가한 17D이다.

```text
15D + [tracking_x_velocity, tracking_y_velocity]
```

두 축은 공이 하강 중일 때만 `_contact_frame_velocity_target()`의 XY target velocity에 더해진다. 기존 `racket_xy_velocity` residual과 분리해 둔 이유는 분석 시 두 역할을 따로 볼 수 있게 하기 위해서다.

새 env/config 주요 인자:

- `contact_frame_tracking_xy_action_limit`
- `reset_ball_angular_velocity_range`

새 preset/config:

- preset: `contact_frame_self_rally_v28_racket_tracking_spin`
- config: `configs/pmk_cf_self_rally_v28_racket_tracking_spin.json`

v28 설정은 robot-base가 아니라 racket-centered이다.

- `reset_xy_range`: `0.075 -> 0.16m` curriculum
- `reset_xy_sampling`: `disk`
- `reset_velocity_xy_range`: `0.06m/s`
- `reset_velocity_z_range`: `(-0.16, 0.04)m/s`
- `reset_ball_angular_velocity_range`: `20rad/s`
- action mode: 17D tracking residual

주의: action dimension이 15D에서 17D로 바뀌었으므로 v26 PPO model에서 resume하지 않는다. 대신 heuristic bootstrap은 새 두 축을 0으로 채우도록 지원했다.

## 3. Spin에 대한 판단

MuJoCo freejoint의 angular qvel로 초기 spin을 넣었다. 단, 공이 단색 sphere이므로 viewer에서 회전 자체가 선명하게 보이지는 않는다.

물리 효과는 주로 racket contact friction을 통해 나타난다. 비행 중 Magnus effect 같은 공기역학은 현재 모델에 없다. "대각선으로 떨어지는 공"은 spin보다 `reset_velocity_xy_range`가 직접 담당한다.

## 4. 2공 keep-up 분석

2공 keep-up은 가능하지만 현재 코드에 바로 섞기에는 위험하다.

현재 구조는 `ball_joint`, `ball`, `ball_geom` 단일 객체를 강하게 전제한다. 2공으로 가려면 아래가 모두 바뀐다.

- XML에 `ball_1`, `ball_2` freejoint 추가
- `PingPongSim`의 ball position/velocity/contact trace를 단일 값에서 배열/active ball 구조로 변경
- observation에 두 공의 상대 위치/속도와 어떤 공을 칠 차례인지 포함
- reward를 "두 공 모두 살리기" 또는 "가장 긴급한 공 우선"으로 재정의
- failure도 한 공 floor/out-of-bounds와 전체 episode 종료 정책을 다시 정의

현실적인 순서:

1. 단일 공에서 17D + racket-centered 16cm + spin/diagonal drop을 먼저 학습한다.
2. 성공하면 2공은 별도 env/action/reward 실험으로 분리한다.
3. 두 공 사이의 초기 phase offset을 크게 두고, 항상 더 빨리 떨어지는 공을 active target으로 선택하는 curriculum부터 시작한다.

## 5. 검증

통과:

- `py_compile`
- `git diff --check`
- 새 17D action space test
- tracking residual target velocity test
- spin reset test
- heuristic 17D support test
- v28 config smoke
- `tests.test_ppo_runs`

전체 `tests.test_keepup_env`는 현재 별도 삭제 상태인 `assets/scene_racket_outward.xml` 때문에 1개 테스트가 실패할 수 있다. 이 파일 삭제는 이번 변경과 무관하다.
