# Sweet Spot Start Range And Next Work

Date: 2026-06-01

## Conclusion

지금은 값을 조금씩 바꿀 단계가 아니다. 먼저 초기 공이 라켓 중앙 sweet spot 안에서만 시작되도록 reset 범위를 잠그고, 그 상태에서 "중앙에서 맞은 공을 다시 치기 쉬운 위치와 높이로 보내는 primitive"가 되는지 확인해야 한다.

현재 프로젝트에서 sweet spot으로 쓸 수 있는 가장 직접적인 기준은 코드의 `contact_centering_radius = 0.04 m`이다. `PingPongKeepUpEnv`는 이 반경 안에서 접촉해야 `useful_keepup_bounce`로 인정한다.

따라서 현재 정사각형 reset sampler를 그대로 쓴다면:

```text
sweet spot circle radius = 0.040000 m
reset_xy_range max       = 0.040000 / sqrt(2)
                         = 0.028284 m
```

즉 sweet spot만 시작 위치가 가능하게 하려면 `--reset-xy-range 0.028` 정도로 시작해야 한다. 현재 기본값 `0.06`은 sweet spot-only가 아니다.

## Current Scene Geometry

아래 값은 현재 MuJoCo scene에서 직접 계산한 값이다.

```text
racket_center world xyz = [0.554512, 0.125000, 0.524502]
racket_head radius      = 0.084 m
racket_rim radius       = 0.090 m
ball radius             = 0.020 m
contact_centering_radius= 0.040 m
strike_zone_xy_radius   = 0.100 m
default reset_xy_range  = 0.060 m
```

현재 reset은 `racket_center + [xy_offset_x, xy_offset_y, ball_height]`로 공을 생성한다. 기본 `ball_height=0.50`이면 시작 z는 약 `1.024502 m`이다.

## Sweet Spot Start Range

현재 코드의 reset sampler는 원이 아니라 정사각형이다.

```python
uniform(-reset_xy_range, reset_xy_range, size=2)
```

그래서 정사각형의 모든 점이 반지름 4 cm sweet spot 안에 들어오려면 `reset_xy_range <= 2.828 cm`여야 한다.

권장 초기 학습 범위:

```text
reset_xy_range = 0.028 m

relative x range = [-0.028284, +0.028284] m
relative y range = [-0.028284, +0.028284] m

world x range    = [0.526227, 0.582796] m
world y range    = [0.096716, 0.153284] m
```

만약 reset sampler를 원형으로 새로 만든다면 `radius <= 0.040 m`까지 sweet spot-only로 쓸 수 있다. 하지만 현 코드 그대로라면 `--reset-xy-range 0.028`이 맞다.

참고로 "라켓 헤드 위에 물리적으로 올라간다"와 "sweet spot에 올라간다"는 다르다.

```text
conservative head-face circle radius = racket_head_radius - ball_radius
                                    = 0.084 - 0.020
                                    = 0.064 m
square reset max for head face       = 0.045255 m
```

하지만 최종 목표에는 라켓 중앙에서 안정적으로 위로 올려치는 것이 먼저라서, 학습 시작 curriculum은 head-face 범위가 아니라 sweet spot 범위인 `0.028 m`로 제한해야 한다.

## Is The Current Direction Correct?

일부 방향은 맞았다. 특히 아래 방향들은 최종 목표와 연결된다.

- 공을 단순히 맞히는 게 아니라 다음에 다시 치기 쉬운 위치로 보내는 `desired_outgoing_velocity`
- 일정 힘이 아니라 목표 apex 높이를 맞추는 `target_ball_height`
- 라켓을 수평 고정하지 않고 pitch/roll을 이용해 다음 궤적을 만드는 `trajectory_tilt`
- PPO 전에 heuristic/BC bootstrap으로 첫 primitive를 잡는 방식

하지만 지금까지의 진행은 값 조정이 너무 많았고, 초기 조건과 제어 가능성 검증이 충분히 고정되지 않았다. 그래서 모델이 나아지는 것처럼 보여도 실패 원인을 분리하기 어렵다.

현재 가장 중요한 판단은 이것이다.

```text
reward 숫자 조정만으로는 해결하기 어렵다.
먼저 sweet spot-only reset에서 반복 타격 primitive가 안정적으로 성립해야 한다.
그 다음 reset 범위를 넓히고, link5 충돌을 controller/IK posture 쪽에서 해결해야 한다.
```

## Next Work Order

다음 에이전트는 아래 순서대로만 진행한다.

1. Sweet spot-only curriculum을 고정한다.

   - `reset_xy_range = 0.028`
   - `reset_velocity_xy_range = 0.0`
   - `reset_velocity_z_range = (-0.01, 0.01)`
   - `target_ball_height = 0.20`
   - 이 조건에서만 먼저 학습/평가한다.

2. 학습 전에 reset 진단을 추가하거나 실행한다.

   - reset된 모든 공의 XY offset이 `sqrt(dx^2 + dy^2) <= 0.04`인지 확인한다.
   - 이 진단이 실패하면 PPO 학습을 하지 않는다.

3. 현재 best 계열 모델을 sweet spot-only 조건에서 다시 평가한다.

   - 확인할 지표는 평균 reward가 아니라 아래 순서다.
   - `mean_useful_bounces`
   - `two_or_more_rate`
   - `three_or_more_rate`
   - `robot_body_contact` 특히 `link5`
   - `predicted_next_intercept_xy_error`
   - `projected_contact_apex_height_above_racket`

4. sweet spot-only에서도 link5 충돌이 남으면 reward를 더 만지지 말고 controller/IK를 손댄다.

   - `src/pingpong_rl2/controllers/ee_pose_controller.py`에 nullspace posture 또는 link5 clearance bias를 넣는다.
   - 목표는 라켓 center pose는 유지하면서 팔꿈치/link5가 공 경로에서 비켜나게 하는 것이다.
   - 라켓 목표 위치를 공에서 멀리 옮기는 방식은 keep-up 성능을 떨어뜨렸으므로 기본 해법이 아니다.

5. sweet spot-only에서 성공 기준을 넘은 뒤에만 reset 범위를 넓힌다.

   - 1단계: `reset_xy_range = 0.028`
   - 2단계: 원형 sampler를 만들면 radius `0.040`, 아니면 square `0.035` 이하로 조심스럽게 확장
   - 3단계: head-face 범위인 square `0.045`
   - 4단계: 기존 기본값 `0.060`

## Stop Criteria For Random Tweaking

아래 중 하나라도 없으면 더 이상 값 sweep을 하지 않는다.

- sweet spot-only reset 진단 결과
- contact별 apex height 분포
- next intercept XY error 분포
- robot body contact 부위별 카운트
- 같은 seed 범위에서 100 episode 평가

이 기준 없이 `target_ball_height`, `tilt`, `offset`, `reward weight`를 조금씩 바꾸는 작업은 최종 목표에 직접적이지 않다.

## Recommended First Command

기존 코드 상태에서 바로 확인할 최소 평가 명령은 아래다.

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_recovery_roll_retract_bootstrap_h020_zero_eval/pmk_cf_recovery_roll_retract_bootstrap_h020_zero_eval_model.zip \
  --episodes 100 \
  --seed 12602 \
  --analysis-name sweet_spot_only_h020_eval \
  --reset-xy-range 0.028 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --target-ball-height 0.20 \
  --require-reachable-next-intercept-for-success \
  --min-easy-next-ball-score-for-success 0.0
```

이 결과가 좋아지지 않거나 link5 충돌이 계속 크면, 다음 작업은 PPO가 아니라 IK/nullspace posture 보완이다.
