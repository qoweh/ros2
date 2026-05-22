# EE Delta Env 계약 메모

이 문서는 현재 `option 1`로 선택한 position-only EE 제어 경로의 계약을 간단히 고정하기 위한 메모다.

## 1. 현재 선택

이번 단계에서 채택한 방향은 다음이다.

- action은 orientation 없이 `racket_center` 기준 delta xyz
- low-level sim 위에 얇은 task env를 하나 더 둠

즉 지금은 full RL wrapper가 아니라, 그 직전의 최소 계약을 만든 상태다.

## 2. 현재 action 계약

형태:
- `action.shape == (3,)`

의미:
- `(dx, dy, dz)` in meters
- 현재 target position에 더해지는 delta
- step당 절대값 `0.03m`로 clip

현재 중요한 점:
- normalization은 아직 없음
- orientation action은 아직 없음

즉 지금 action은 “실제로 얼마나 움직일지”를 바로 넣는 low-level 계약이다.

## 2.1 현재 success 계약

이번 단계에서 success termination도 추가했다.

- `success_velocity_threshold`: 기본 `0.5`
- 조건:
	- `ball_geom`와 `racket_head`가 contact 상태이고
	- `ball_velocity_z > success_velocity_threshold`

현재 success reason 이름은 아래 하나다.

- `upward_racket_bounce`

의도:
- 라켓을 그냥 스친 것과
- 실제로 위로 튀겨 보낸 것을
구분하기 위함이다.

## 3. 현재 observation 계약

public observation은 flat vector 하나로 고정한다.

- `observation.shape == (29,)`

현재 순서는 아래와 같다.

- `observation[0:7]`: `joint_positions`
- `observation[7:14]`: `joint_velocities`
- `observation[14:17]`: `racket_position`
- `observation[17:20]`: `racket_velocity`
- `observation[20:23]`: `target_position`
- `observation[23:26]`: `ball_position`
- `observation[26:29]`: `ball_velocity`

`target_position`을 추가한 이유:
- 현재 env는 delta action을 controller 내부의 `target_position`에 누적한다.
- 이 값이 observation에 없으면 같은 `racket_position`이라도 다음 transition이 달라질 수 있다.
- 즉 RL 관점에서 state를 닫으려면 `target_position`을 같이 관측해야 한다.
- contact 순간의 active hit를 판단하기 위해 `racket_velocity`도 관측한다.

디버그가 필요할 때는 다음 helper를 사용한다.

- `observation_dict()`
- `unflatten_observation()`

아직 빠진 것:
- racket orientation
- racket velocity
- ball spin

## 4. 현재 reset/step 반환 형식

이번 단계에서 Gymnasium 스타일로 반환 형식도 같이 고정한다.

- `reset() -> (observation, info)`
- `step() -> (observation, reward, terminated, truncated, info)`

현재는 env 내부 time limit도 같이 둔다.

- `max_episode_steps`: 기본 `300`
- `reset()` 시 `step_count = 0`
- `step()`마다 `step_count += 1`

- `terminated`: `failure_reason()`이 생기면 `True`
- `truncated`: `terminated`가 아니고 `step_count >= max_episode_steps`이면 `True`

현재 `info`에는 아래를 같이 넣는다.

- `step_count`
- `time_limit_reached`
- `failure_reason`
- `success_reason`
- `target_position`

그리고 logging용으로 아래도 같이 넣는다.

- reward 분해:
	- `reward_total`
	- `reward_height`
	- `reward_distance`
	- `reward_contact`
	- `reward_success`
	- `reward_failure`
- episode 경계:
	- `terminated`
	- `truncated`
	- `episode_steps`
- velocity:
	- `ball_velocity_x`
	- `ball_velocity_y`
	- `ball_velocity_z`
	- `ball_speed_norm`
- contact trace:
	- `contact_observed_during_step`
	- `contact_substep`
	- `contact_ball_velocity_x`
	- `contact_ball_velocity_y`
	- `contact_ball_velocity_z`
	- `contact_ball_speed_norm`

현재 reward logging에는 active hit 분해값도 포함한다.

## 5. 현재 reward draft

현재는 아래 정도만 들어 있다.

- `contact_bonus`
- `height_term`
- `active_hit_term`
- `failure_penalty`

이건 아직 최종 shaping이 아니라, `step()` 반환 형식이 성립하는지 확인하려는 임시 초안이다.

## 6. 현재 termination 계약

termination은 기존 `failure_reason()`를 그대로 따른다.

그리고 success가 잡히면 그 역시 `terminated = True`로 처리한다.

즉 아래가 현재 종료 후보다.
- `floor_contact`
- `ball_out_of_bounds`
- `nonfinite_state`
- `ball_speed_limit`
- `upward_racket_bounce`

time limit은 위 failure와 분리해서 `truncated`로만 처리한다.

## 7. 다음에 고정해야 할 것

다음 단계에서는 아래를 먼저 결정하는 편이 좋다.

1. reward에서 height term을 유지할지, contact 중심 sparse reward로 갈지
2. termination 후 즉시 auto-reset을 env 안에서 할지, wrapper 바깥에서 할지
3. success threshold를 얼마로 유지할지

현재 기준으로는 observation/반환 형식, time limit, success termination 1차안은 고정했고, 다음은 reward와 success threshold 튜닝을 정리하는 것이 맞다.
