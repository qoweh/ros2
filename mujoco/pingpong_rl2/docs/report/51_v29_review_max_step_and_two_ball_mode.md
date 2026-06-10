# v29 검토, max step, 1공/2공 모드 판단

작성일: 2026-06-04

## 1. v29 결론

`pmk_cf_self_rally_v29_racket_tracking_staged_distribution`은 v28보다 개선됐다고 보기 어렵다.

100 episode rebound analysis 기준:

| 모델 | 조건 | mean useful | max useful | 30+ rate | 주요 실패 |
|---|---:|---:|---:|---:|---|
| v28 tracking/spin | wide+velocity+spin | 8.16 | 45 | 8.75% | `ball_out_of_bounds`, `low_apex_contact` |
| v29 staged tracking/spin | wide+velocity+spin | 6.31 | 33 | 2.0% | `ball_out_of_bounds`, `robot_body_contact`, `ball_speed_limit` |
| v26 | v26 easy distribution | 42.34 | 85 | 60.0% | mostly `low_apex_contact`/`time_limit` |

v29의 좋은 변화:

- 평균 apex는 v28보다 높아졌다.
  - v28 mean projected apex: `0.205m`
  - v29 mean projected apex: `0.230m`
- low-apex failure는 줄었다.
  - v28: `27/80`
  - v29: `14/100`
- tracking residual 사용량은 늘었다.
  - v28 tracking vx/vy mean abs: `0.0056 / 0.0105`
  - v29 tracking vx/vy mean abs: `0.0237 / 0.0483`

v29의 나쁜 변화:

- useful contact rate가 낮아졌다.
  - v28: `26.0%`
  - v29: `21.9%`
- next-intercept reachable rate가 낮아졌다.
  - v28: `72.9%`
  - v29: `61.1%`
- lateral speed/outward speed가 커졌다.
  - mean ball lateral speed: `0.102 -> 0.124`
  - mean racket outward speed: `0.028 -> 0.046`
- `ball_out_of_bounds`가 늘었다.
  - v28: `34/80`
  - v29: `45/100`

해석:

v29는 tracking action을 쓰기 시작했지만 안정적인 추적 제어로 정착하지 못했다. tracking residual이 lateral drift를 줄이는 쪽으로만 학습된 것이 아니라, 일부 episode에서는 공을 더 밖으로 보내는 방향의 과보정이 생겼다.

## 2. max episode step을 늘리면 좋아지는가?

현재 v29 env는 `max_episode_steps=null`, 즉 학습 환경 자체는 무제한이다.

`run_ppo_rebound_analysis.py`와 viewer에서 넣는 `--episode-step-limit 1800` / `--max-episode-steps 1800`은 분석/렌더링 안전 cap이다.

따라서 max step을 늘리는 것은 "정책 성능을 올리는 학습 기법"은 아니다.

다만 평가나 viewer에서 이미 잘하고 있는 episode가 `time_limit`으로 끝나는 경우, 더 오래 보는 효과는 있다.

v29 100 episode 분석:

- `time_limit`: `6/100`
- 나머지 대부분은 `ball_out_of_bounds`, `robot_body_contact`, `ball_speed_limit`, `low_apex_contact`

즉 v29에서는 max step을 늘려도 대부분의 실패가 해결되지 않는다. 핵심은 horizon 부족이 아니라 lateral stability와 next-intercept 품질이다.

## 3. 다음 개선 방향

v29 계열을 계속 밀기보다, 발표/웹서비스 기본 모델은 v26 계열을 기준으로 잡는 것이 맞다.

v26은 더 쉬운 reset 분포이지만 안정성이 압도적으로 좋다.

- mean useful: `42.34`
- max useful: `85`
- 30+ rate: `60%`
- ball_out_of_bounds: `5/100`

따라서 다음 실험은 v26 모델에서 action 구조를 바꾸지 않고, reset XY만 작게 확장한다.

새 preset:

- `contact_frame_self_rally_v30_v26_wider_xy_stability`

새 config:

- `configs/keep1_v30.json`

v30 설정:

- resume base: `pmk_cf_self_rally_v26_model.zip`
- action mode: v26과 같은 15D apex residual
- reset XY: `0.075m -> 0.10m`
- reset velocity: v26 유지
- spin: `0`
- total timesteps: `1M`

목표:

- v26의 30+ 안정성을 최대한 유지하면서 시작 위치 범위만 약간 넓힌다.
- v29처럼 action 차원/스핀/속도/넓은 XY를 한 번에 바꾸지 않는다.

## 4. 1공/2공 모드

웹서비스에서 "공 하나"를 기본값으로 두고, 체크하면 "공 2개"로 바꾸는 UI는 가능하다. 다만 모델은 분리하는 것이 맞다.

현재 1공 모델은 observation/action/reward가 모두 "공 하나를 추적해서 한 번에 하나의 contact를 만드는" 구조다. 공 2개가 되면 최소한 아래가 바뀐다.

- observation: ball state가 2개 필요
- reward: 두 공 각각의 useful bounce/실패 판정 필요
- termination: 한 공 실패 시 전체 실패인지, 남은 공 유지인지 정의 필요
- planner: 다음에 칠 공을 선택하는 scheduling logic 필요
- policy: 같은 라켓 하나로 두 공을 번갈아 치는 타이밍을 학습해야 함

따라서 현재 1공 모델을 그대로 2공 모드에 쓰면 제대로 일반화되기 어렵다.

권장 구조:

- 기본 1공 모드: 안정 모델 `v26` 또는 v30 계열 사용
- 2공 모드: 별도 environment + 별도 PPO 모델
- 웹 UI: checkbox에 따라 model path와 env mode를 바꾸는 방식

한 모델로 1공/2공을 모두 처리하려면 처음부터 "max 2 balls + active mask" observation으로 설계해야 한다. 현재 구조에서는 발표 안정성을 위해 모델 분리가 낫다.

## 5. v30 학습 명령

```bash
cd mujoco/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py --config-file configs/keep1_v30.json
```

학습 후 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v30/keep1_v30_model.zip \
  --episodes 100 \
  --seed 231 \
  --episode-step-limit 1800 \
  --analysis-name keep1_v30_eval100
```
