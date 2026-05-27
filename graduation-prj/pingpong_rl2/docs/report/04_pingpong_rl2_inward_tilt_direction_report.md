# inward tilt direction A/B 보고서

## 1. 목적

이 문서는 `xycatch_v1` baseline이 공을 계속 로봇에서 먼 쪽으로 보내는지 확인하고, 그 다음 최소 control 수정으로 `inward rebound`를 만들 수 있는지 점검한 기록이다.

이번 턴의 목표는 세 가지였다.

1. 실제 rebound 방향이 정말 `away-side`인지 숫자로 확인한다.
2. reward를 더 붙이지 않고 `position_tilt`에서 inward pitch만 허용하는 최소 A/B를 만든다.
3. 그 A/B가 실제로 tilt를 사용하는지까지 50-episode 기준으로 확인한다.

## 2. 사전 진단

기준 모델은 `ppo_minimal_keepup_xycatch_v1`이다.

50-episode rebound analysis 결과는 아래와 같았다.

- total contacts: `97`
- useful contacts: `26`
- `ball_velocity_x > 0`: 전체 `95/97`, useful `24/26`
- mean contact `ball_velocity_x`: `+0.4727`
- mean useful contact `ball_velocity_x`: `+0.2464`

즉 현재 병목은 단순 XY catch-up만이 아니라, contact 이후 공을 거의 항상 `+x` 바깥쪽으로 보내는 방향 편향이다.

기존 `ppo_position_tilt_limited_500k`도 같은 방향 문제를 보였다.

- negative pitch 사용: `567/572` contacts
- useful contact의 negative pitch: `22/22`

따라서 가설은 아래였다.

- `position_tilt`가 inward 쪽 pitch만 쓰도록 제한하면, 최소한 rebound 방향 편향은 줄어들 수 있다.

## 3. 적용한 변경

### 3.1 env 제어 표면 추가

`src/pingpong_rl2/envs/keepup_env.py`에 아래 두 옵션을 추가했다.

- `target_pitch_range`
  - tilt action을 누적한 뒤 target pitch를 지정 구간으로 clamp
  - 이번 실험에서는 `0.0 .. 0.06`만 허용해서 outward-side pitch를 금지
- `initial_target_tilt`
  - env reset 직후 초기 target tilt를 지정
  - 이번 실험에서는 `(0.03, 0.0)`을 넣어 zero-tilt 경계에 붙는 현상을 깨려 했다

이 변경은 opt-in이다.

- 기존 run은 아무 동작 변화가 없다.
- 새 A/B run만 명시적으로 이 옵션을 사용한다.

### 3.2 학습 스크립트 실험 인자 추가

`scripts/run_ppo_learning.py`에 아래 CLI를 추가했다.

- `--target-pitch-range LOW HIGH`
- `--initial-target-tilt PITCH ROLL`

training summary의 `env_config`에도 그대로 저장되므로, 이후 evaluation/viewer/rebound analysis가 같은 env 설정을 재구성할 수 있다.

### 3.3 회귀 테스트 추가

`tests/test_keepup_env.py`에 아래를 추가했다.

- outward pitch action이 `target_pitch_range`에 의해 `0.0`으로 잘리는지 확인
- `training_config()`에 `target_pitch_range`가 남는지 확인
- reset 시 `initial_target_tilt`가 실제 observation/info에 적용되는지 확인

## 4. 검증

코드 변경 뒤 아래 검증을 실행했다.

1. `PYTHONPATH=pingpong_rl2/src conda run -n mujoco_env python -m unittest discover -s pingpong_rl2/tests -p 'test_*.py'`
   - `23` tests passed
2. inward-only clamp smoke train
   - `ppo_position_tilt_inward_smoke_v1`
3. initial tilt bias smoke train
   - `ppo_position_tilt_inward_bias_smoke_v1`

그 다음 실제 A/B는 아래 두 개를 실행했다.

1. `ppo_position_tilt_inward_early_200k`
   - `target_pitch_range=(0.0, 0.06)`
   - `initial_target_tilt` 없음
2. `ppo_position_tilt_inward_bias_early_100k`
   - `target_pitch_range=(0.0, 0.06)`
   - `initial_target_tilt=(0.03, 0.0)`

각 run에 대해 50-episode evaluation과 50-episode rebound analysis를 함께 실행했다.

## 5. 결과

### 5.1 정리 표

| run | 설정 | 50ep mean useful bounces | mean contacts | total contacts | useful contact rate | `+x` contacts | mean useful `x` | contact 시 mean pitch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_minimal_keepup_xycatch_v1` | position baseline | `0.52` | `2.08` | `97` | `26.8%` | `95/97` | `+0.2464` | 해당 없음 |
| `ppo_position_tilt_inward_early_200k` | inward clamp만 적용 | `0.20` | `0.94` | `39` | `20.5%` | `39/39` | `+0.1410` | `0.0` |
| `ppo_position_tilt_inward_bias_early_100k` | inward clamp + initial tilt bias | `0.08` | `0.76` | `36` | `13.9%` | `36/36` | `+0.1468` | `0.0` |

### 5.2 핵심 관찰

핵심은 성능 수치보다 `tilt usage`다.

두 inward A/B 모두 contact 시점 `target_tilt_0`가 사실상 전부 `0.0`이었다.

- clamp-only run
  - all negative pitch: `0`
  - all positive pitch: `0`
  - mean pitch at contact: `0.0`
- initial-bias run
  - all zero pitch contacts: `36/36`
  - useful zero pitch contacts: `5/5`
  - mean pitch at contact: `0.0`

즉 policy는 `negative pitch`를 positive pitch로 바꿔 쓴 것이 아니라, 그냥 `zero pitch`로 수렴했다.

## 6. 해석

### 6.1 확인된 것

이번 작업으로 아래는 확인됐다.

1. 현재 baseline의 outward rebound 편향은 실제 현상이다.
2. 기존 tilt run의 문제도 방향적으로는 같은 쪽이었다.
3. 하지만 `negative pitch 금지`만으로는 inward steering이 생기지 않는다.

### 6.2 왜 실패했는가

실패 원인은 간단하다.

- PPO가 `positive pitch`를 활용하는 정책을 찾은 것이 아니라
- 제약된 action 공간 안에서 `tilt를 안 쓰는 것이 더 쉬운 local optimum`으로 간 것에 가깝다.

`initial_target_tilt=(0.03, 0.0)`까지 줘도 contact 시점에는 다시 `0.0`으로 돌아왔다는 점이 이 해석을 뒷받침한다.

즉 다음 병목은 `tilt direction sign` 자체가 아니라, `tilt를 실제 strike control에 참여시키는 bootstrap/control contract`다.

## 7. 현재 결론

현재 결론은 아래와 같다.

1. inward-only clamp 옵션은 유지할 가치가 있다.
   - 방향 부호 실험 표면이 분리됐기 때문이다.
2. 하지만 이것만으로는 성능 개선이 없다.
3. reset 시 작은 initial bias를 줘도, 현재 학습에서는 contact 시점 tilt usage가 살아나지 않는다.

따라서 다음 실험은 reward가 아니라 아래 둘 중 하나가 맞다.

1. `tilt` action을 `0` 기준 residual이 아니라 `positive inward base tilt + residual` 형태로 재파라미터화한다.
2. position에서 첫 contact를 더 안정화한 뒤, 그 정책 근처에서 tilt branch를 warm-start 하거나 curriculum으로 연다.

이번 턴 기준으로는 `tilt를 inward 쪽으로만 허용하면 해결된다`라는 가설은 기각됐다.

더 정확히 말하면,

- `tilt 방향을 바르게 제한하는 것`은 필요하지만
- `tilt를 쓰도록 학습시키는 장치`가 따로 없으면 policy는 그냥 tilt를 버린다.