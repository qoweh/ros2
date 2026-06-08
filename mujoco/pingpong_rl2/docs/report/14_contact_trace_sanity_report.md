# contact trace sanity report

## 1. what changed

이번 단계에서는 reward를 더 만지지 않고, 먼저 `contact 직후 outgoing velocity`를 정말 제대로 보고 있는지부터 확인했다.

변경한 코드:

- `src/pingpong_rl2/envs/pingpong_sim.py`
  - `step_with_contact_trace()`를 확장해서 아래를 기록하게 했다.
    - contact 직전 ball velocity / position
    - first observed contact substep ball velocity
    - contact 이후 1-5 substep ball velocity
    - contact 종료 직후 ball velocity
    - MuJoCo raw contact position / raw contact normal
    - racket center velocity
    - racket face normal
    - pre-contact / contact relative velocity
- `src/pingpong_rl2/envs/keepup_env.py`
  - outgoing trajectory metric이 `contact_ball_velocity_*`만 쓰지 않고, 가능한 경우 `contact_end_ball_velocity` 또는 post-contact substep velocity를 우선 사용하게 바꿨다.
  - 기존 first-contact substep 기준 error도 같이 남겨서 비교 가능하게 했다.
- `scripts/run_heuristic_keepup_diagnostic.py`
  - contact-level CSV를 쓰도록 확장했다.
  - `actual_outgoing_velocity_source` 집계와 `raw contact vs stabilized post-contact` 차이를 summary에 추가했다.

핵심 목적은 하나였다.

> 기존 `outgoing_velocity_error_norm`이 실제 post-contact outgoing velocity를 보고 있었는지, 아니면 first observed contact substep의 중간값을 보고 있었는지 숫자로 확인한다.

## 2. commands executed

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name contact_trace_sanity_100ep_v1 \
  --variant-name followup_trace_sanity \
  --episodes 100 \
  --reset-xy-range 0.0 \
  --reset-velocity-xy-range 0.0 \
  --reset-velocity-z-range -0.01 0.01 \
  --followup-strike-target-tilt -0.03 0.0
```

보조 sign check:

```bash
PYTHONPATH=$PWD/src conda run -n mujoco_env python -c '... print(sim.racket_face_normal, controller.target_face_normal)'
conda run -n mujoco_env python -c '... compute raw contact normal alignment from contacts.csv ...'
```

## 3. numeric result

`contact_trace_sanity_100ep_v1_summary.json` 기준:

- `episodes = 100`
- `mean_useful_bounces = 0.53`
- `max_useful_bounces = 2`
- `two_or_more_useful_bounce_rate = 0.05`
- `three_or_more_useful_bounce_rate = 0.0`
- `contact_event_count = 325`

새 contact trace source 집계:

- `contact_end_ball_velocity = 305`
- `contact_ball_velocity = 19`
- `post_contact_1_ball_velocity = 1`
- `resolved_post_contact_source_rate = 0.9415`
- `contact_end_velocity_source_rate = 0.9385`

즉, 실제 outgoing metric은 대부분의 contact에서 first contact substep이 아니라 `contact 종료 직후 velocity`를 써야 맞았다. 기존 방식은 대다수 contact에서 안정된 post-contact velocity를 직접 보고 있지 않았다.

raw contact vs stabilized post-contact 차이:

- `mean_resolved_minus_contact_velocity_z = -0.0187`
- `mean_abs_resolved_minus_contact_velocity_z = 0.0187`
- `max_abs_resolved_minus_contact_velocity_z = 0.03924`
- `mean_resolved_minus_contact_error_norm = 0.01036`
- `mean_abs_resolved_minus_contact_error_norm = 0.01662`

해석:

- first observed contact substep의 `vz`는 평균적으로 stabilized post-contact `vz`보다 약 `0.0187 m/s` 더 컸다.
- 차이가 매우 크지는 않지만, 거의 모든 contact에서 같은 방향으로 systematic하게 존재했다.
- 따라서 stage-1 결론은 `noise 수준이라 무시 가능`이 아니라 `작지만 구조적으로 잘못된 measurement였다` 쪽이다.

success correlation은 여전히 유지됐다.

- `useful_contact_mean_outgoing_velocity_error_norm = 0.7012`
- `two_or_more_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm = 0.6543`
- `zero_useful_bounce_episode_contact_mean_outgoing_velocity_error_norm = 1.9842`

즉 metric 자체는 여전히 유효하고, 바뀐 것은 metric의 source를 `stable post-contact`로 바로잡았다는 점이다.

## 4. contact normal sign sanity

100-episode summary에서

- `contact_normal_alignment_mean = -0.9962`

로 나온 값은 `contact_mujoco_normal_racket_to_ball`과 `contact_racket_face_normal`을 비교한 결과다. 이 값이 거의 `-1`이라는 것은 현재 `racket_face_normal`이 `racket -> ball`이 아니라 그 반대 방향을 가리킨다는 뜻이다.

하지만 raw MuJoCo contact normal을 그대로 비교하면 정렬은 양수였다.

- `raw_alignment_mean = 0.9962`
- `raw_alignment_min = 0.5841`
- `raw_alignment_max = 1.0`

또 reset 시점에서 controller target face normal과 current racket face normal의 dot은 `~1.0`이었다.

즉 이번 단계의 결론은 아래다.

- controller가 180도 반대로 target을 잡고 있었던 것은 아니다.
- 다만 `raw MuJoCo normal`과 `racket_to_ball`로 재정렬한 normal은 부호가 다르므로, sign을 해석할 때 방향 convention을 명확히 구분해야 한다.
- 이후 feasibility map이나 primitive 설계에서는 `raw contact normal`과 `actual outgoing velocity`를 같이 보고 판단해야 한다.

## 5. conclusion

이번 stage-1의 결론은 명확하다.

1. 기존 metric은 진짜 stable post-contact velocity가 아니라 `first observed contact substep velocity`를 보고 있었다.
2. 수정 후에는 대부분의 contact에서 `contact_end_ball_velocity`를 실제 outgoing source로 쓸 수 있었다.
3. 차이는 작지만 systematic했고, 앞으로는 `stable post-contact velocity`를 기준으로 feasibility를 판단해야 한다.
4. metric 자체의 의미는 유지된다. 낮은 outgoing velocity error가 여전히 성공한 keep-up과 더 강하게 연결된다.
5. 따라서 다음 단계는 reward가 아니라 `scripted feasibility map`과 `contact primitive 여부 판단`이다.