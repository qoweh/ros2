# `position_tilt` chatter fix 및 staged tilt profile 보고서

## 1. 목적

이 문서는 `ppo_position_tilt_limited_500k`에서 확인된 `덜덜거리는 chatter contact` 문제를 최소 수정으로 줄이고, 그 결과를 fresh A/B 실험으로 확인한 기록이다.

이번 턴의 목표는 아래 세 가지였다.

1. chatter를 만드는 보상 누수를 먼저 막는다.
2. tilt freedom을 stage별로 통제 가능한 실험 표면으로 바꾼다.
3. reward shaping을 더 덕지덕지 붙이지 않고, 50-episode 기준으로 결과를 비교한다.

## 2. 적용한 변경

### 2.1 env reward/control 계약 수정

`src/pingpong_rl2/envs/keepup_env.py`에 아래를 추가했다.

- contact 중 `tracking_term`을 지급하지 않도록 `tracking_during_contact_scale` 추가
- `position_tilt`에서만 기본 활성화되는 `tilt_angle_penalty_weight` 추가
- `position_tilt`에서만 기본 활성화되는 `tilt_action_delta_penalty_weight` 추가
- info에 `tilt_magnitude_norm`, `tilt_action_delta_norm` 기록 추가

핵심 의도는 단순하다.

- 공 근처에서 잦은 접촉을 만들며 tracking reward를 파밍하는 경로를 닫는다.
- tilt target을 매 step 크게 흔드는 정책에 즉시 비용을 준다.

### 2.2 학습 스크립트 실험 표면 정리

`scripts/run_ppo_learning.py`에 아래를 추가했다.

- `--tilt-profile auto|custom|early|mid|late|final`
- `--tracking-during-contact-scale`
- `--tilt-angle-penalty-weight`
- `--tilt-action-delta-penalty-weight`
- `resolved_tilt_profile` 출력
- `tilt_limit_ratio` 출력
- 기존 checkpoint를 자동으로 이어받을 때 `resume_note=existing_checkpoint_in_run_dir` 출력

`position_tilt`에서 `--tilt-profile auto`는 `early`로 해석된다.

### 2.3 staged tilt profile 값

현재 추가한 stage 값은 아래다.

| profile | tilt_action_limit | target_tilt_limit | tilt_angle_penalty_weight | tilt_action_delta_penalty_weight |
| --- | ---: | ---: | ---: | ---: |
| early | `0.015` | `(0.06, 0.06)` | `0.06` | `0.12` |
| mid | `0.025` | `(0.09, 0.09)` | `0.05` | `0.10` |
| late | `0.035` | `(0.12, 0.12)` | `0.04` | `0.08` |
| final | `0.04` | `(0.12, 0.12)` | `0.03` | `0.06` |

지금 단계에서 핵심은 `더 큰 tilt`가 아니라 `tilt를 허용해도 chatter로 안 새는가`다.

## 3. 검증 및 실행한 실험

이번 턴에서 실제로 실행한 검증/실험은 아래다.

1. `python -m unittest tests.test_keepup_env -q`
   - `14` tests passed
2. `position_tilt + early` smoke train
   - `ppo_position_tilt_chatterfix_smoke`
3. fresh early profile 200k train
   - `ppo_position_tilt_chatterfix_200k`
4. `ppo_position_tilt_chatterfix_200k` 50-episode evaluation
5. `ppo_position_tilt_chatterfix_200k` 50-episode rebound analysis
6. fresh mid profile 200k train
   - `ppo_position_tilt_chatterfix_mid_200k`
7. `ppo_position_tilt_chatterfix_mid_200k` 50-episode evaluation
8. `ppo_position_tilt_chatterfix_mid_200k` 50-episode rebound analysis

비교 기준으로는 기존 문제 run인 `ppo_position_tilt_limited_500k`의 50-episode rebound analysis 결과를 그대로 사용했다.

## 4. 결과 비교

### 4.1 비교 표

| run | 학습 상태 | tilt 설정 | 50ep mean useful bounces | total contacts | useful contact rate | 해석 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `ppo_position_tilt_limited_500k` | `resume` 오염 | `0.08 / (0.15, 0.15)` | `0.44` | `572` | `3.85%` | contact를 과도하게 만들며 chatter exploit에 빠짐 |
| `ppo_position_tilt_chatterfix_200k` | fresh | `0.015 / (0.06, 0.06)` | `0.34` | `47` | `36.17%` | chatter는 크게 줄었지만 너무 보수적이라 useful bounce 반복이 약함 |
| `ppo_position_tilt_chatterfix_mid_200k` | fresh | `0.025 / (0.09, 0.09)` | `0.06` | `59` | `5.08%` | early보다 성능이 나빠졌고 wasted contact가 다시 늘어남 |

### 4.2 early profile의 의미

`early`는 절대 성능은 아직 낮다. 하지만 구조적으로는 가장 중요한 변화가 확인됐다.

- total contacts: `572 -> 47`
- useful contact rate: `3.85% -> 36.17%`

즉 `많이 부딪히는 정책`이 아니라 `드물게 부딪혀도 그 contact 품질은 상대적으로 낫다`는 방향으로 바뀌었다.

이건 chatter exploit이 실제로 차단됐다는 뜻이다.

### 4.3 mid profile의 의미

`mid`는 기대와 달리 `early`의 단점을 보완하지 못했다.

- mean useful bounces: `0.34 -> 0.06`
- useful contact rate: `36.17% -> 5.08%`
- total contacts: `47 -> 59`

즉 tilt freedom을 조금 더 준 것만으로도 다시 `낮은 품질의 contact` 쪽으로 새기 시작했다.

현재 단계에서 `mid`를 default로 두는 것은 근거가 없다.

## 5. 해석

### 5.1 이번 수정으로 해결된 것

해결된 것은 `tilt branch가 chatter를 보상 파밍 경로로 삼는 문제`다.

이 문제는 사실상 구조적 버그에 가까웠다.

- contact 중에도 tracking reward가 남아 있었고
- tilt magnitude/delta에 비용이 없었고
- 큰 tilt freedom이 early training부터 열려 있었다

이 조합에서 PPO가 `정상 strike`보다 `contact를 자주 만드는 local optimum`을 먼저 찾는 것은 자연스러웠다.

### 5.2 아직 해결되지 않은 것

반대로, 이번 수정만으로 `stable upward keep-up`이 바로 생기지는 않았다.

현재 early 결과는 아래 쪽에 가깝다.

- chatter exploit은 줄었다
- 하지만 아직 `1회 useful bounce`를 안정적으로 반복하는 control skill이 약하다

즉 지금 남은 문제는 `reward leak`가 아니라 `정상 strike를 반복 학습시키는 control/training schedule` 쪽이다.

## 6. 현재 결론

현재 시점에서의 결론은 명확하다.

1. anti-chatter 계약 수정은 유지한다.
2. `position_tilt` 기본 진입점은 `auto -> early`가 맞다.
3. `mid`는 현재 기준으로 채택하지 않는다.
4. 다음 단계는 reward 추가가 아니라 `position baseline` 또는 `early tilt` 기준의 더 깨끗한 학습 스케줄 실험이다.

특히 다음 tilt 실험은 아래 순서를 권장한다.

1. `position`으로 upward strike가 더 안정적인 fresh run 확보
2. 그 다음 `early` profile로만 tilt branch 진입
3. 50-episode 기준으로 useful bounce가 안정화되기 전에는 `mid` 이상으로 넓히지 않기

## 7. 요약

이번 턴의 핵심 결과는 아래 한 줄로 정리된다.

- `chatter는 줄였고, 그래서 원인 분리는 성공했다. 하지만 tilt가 성능 이득을 주는 단계는 아직 아니다.`

따라서 다음 작업은 reward를 더 붙이는 것이 아니라, `early tilt를 유지한 채 더 깨끗한 training schedule/control A/B`로 넘어가는 것이 맞다.