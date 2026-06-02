# 31. v6 저높이 반복 접촉 병목과 strict cycle 보강

작성일: 2026-06-02

## v6 관찰

`pmk_cf_self_rally_v6`는 약 4M timesteps까지 학습됐지만 useful bounce가 낮았다.

사용자 viewer 결과:

- contacts는 계속 발생
- useful_bounces는 대체로 `0~1`
- 긴 episode도 낮은 접촉을 반복하다가 `ball_out_of_bounds` 또는 `ball_speed_limit`로 끝남

학습 로그에서 `training_mode=resume`으로 표시됐다. 즉 v6는 같은 run directory의 기존 checkpoint에서 이어 학습됐다. 앞으로 비교 실험은 같은 run version을 재사용하지 말고 새 version과 `--reset-model`을 같이 써야 한다.

## v6 contact 분석

100 episode rebound 분석:

- total contacts: `1075`
- useful contacts: `46`
- useful contact rate: `0.043`
- mean useful bounces: `0.46`
- max useful bounces: `4`
- failure:
  - `ball_out_of_bounds = 77`
  - `floor_contact = 12`
  - `ball_speed_limit = 8`
  - `time_limit = 3`

핵심 contact 통계:

- mean projected apex height: `0.199m`
- median projected apex height: `0.165m`
- useful projected apex height: `0.347m`
- mean actual outgoing z: `1.56`
- useful actual outgoing z: `2.40`
- desired outgoing z: `2.24`
- next intercept reachable rate: `0.345`
- useful next intercept reachable rate: `1.0`

조건별 통과율:

- upward velocity 통과: `0.940`
- racket z velocity 통과: `0.967`
- contact xy 통과: `0.775`
- apex `>= 0.30m` 통과: `0.213`
- next intercept reachable 통과: `0.345`

첫 실패 조건 기준으로는 `apex >= 0.30m` 미달이 `590 / 1075`로 가장 많았다.

## 결론

v6의 핵심 문제는 reward weight가 조금 부족한 정도가 아니다.

정책이 "낮게라도 계속 맞히는" rollout을 오래 유지할 수 있고, PPO가 그 상태를 많이 보게 된다. 이러면 `contact`는 늘지만 self-rally 목표인 "적절한 높이로 띄워 다음에 다시 치기 쉬운 공 만들기"는 강화되지 않는다.

따라서 v7은 reward 숫자를 더 키우기보다 task 구조를 바꾼다.

## 구현 변경

### 1. 낮은 apex 반복 접촉 종료

새 env 옵션:

- `terminate_on_low_apex_contact`
- `low_apex_contact_height_threshold`
- `low_apex_contact_grace_count`

동작:

- upward contact가 발생했지만 projected apex가 threshold보다 낮으면 low-apex contact로 본다.
- self-rally preset에서는 1회 grace를 둔다.
- 같은 episode에서 low-apex contact가 반복되면 `failure_reason = low_apex_contact`로 종료한다.

self-rally preset:

- `terminate_on_low_apex_contact = True`
- `low_apex_contact_height_threshold = 0.20`
- `low_apex_contact_grace_count = 1`

이 기준은 `target_ball_height=0.30`, `height_tolerance=0.10`에서 `target - tolerance`에 해당한다. 즉 0.20m보다 낮게 끝나는 upward hit는 self-rally cycle로 인정하지 않는다.

전체 `terminate_on_nonuseful_contact=True`는 쓰지 않았다. lateral/next-intercept만 살짝 부족한 exploratory hit까지 모두 죽이면 학습이 너무 sparse해질 수 있기 때문이다.

### 2. reset 시작 높이를 목표 cycle에 맞춤

v6 preset은 `ball_height=0.50`으로 시작했다. 하지만 목표 post-contact apex는 `0.30m`이기 때문에, 시작 분포와 목표 반복 cycle이 다소 달랐다.

self-rally preset 변경:

- `ball_height = 0.34`
- `reset_ball_height_range = 0.02`
- `target_ball_height = 0.30` 유지

이제 초기 공은 목표 apex 근처에서 시작한다. 너무 낮은 시작은 아니고, 0.30m target cycle을 직접 연습하는 분포다.

### 3. reset height CLI/preset 통로 추가

`run_ppo_learning.py`에 `--reset-ball-height-range`를 추가했다. preset에서도 이 값을 관리한다.

## 다음 학습

v6와 섞지 말고 v7으로 새로 학습한다.

```bash
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v7 \
  --reset-model \
  --total-timesteps 2000000
```

중요:

- `--run-version v6`를 다시 쓰면 기존 v6 checkpoint에서 resume될 수 있다.
- 같은 version을 다시 쓸 때도 반드시 `--reset-model`이 필요하다.
- 비교를 명확히 하려면 그냥 새 version을 쓰는 것이 낫다.

## 평가 기준

v7 평가에서 볼 것:

- `failure_reason=low_apex_contact`가 초반에 많이 나오는 것은 정상이다.
- 학습이 되면 `low_apex_contact` 비율이 줄고, useful contact rate가 올라가야 한다.
- mean projected apex height가 `0.199m`에서 최소 `0.25m+`로 올라가야 한다.
- useful projected apex는 `0.30~0.40m` window에 남아야 한다.
- next intercept reachable rate가 `0.35`보다 올라가야 한다.
- `ball_speed_limit`이 크게 늘면 vertical primitive가 과해진 것이다.

정량 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v7/pmk_cf_self_rally_v7_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v7_contact_diagnosis
```

## 검증

로컬 검증:

- `python -m py_compile src/pingpong_rl2/envs/keepup_env.py scripts/run_ppo_learning.py`
- `python -m unittest tests/test_keepup_env.py tests/test_keepup_contract_features.py tests/test_ppo_runs.py tests/test_vector_env.py`

결과:

- 100 tests 통과

Smoke:

- `total_timesteps = 0`
- `mean_useful_bounces = 0.100`
- `max_useful_bounces = 2`

Short PPO smoke:

- `total_timesteps = 2048`
- `mean_useful_bounces = 0.150`
- `max_useful_bounces = 2`

이 smoke 수치는 최종 성능이 아니라 새 termination/reset 구조에서 학습 루프가 깨지지 않는지 확인한 결과다.
