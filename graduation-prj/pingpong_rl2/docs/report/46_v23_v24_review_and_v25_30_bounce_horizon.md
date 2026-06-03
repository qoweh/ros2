# v23/v24 Review And v25 30-Bounce Horizon

## 한줄 결론

`pmk_cf_self_rally_v25`는 현재까지 가장 발표 가능한 모델이다. 100 episode rebound analysis 기준 mean useful bounce `28.51`, max `51`, `30+ useful bounce rate=0.61`까지 올라왔다. 방금 성능이 크게 오른 핵심 이유는 새 reward를 많이 뒤집은 것이 아니라, v23의 안정적인 정책을 유지한 채 `max_episode_steps=600 -> 1800`으로 늘리고 checkpoint/evaluation 기준을 `30회 이상 지속`에 맞췄기 때문이다.

## v23/v24/v25 비교

최종 rebound analysis 100 episode 기준:

| run | mean useful | max useful | `>=10` | `>=20` | `>=30` | 주요 실패 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v23 | `10.94` | `19` | `73/100` | `0/100` | `0/100` | `time_limit=87`, `low_apex=9`, `ball_out=3` |
| v24 | `7.30` | `20` | `38/100` | `1/100` | `0/100` | `time_limit=41`, `ball_out=25`, `low_apex=24` |
| v25 | `28.51` | `51` | `84/100` | `72/100` | `61/100` | `time_limit=62`, `low_apex=27`, `ball_out=7` |

해석:

- v23은 `ball_out_of_bounds`가 거의 잡혔다.
- v24는 scratch 학습이라 v23보다 불안정하다.
- v25는 v23 resume 계열에 긴 horizon과 30회 지표를 붙이면서 실제 목표 지표가 크게 올랐다.

## 방금 성능이 오른 이유

1. horizon 병목 제거

v23은 `max_episode_steps=600`이었다. viewer에서 계속 치는 것처럼 보여도 episode가 600 step에서 끊기므로 `30 useful bounce`가 안정적으로 기록되기 어려웠다. v25는 `max_episode_steps=1800`으로 늘려 정책이 이미 갖고 있던 반복 능력을 더 오래 드러낼 수 있게 했다.

2. 목표에 맞는 checkpoint 선택

기존 checkpoint ranking은 `2~3회 이상 useful/stable`에 더 민감했다. v25는 `10+`, `20+`, `30+` useful/stable rate를 평가와 best checkpoint 기준에 넣었다. 그 결과 `200k` checkpoint가 `30+ rate=0.7667`로 best에 잡혔다.

3. v23의 안정성을 버리지 않음

v24처럼 scratch로 다시 시작하지 않고, `ball_out_of_bounds`를 많이 줄인 v23 final checkpoint에서 이어서 학습했다. 즉 v25의 상승은 "새로 처음부터 배운 성능"보다 "좋은 정책을 더 긴 과제와 맞는 선택 기준으로 다듬은 성능"에 가깝다.

4. reward cap을 긴 episode에 맞춤

`stable_cycle_reward_cap=12`로 늘려 긴 안정 루프에 보상이 계속 의미 있게 남도록 했다. 기존 cap은 짧은 성공만 구분하고 긴 유지 능력에는 둔감했다.

## v25 세부 결과

`pmk_cf_self_rally_v25_final_contact_diagnosis` 100 episode:

- mean useful bounces: `28.51`
- max useful bounces: `51`
- `>=10`: `84/100`
- `>=20`: `72/100`
- `>=30`: `61/100`
- time-limit episode: `62/100`, mean useful `37.37`, max `51`
- low-apex episode: `27/100`, mean useful `14.15`, max `39`
- ball-out episode: `7/100`, mean useful `16.43`, max `28`

학습 중 checkpoint 평가:

| timesteps | mean useful | max useful | `>=10` | `>=20` | `>=30` | 실패 요약 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0` | `20.43` | `43` | `0.7667` | `0.4667` | `0.3000` | `low_apex=18`, `ball_out=4`, `time_limit=7` |
| `100k` | `28.97` | `49` | `0.8667` | `0.7000` | `0.6333` | `time_limit=19`, `low_apex=9`, `ball_out=2` |
| `200k` | `30.20` | `49` | `0.8667` | `0.8000` | `0.7667` | `time_limit=20`, `ball_out=6`, `low_apex=4` |
| `300k` | `29.97` | `50` | `0.8667` | `0.7333` | `0.6000` | `time_limit=18`, `low_apex=5`, misc=3 |
| `400k` | `31.17` | `48` | `0.9000` | `0.7333` | `0.7000` | `time_limit=22`, `ball_out=6`, `low_apex=2` |
| `500k` | `26.53` | `48` | `0.8333` | `0.6000` | `0.5000` | `low_apex=13`, `time_limit=14`, `ball_out=3` |

현재 시연 후보는 final model도 괜찮지만, best checkpoint인 `pmk_cf_self_rally_v25_best_model.zip`을 우선 확인하는 것이 좋다.

## 현재 병목과 추가 개선 후보

1. `low_apex_contact`의 초중반 실패

v25 low-apex episode는 `27/100`이고 mean useful `14.15`다. 일부는 useful `30` 이상까지 가고 나서 낮아지지만, `10` 미만으로 끝나는 episode도 `11개`다. 남은 개선은 높이 reward를 더 세게 넣는 것보다, 낮아지는 loop를 초기에 감지하고 z/timing residual이 회복 동작을 하게 만드는 쪽이 맞다.

2. 긴 episode 말기의 lateral drift

`ball_out_of_bounds`는 `7/100`까지 줄었지만, 그중 4개는 useful `25`회 이상 버틴 뒤 발생했다. 즉 초반 정책 실패라기보다 긴 루프 후반의 누적 drift 성격이 강하다. 현재는 발표 전 대규모 구조 변경보다 best checkpoint 선택과 viewer 시연 seed 선별이 더 현실적이다.

3. final보다 best checkpoint가 더 좋을 가능성

500k final은 100-episode analysis에서 충분히 좋지만, checkpoint evaluation에서는 200k 또는 400k가 더 강했다. 발표 시연은 `pmk_cf_self_rally_v25_model.zip`과 `pmk_cf_self_rally_v25_best_model.zip`을 viewer로 비교해서 더 안정적인 쪽을 쓰는 것이 좋다.

4. 시작 위치 일반화는 아직 별도 과제

현재 reset은 `reset_xy_range=0.028`, `reset_ball_height_range=0.02`로 비교적 좁다. 임의 xy/높이 일반화는 가능하지만, 지금 발표 전에는 성능을 크게 흔들 수 있다. 다음 단계에서는 target-conditioned observation/curriculum으로 따로 확장하는 것이 맞다.

## 구현 변경

- `contact_frame_self_rally_v25_long_horizon_30_bounce` preset 추가
  - v23 기반
  - `max_episode_steps=1800`
  - `stable_cycle_reward_cap=12`
  - `checkpoint_eval_episodes=30`
  - `eval_episodes=80`
- `run_ppo_learning.py` 평가 metric 추가
  - `ten_or_more_useful_bounce_rate`
  - `twenty_or_more_useful_bounce_rate`
  - `thirty_or_more_useful_bounce_rate`
  - stable cycle도 동일
- checkpoint sort key에 30/20/10회 rate 반영
- `run_ppo_rebound_analysis.py`, `run_ppo_evaluation.py`에도 10/20/30회 지표 추가
- preset override 목록에 `max_episode_steps` 허용
- `run_ppo_learning.py`에 `--config-file` 추가
  - 긴 CLI 대신 JSON 설정파일에서 실행 인자를 읽는다.
  - CLI로 직접 넣은 값은 config file보다 우선한다.
  - preset 자체의 env 값은 계속 코드 preset에 고정해 실험 재현성을 유지한다.
- `run_ppo_learning.py`에 `--set KEY=VALUE` 추가
  - 드문 일회성 override는 개별 CLI 인자를 계속 늘리지 않고 generic override로 처리한다.
  - `--set`은 preset 적용 후 실행되므로 preset 값을 확실히 덮어쓴다.
- `configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json` 추가
  - v25 재현용 실행 설정
  - 새 실험에 그대로 쓰면 v25 run directory를 덮을 수 있으므로 `run_version`을 바꿔서 사용한다.

## 검증

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  scripts/run_ppo_learning.py \
  scripts/run_ppo_rebound_analysis.py \
  scripts/run_ppo_evaluation.py
```

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json \
  --run-version config_set_summary_smoke \
  --output-dir artifacts/tmp/tmp_v25_config_set_summary_check \
  --total-timesteps 64 \
  --smoke \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0
```

결과:

- config file load 정상
- resolved preset: `contact_frame_self_rally_v25_long_horizon_30_bounce`
- resolved run: `pmk_cf_self_rally_config_set_summary_smoke`
- `max_episode_steps=1800`
- smoke eval mean useful `24.0`
- smoke eval max useful `26`

이전 smoke 확인:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v25_long_horizon_30_bounce \
  --run-name tmp_v25_long_horizon_check \
  --run-version codex \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v23/pmk_cf_self_rally_v23_model.zip \
  --total-timesteps 64 \
  --smoke \
  --set bootstrap_heuristic_episodes=0 \
  --set bootstrap_epochs=0 \
  --set bootstrap_followup_epochs=0 \
  --output-dir artifacts/tmp/tmp_v25_long_horizon_check_codex
```

결과:

- `max_episode_steps=1800`
- smoke eval mean useful `24.0`
- max useful `26`
- `twenty_or_more_useful_bounce_rate=1.0`
- `thirty_or_more_useful_bounce_rate=0.0`

통과:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/tmp/tmp_v25_long_horizon_check_codex/tmp_v25_long_horizon_check_codex_model.zip \
  --episodes 2 \
  --seed 252 \
  --output-dir artifacts/tmp/tmp_v25_long_horizon_check_codex/analysis \
  --analysis-name tmp_v25_long_horizon_analysis_check
```

결과:

- episode 1: useful `24`, failure `low_apex_contact`
- episode 2: useful `6`, failure `low_apex_contact`
- 새 10/20/30회 지표가 summary에 기록됨

## v25 재현 명령

v25는 이미 완료됐다. 같은 실험을 재현할 때는 아래 설정파일 경로가 더 짧다. 단, 그대로 실행하면 `pmk_cf_self_rally_v25` run directory에 다시 쓸 수 있으므로 새 실험은 `run_version`을 `v26`처럼 바꾼다.

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v25_long_horizon_30_bounce.json
```

같은 의미의 긴 CLI:

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_v25_long_horizon_30_bounce \
  --run-name pmk_cf_self_rally \
  --run-version v25 \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v23/pmk_cf_self_rally_v23_model.zip \
  --total-timesteps 500000
```

학습 후 분석:

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name pmk_cf_self_rally \
  --run-version v25 \
  --episodes 100 \
  --seed 251 \
  --analysis-name pmk_cf_self_rally_v25_final_contact_diagnosis
```

확인:

```bash
jq '{mean_useful_bounces,max_useful_bounces,ten_or_more_useful_bounce_rate,twenty_or_more_useful_bounce_rate,thirty_or_more_useful_bounce_rate,failure_counts}' \
  artifacts/ppo_runs/pmk_cf_self_rally_v25/analysis/pmk_cf_self_rally_v25_final_contact_diagnosis_summary.json
```

## preset 사용 여부

학습에서는 `--preset` 또는 `--config-file`이 사실상 필요하다. preset 없이 수십 개 인자를 수동으로 넣으면 실수하기 쉽고, 기본값은 현재 self-rally 설정과 다르다. `--config-file`은 실행 인자를 줄이는 용도이고, 실제 환경/reward/action 설정은 여전히 preset이 담당한다. 드문 값 변경은 `--set KEY=VALUE`로 남긴다.
