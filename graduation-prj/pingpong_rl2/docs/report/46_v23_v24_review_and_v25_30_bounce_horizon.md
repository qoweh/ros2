# v23/v24 Review And v25 30-Bounce Horizon

## 한줄 결론

`pmk_cf_self_rally_v23`은 지금까지 가장 안정적이다. 하지만 `max_episode_steps=600` 때문에 30 useful bounce 목표를 제대로 학습/평가하기 어렵다. v23을 긴 horizon에서 smoke 평가하니 곧바로 useful `24~26`까지 올라갔다. 따라서 다음 실험은 reward를 크게 흔들기보다 episode horizon과 평가 metric/checkpoint ranking을 30회 목표에 맞추는 `v25`가 맞다.

## v23/v24 비교

최종 rebound analysis 100 episode 기준:

| run | mean useful | max useful | `>=10` | `>=20` | `>=30` | 주요 실패 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v23 | `10.94` | `19` | `73/100` | `0/100` | `0/100` | `time_limit=87`, `low_apex=9`, `ball_out=3` |
| v24 | `7.30` | `20` | `38/100` | `1/100` | `0/100` | `time_limit=41`, `ball_out=25`, `low_apex=24` |

해석:

- v23은 `ball_out_of_bounds`가 거의 잡혔다.
- v24는 scratch 학습이라 v23보다 불안정하다.
- 30회 목표에는 v24 scratch보다 v23 resume 계열이 맞다.

## 현재 병목

1. `max_episode_steps=600`

v23 time-limit episode의 평균 contact 수가 약 `33`이다. useful contact rate가 약 0.3~0.4 수준이면, 600 step 안에서 30 useful를 안정적으로 만들기는 어렵다.

2. checkpoint ranking이 30회를 보지 않음

기존 checkpoint sort key는 `3+ stable/useful`를 주로 봤다. 30회를 목표로 하면 `10+`, `20+`, `30+` rate가 checkpoint 선택 기준에 들어가야 한다.

3. 남은 실패는 긴 horizon에서 low-apex

v25 smoke rebound analysis에서 긴 episode는 useful `24`까지 갔지만 결국 `low_apex_contact`로 끝났다. 즉 다음 병목은 긴 loop 후반의 낮아지는 apex다.

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
  --preset contact_frame_self_rally_v25_long_horizon_30_bounce \
  --run-name tmp_v25_long_horizon_check \
  --run-version codex \
  --resume-from artifacts/ppo_runs/pmk_cf_self_rally_v23/pmk_cf_self_rally_v23_model.zip \
  --total-timesteps 64 \
  --smoke \
  --bootstrap-heuristic-episodes 0 \
  --bootstrap-followup-epochs 0 \
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

## 학습 명령

v25는 v23 final에서 이어서 시작한다.

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

학습에서는 `--preset`이 사실상 필요하다. preset 없이 수십 개 인자를 수동으로 넣으면 실수하기 쉽고, 기본값은 현재 self-rally 설정과 다르다. resume 여부와 별개로 v25처럼 horizon/reward/eval 기준을 바꾸려면 반드시 preset을 명시하는 쪽이 안전하다.
