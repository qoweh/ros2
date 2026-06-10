# v30 검토와 짧은 모델 이름 정리

작성일: 2026-06-04

## 1. v30 결론

`v30`은 성공적인 안정 확장으로 본다.

기존 v26 안정 모델을 그대로 이어받고, action mode는 바꾸지 않은 채 reset XY만 `0.075m -> 0.10m`로 넓혔다.

100 episode rebound analysis 기준:

| 모델 | reset XY | mean useful | max useful | 30+ rate | time_limit |
|---|---:|---:|---:|---:|---:|
| v26 | `0.075m` | `42.34` | `85` | `60%` | `35/100` |
| v29 | `0.16m + velocity + spin` | `6.31` | `33` | `2%` | `6/100` |
| v30 | `0.10m` | `30.56` | `53` | `71%` | `70/100` |

학습 summary 기준으로도 v30은 좋았다.

- mean useful: `59.19`
- max useful: `88`
- 30+ rate: `78.75%`
- failure counts: `time_limit=49`, `ball_out_of_bounds=16`, `low_apex_contact=12`

해석:

- v30은 v29보다 훨씬 안정적이다.
- v30은 v26보다 시작 XY 범위가 넓지만, 30+ 성공률은 유지되거나 개선됐다.
- 분석 기준에서 mean useful가 summary보다 낮은 이유는 seed/evaluation set 차이와 1800 step safety cap 때문이다.
- `time_limit`이 많다는 것은 대부분의 episode가 안전 cap까지 살아남았다는 뜻이라, 발표/웹서비스 기본 모델로 쓰기에 적합하다.

## 2. contact 품질

v30 100 episode contact 분석:

- total contacts: `8352`
- useful contact rate: `36.6%`
- next-intercept reachable rate: `91.4%`
- useful-contact next-intercept reachable rate: `99.9%`
- mean ball lateral speed: `0.051m/s`
- mean projected apex XY error: `0.018m`
- useful projected apex XY error: `0.015m`
- mean projected apex height above racket: `0.252m`
- useful projected apex height above racket: `0.285m`

v29에서 문제였던 lateral drift가 크게 줄었다.

- v29 mean ball lateral speed: `0.124m/s`
- v30 mean ball lateral speed: `0.051m/s`

따라서 현재 발표/웹서비스용 1공 모델은 `v30`을 기준으로 잡는다.

## 3. 모델 디렉터리 정리

기존 긴 모델 이름들은 모두 아래로 이동했다.

```text
artifacts/ppo_runs/_legacy_models/
```

현재 root에 남긴 active 모델:

```text
artifacts/ppo_runs/keep1_v26/
artifacts/ppo_runs/keep1_v30/
```

짧은 모델 경로:

```text
artifacts/ppo_runs/keep1_v30/keep1_v30_model.zip
artifacts/ppo_runs/keep1_v26/keep1_v26_model.zip
```

`keep1`은 "1-ball keep-up"이라는 뜻으로 쓴다.

권장 네이밍:

- `keep1_v30`: 현재 발표/웹서비스 기본 1공 모델
- `keep1_v26`: 안정 baseline 백업
- `keep2_v1`: 나중에 2공 모델을 만들 때 사용할 이름

## 4. 실행 명령

viewer:

```bash
cd mujoco/pingpong_rl2
conda activate mujoco_env
mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/keep1_v30/keep1_v30_model.zip \
  --episodes 100 \
  --max-episode-steps 1800
```

분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v30/keep1_v30_model.zip \
  --episodes 100 \
  --seed 231 \
  --episode-step-limit 1800 \
  --analysis-name keep1_v30_eval100
```

추가 학습:

```bash
python scripts/run_ppo_learning.py --config-file configs/keep1_v30.json
```

