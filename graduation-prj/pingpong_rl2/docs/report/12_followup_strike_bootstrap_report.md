# follow-up strike PPO와 bootstrap 보고서

## 1. 배경

이 단계의 핵심 문제는 아래였다.

- heuristic은 이제 가끔 `2+ useful bounce`를 만든다.
- 하지만 PPO는 여전히 `다시 맞히기는 하는데 centered upward useful second strike로 이어지지 않는` 구간에서 흔들린다.

그래서 이번 단계에서는 두 가지를 분리해서 봤다.

1. control-side follow-up strike contract 자체가 PPO에서도 second strike를 여는가
2. heuristic에서 확인된 구조를 PPO가 더 빨리 배우도록 bootstrap warm-start를 붙일 가치가 있는가

## 2. 수행 내용

### 2.1 follow-up strike PPO 50-episode 분석

아래 run을 50-episode rebound analysis로 다시 봤다.

- `followup_strike_contract_v1_best_model.zip`
- 비교 기준 `clean_tnp_return_assist_v1_best_model.zip`

핵심 수치:

| model | mean useful bounces | one+ rate | two+ rate | useful contact rate | useful-contact reachable rate | useful-contact easy-next-ball score | ball out of bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_tnp_return_assist_v1_best` | `0.40` | `0.38` | `0.02` | `16.3%` | `25.0%` | `-0.267` | `38/50` |
| `followup_strike_contract_v1_best` | `0.28` | `0.24` | `0.04` | `10.4%` | `28.6%` | `-0.340` | `34/50` |

해석:

- 기존 active candidate는 first useful bounce 안정성이 더 좋다.
- follow-up strike contract run은 `two+ rate`와 useful-contact reachable rate가 더 좋다.
- 즉, second strike 방향성 자체는 follow-up contract가 더 직접적으로 건드리고 있다.
- 반대로 이 run은 첫 useful bounce를 만드는 빈도는 낮아졌다.

따라서 `second strike를 여는 구조`와 `그 구조를 안정적으로 학습하는 경로`를 따로 다뤄야 한다는 결론이 강화됐다.

### 2.2 heuristic bootstrap warm-start 구현

`scripts/run_ppo_learning.py`에 아래 bootstrap 경로를 추가했다.

- `--bootstrap-heuristic-episodes`
- `--bootstrap-min-useful-bounces`
- `--bootstrap-max-samples`
- `--bootstrap-epochs`
- `--bootstrap-batch-size`
- `--bootstrap-learning-rate`

동작 방식:

1. 현재 env 설정으로 heuristic rollout episode를 수집한다.
2. `successful_bounce_count >= bootstrap_min_useful_bounces`인 episode만 남긴다.
3. 남은 observation/action pair로 actor를 MSE supervised pretrain한다.
4. 그 다음 동일한 PPO training loop와 checkpoint selection을 수행한다.

이 구현은 heuristic action을 PPO reward 안에 억지로 섞지 않고, actor 초기화에만 제한적으로 사용한다.

### 2.3 bootstrap smoke 검증

짧은 smoke run에서 bootstrap 경로가 끝까지 정상 동작하는 것을 확인했다.

- accepted episodes: `4`
- accepted samples: `324`
- mean bootstrap loss: `1.33e-4`

즉, rollout 수집, actor pretrain, PPO learn, checkpoint 저장까지 모두 깨지지 않았다.

## 3. bootstrap PPO 결과

실제 short run:

- run: `followup_strike_bootstrap_v1`
- preset: `followup_strike_candidate`
- bootstrap dataset: `40` requested episodes, `18` accepted episodes, `1572` samples
- bootstrap mean episode useful bounces: `1.17`

checkpoint eval 결과:

| model | checkpoint | mean useful bounces | one+ rate | two+ rate |
| --- | ---: | ---: | ---: | ---: |
| `followup_strike_contract_v1` | `5k` | `0.4` | `0.4` | `0.0` |
| `followup_strike_bootstrap_v1` | `5k` | `0.7` | `0.6` | `0.1` |

이 시점에서는 bootstrap 효과가 분명했다. PPO가 useful bounce를 훨씬 빨리 배웠다.

## 4. 50-episode rebound 비교

best checkpoint 50-episode 분석 결과는 아래였다.

| model | mean useful bounces | one+ rate | two+ rate | useful contact rate | useful-contact reachable rate | useful-contact easy-next-ball score | ball out of bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean_tnp_return_assist_v1_best` | `0.40` | `0.38` | `0.02` | `16.3%` | `25.0%` | `-0.267` | `38/50` |
| `followup_strike_contract_v1_best` | `0.28` | `0.24` | `0.04` | `10.4%` | `28.6%` | `-0.340` | `34/50` |
| `followup_strike_bootstrap_v1_best` | `0.50` | `0.48` | `0.02` | `22.9%` | `12.0%` | `-0.503` | `31/50` |

해석:

- bootstrap run은 first useful bounce 안정화에서는 가장 좋다.
- `mean useful bounces`, `one+ rate`, `useful contact rate`, `ball_out_of_bounds`는 모두 bootstrap이 가장 좋다.
- 하지만 centered upward useful second strike 관점의 `two+ rate`는 baseline active candidate와 동일하고, useful-contact next-intercept quality는 오히려 나빠졌다.
- 반대로 no-bootstrap follow-up run은 `two+ rate`와 useful-contact reachable rate가 가장 좋다.

즉, bootstrap은 `첫 useful strike를 배우는 문제`에는 효과적이지만, `다음 공을 centered/upward/useful하게 만드는 문제`를 최종적으로 해결한 것은 아니다.

## 5. 현재 결론

이번 단계에서 확인된 사실은 아래와 같다.

- centered upward useful second strike를 여는 데 필요한 핵심 구조 변경은 follow-up strike contract다.
- heuristic bootstrap은 PPO가 useful bounce를 더 빨리 배우게 하는 유효한 training-side 도구다.
- 하지만 50-episode 기준 second-strike quality를 가장 직접적으로 보여 주는 run은 아직 `followup_strike_contract_v1` best checkpoint다.
- 반대로 first-bounce stability와 전체 useful-contact volume은 `followup_strike_bootstrap_v1`가 가장 좋다.

따라서 현재 active 해석은 아래처럼 나뉜다.

- control-side active direction: `followup_strike_candidate`
- training-side auxiliary tool: heuristic bootstrap warm-start

## 6. 다음 우선순위

다음 단계는 단순 스칼라 튜닝보다 아래 구조가 우선이다.

1. bootstrap best checkpoint를 시작점으로 하되, second-strike quality가 무너지지 않도록 더 보수적인 후속 학습 스케줄을 붙인다.
2. 또는 bootstrap dataset을 `1+ useful bounce episode 전체`가 아니라 follow-up strike quality가 좋은 segment 중심으로 더 엄격하게 고른다.
3. checkpoint selection은 계속 `two+ useful bounce` 우선 기준을 유지한다.

요약하면, 이번 단계에서 `2+`를 여는 control contract와 PPO bootstrap 경로는 둘 다 확보했다. 아직 둘이 동시에 strongest가 되지는 않았지만, 이제는 무엇이 control bottleneck이고 무엇이 training bottleneck인지 분리해서 다룰 수 있는 상태가 되었다.