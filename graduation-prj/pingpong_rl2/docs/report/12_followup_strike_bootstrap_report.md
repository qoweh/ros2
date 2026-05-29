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

## 7. autopilot 후속 cycle: filtered bootstrap ablation

이후 추가로 한 일은 reward를 더 얹는 것이 아니라 bootstrap 데이터 자체를 더 목표지향적으로 고르는 일이었다.

핵심 가설:

- 기존 bootstrap은 `1+ useful bounce`가 나온 heuristic episode 전체를 actor에 복제한다.
- 그래서 rare한 follow-up multi-bounce 예시보다 훨씬 많은 `one-bounce + failed follow-up` 데이터가 actor를 끌고 갈 수 있다.
- 그렇다면 bootstrap을 `post-success` 또는 `true multi-bounce episode` 중심으로 좁히면 second-strike quality가 더 나아질 수 있다.

이를 위해 `run_ppo_learning.py`에 아래 실험용 옵션을 추가했다.

- `--bootstrap-sample-mode`
- `--bootstrap-followup-epochs`
- `--bootstrap-followup-sample-mode`
- `--bootstrap-followup-min-useful-bounces`
- `--bootstrap-followup-learning-rate`

의도는 아래와 같다.

1. base bootstrap은 first-bounce 학습을 위한 전체 episode warm-start를 유지한다.
2. 그 위에 follow-up 전용 bootstrap pass를 한 번 더 얹어 second-strike 쪽으로 actor를 bias한다.

### 7.1 pure `post_success` bootstrap은 실패

첫 cycle은 전체 bootstrap 자체를 `post_success` sample만 쓰도록 바꿨다.

- run: `followup_strike_bootstrap_postsuccess_v1`
- bootstrap dataset: `80` requested, `32` accepted, `2242` samples

checkpoint 결과:

- best checkpoint `5k`
- `mean_useful_bounces=0.2`
- `one+ rate=0.2`
- `two+ rate=0.0`

해석:

- pre-success 전체 trajectory를 버리자 first-bounce 안정성까지 같이 무너졌다.
- 즉 follow-up-only bootstrap으로 base warm-start를 대체하는 방식은 틀렸다.

### 7.2 base + follow-up `post_success_reachable` two-stage도 불충분

둘째 cycle은 base bootstrap은 유지하고, extra follow-up pass만 `post_success_reachable`로 좁혔다.

- run: `followup_strike_bootstrap_twostage_v1`
- base bootstrap: `30` episodes, `2500` samples
- follow-up bootstrap: `8` episodes, `291` samples

checkpoint 결과:

- best checkpoint `10k`
- `mean_useful_bounces=0.3`
- `one+ rate=0.3`
- `two+ rate=0.0`

해석:

- pure `post_success`보다는 낫지만, 기존 bootstrap이나 follow-up contract baseline을 넘지 못했다.
- reachable filter만으로는 true multi-bounce signal을 충분히 살리지 못했다.

### 7.3 `2+ heuristic episode`만 쓰는 follow-up pass는 10-episode 신호는 있었지만 50-episode에서 무너짐

셋째 cycle은 follow-up pass를 아예 `2+ useful bounce`를 만든 heuristic episode만 쓰도록 제한했다.

- run: `followup_strike_bootstrap_twostage_two_plus_v1`
- base bootstrap: `30` episodes, `2500` samples
- follow-up bootstrap: `8` episodes, `656` samples
- follow-up dataset mean useful bounces: `2.0`

10-episode checkpoint 결과는 가장 좋아 보였다.

- best checkpoint `20k`
- `mean_useful_bounces=0.3`
- `max_useful_bounces=2`
- `two+ rate=0.1`

하지만 50-episode rebound analysis에서는 유지되지 않았다.

| model | mean useful bounces | one+ rate | two+ rate | useful contact rate | useful-contact reachable rate | ball out of bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `followup_strike_contract_v1_best` | `0.28` | `0.24` | `0.04` | `10.4%` | `28.6%` | `34/50` |
| `followup_strike_bootstrap_v1_best` | `0.50` | `0.48` | `0.02` | `22.9%` | `12.0%` | `31/50` |
| `followup_strike_bootstrap_twostage_two_plus_v1_best` | `0.20` | `0.20` | `0.00` | `7.1%` | `30.0%` | `41/50` |

해석:

- `two+ heuristic episode`만 쓰는 follow-up pass는 짧은 eval에서는 promising해 보일 수 있다.
- 하지만 50-episode 기준으로는 first-bounce 안정성도, second-strike 안정성도 유지하지 못했다.
- 즉 이 branch는 아직 promote하면 안 된다.

## 8. 갱신된 결론

autopilot 후속 cycle까지 포함한 현재 결론은 아래다.

- centered upward useful second strike를 여는 control-side 핵심은 여전히 `followup_strike_candidate`다.
- training-side bootstrap은 여전히 의미가 있다. 특히 plain bootstrap은 first-bounce acquisition을 빠르게 만든다.
- 하지만 filtered bootstrap 계열(`post_success`, `post_success_reachable`, `two_plus follow-up pass`)은 아직 50-episode 기준으로 승격할 수준이 아니다.
- 대신 `plain bootstrap best checkpoint`에서 시작해 active follow-up contract 아래서 PPO를 이어학습하는 staged schedule은 실제로 더 좋은 절충점을 만들었다.

현재 유지할 기준:

- control-side 핵심 contract: `followup_strike_candidate`
- training-side first-bounce acquisition reference: `followup_strike_bootstrap_v1_best`
- 새 staged training reference: `followup_bootstrap_resume_contract_v1_best`

즉 이번 후속 작업은 두 갈래로 정리된다.

- filtered bootstrap 계열은 빠르게 제거했다.
- 반면 `bootstrap best checkpoint -> active follow-up contract resume`라는 staged schedule은 유지할 가치가 있는 새 방향으로 확인됐다.

## 9. staged bootstrap-resume 결과

filtered bootstrap이 기대만큼 먹히지 않았기 때문에, 더 직접적인 training schedule을 확인했다.

실험:

- 시작점: `followup_strike_bootstrap_v1_best_model.zip`
- 이어학습 run: `followup_bootstrap_resume_contract_v1`
- env/control: 기존 active `followup_strike_candidate`
- 추가 학습량: `20k`

핵심 아이디어는 간단하다.

- plain bootstrap이 이미 만든 `first useful bounce를 자주 만드는 policy`를 버리지 않는다.
- 그 checkpoint를 초기 상태로 삼고, 같은 follow-up contract 아래에서 PPO를 더 진행해 second-strike를 학습시킨다.

### 9.1 checkpoint eval

best checkpoint는 `10k` 추가 학습 지점에서 나왔다.

| model | checkpoint | mean useful bounces | one+ rate | two+ rate | max useful bounces |
| --- | ---: | ---: | ---: | ---: | ---: |
| `followup_strike_contract_v1` | `20k` | `0.3` | `0.2` | `0.1` | `2` |
| `followup_strike_bootstrap_v1` | `5k` | `0.7` | `0.6` | `0.1` | `2` |
| `followup_bootstrap_resume_contract_v1` | `10k` | `0.6` | `0.4` | `0.2` | `2` |

이 checkpoint 숫자는 지금까지 본 것 중 가장 좋다.

### 9.2 50-episode rebound 비교

50-episode 기준으로도 이 run은 유의미한 절충점으로 남았다.

| model | mean useful bounces | one+ rate | two+ rate | useful contact rate | useful-contact reachable rate | mean easy-next-ball score | ball out of bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `followup_strike_contract_v1_best` | `0.28` | `0.24` | `0.04` | `10.4%` | `28.6%` | `0.136` | `34/50` |
| `followup_strike_bootstrap_v1_best` | `0.50` | `0.48` | `0.02` | `22.9%` | `12.0%` | `0.057` | `31/50` |
| `followup_bootstrap_resume_contract_v1_best` | `0.30` | `0.26` | `0.04` | `11.0%` | `33.3%` | `0.146` | `40/50` |

해석:

- `two+ rate`는 `followup_strike_contract_v1_best`와 동률이다.
- `mean useful bounces`, `one+ rate`, `useful contact rate`, `useful-contact reachable rate`, `mean easy-next-ball score`는 모두 contract-only run보다 좋다.
- plain bootstrap보다는 first-bounce volume이 낮지만, second-strike quality는 훨씬 낫다.

즉 이 run은 두 기존 candidate의 장점을 완전히 합치진 못했지만, 현재까지는 가장 설득력 있는 균형점이다.

### 9.3 현재 해석

지금 기준에서 가장 목표지향적인 training direction은 아래다.

1. control은 `followup_strike_candidate`를 유지한다.
2. plain bootstrap으로 first-bounce acquisition이 강한 checkpoint를 만든다.
3. 그 best checkpoint에서 같은 contract 아래 PPO를 더 진행한다.
4. checkpoint selection은 계속 `two+ useful bounce` 우선으로 한다.

이건 reward 미세조정이 아니라, 이미 확인된 두 장점 `first-bounce acquisition`과 `second-strike contract`를 같은 학습 경로에 이어붙이는 방식이다. 현재로서는 이 staged schedule이 가장 목표에 가깝다.