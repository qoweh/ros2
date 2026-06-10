# v13 fast-episode / low-apex recovery fix

작성일: 2026-06-02

## 목표

최종 목표는 공을 단순히 맞히는 것이 아니라, 탁구공을 일정한 높이와 방향으로 계속 위로 올려치는 self-rally다. `pmk_cf_self_rally_v13`은 방향과 다음 intercept 품질은 좋아졌지만, viewer에서 보인 것처럼 episode가 너무 빨리 끝났고 대부분 `low_apex_contact`로 종료됐다. 이번 작업은 이 병목을 "낮은 공을 바로 실패로 자르는 문제"와 "낮은 공 이후 회복 행동을 배울 신호가 부족한 문제"로 보고 수정했다.

## v13 진단

`artifacts/ppo_runs/pmk_cf_self_rally_v13/analysis/pmk_cf_self_rally_v13_contact_diagnosis_summary.json` 기준:

- 100 episode 전부 `low_apex_contact`로 종료됐다.
- `mean_useful_bounces = 0.31`, `max_useful_bounces = 3`
- `one_or_more_useful_bounce_rate = 0.22`, `two_or_more_useful_bounce_rate = 0.08`
- 전체 contact `430`, useful contact rate `0.0721`
- mean projected apex `0.200m`, median `0.169m`
- upward contact 중 apex `< 0.20m` 비율 `0.597`
- upward contact 중 target apex `< 0.30m` 미달 비율 `0.809`
- mean next intercept xy error `0.0326m`
- next intercept reachable rate `0.770`
- mean easy-next-ball score `0.779`

v12와 episode 길이 비교:

- v12: mean steps `161.91`, median steps `115`, mean contacts `9.19`, mean useful `0.40`
- v13: mean steps `70.39`, median steps `53`, mean contacts `4.30`, mean useful `0.31`

해석:

- v13은 lateral drift와 다음 intercept 가능성이 v12보다 좋아졌다. 즉 방향성/받을 위치 쪽은 일부 개선됐다.
- 하지만 `low_apex_contact_height_threshold=0.20`, `low_apex_contact_grace_count=1`이 너무 강해서, 낮은 apex에서 회복을 시도할 episode 길이가 사라졌다.
- viewer에서 "더 이상한 방향, 너무 빨리 끝남"처럼 보인 것은, 방향을 잡는 동안 높이 회복 curriculum이 너무 일찍 잘린 결과로 보는 게 맞다.

## 외부 지식 검토

이번에는 새 네트워크 구조를 바로 도입하지 않고, 현재 contact-frame primitive를 강화했다. 이유는 관련 연구의 공통 방향이 "명시적인 타격 목표/skill/controller 위에 학습 정책을 얹는 구조"에 가깝기 때문이다.

- Google DeepMind의 competitive robot table tennis 연구는 로봇 탁구에서 skill/controller, task distribution, sim-to-real gap을 함께 다루는 계층적 접근을 사용한다. 현재 코드의 contact-frame planner + base strike + residual PPO 구조와 방향이 맞다.  
  Source: https://deepmind.google/research/publications/achieving-human-level-competitive-robot-table-tennis/
- Google Research의 model-free robot table tennis 연구는 reward와 curriculum tuning이 return rate에 중요하다고 보고한다. 그래서 단순히 2M을 더 학습시키기보다 low-apex curriculum과 recovery reward를 바꿨다.  
  Source: https://research.google/pubs/robotic-table-tennis-with-model-free-reinforcement-learning/
- Ploeger et al.의 real-world juggling 연구는 동적 juggling에서 policy representation, initialization, optimization을 같이 설계해야 한다는 점을 보여준다. 현재 단계에서는 multi-ball보다 1-ball recovery primitive와 bootstrap 품질을 먼저 잡는 것이 타당하다.  
  Source: https://proceedings.mlr.press/v155/ploeger21a.html
- MuJoCo 문서 기준 `solref/solimp`는 단순 반발계수 입력값이 아니라 contact solver parameter다. 따라서 XML 숫자만 보고 물성을 판단하지 않고, 실제 drop sanity로 effective restitution을 확인했다.  
  Source: https://mujoco.readthedocs.io/en/stable/modeling.html#restitution

## 코드 변경

### 1. low-apex recovery memory 추가

파일:

- `src/pingpong_rl2/envs/keepup_env.py`

추가 state:

- `_last_projected_contact_apex_height`
- `_last_contact_apex_shortfall`

동작:

- contact가 발생하면 projected apex를 저장한다.
- target apex `0.30m` 대비 부족분을 다음 descent에서 사용할 수 있게 기억한다.
- reset/info/training config에도 이 값과 관련 옵션을 노출했다.

### 2. recovery progress reward 추가

추가 env 옵션:

- `contact_apex_recovery_progress_reward_weight`

새 reward term:

- `contact_apex_recovery_progress_term`

동작:

- 이전 contact apex가 target보다 낮았고, 현재 upward contact가 이전 apex보다 높아졌을 때만 보상한다.
- 즉 `0.12m -> 0.18m` 같은 회복은 보상하지만, 여전히 target 미만이라는 이유만으로 모든 신호를 음수로 만들지 않는다.
- 이미 target 이상으로 회복된 뒤에는 이 보상이 꺼진다.

의도:

- 기존 `contact_apex_progress_term`은 현재 contact가 target에 가까운지 본다.
- 새 term은 낮게 망친 뒤 다음 타격에서 더 높게 띄우는 회복 행동을 따로 학습시킨다.

### 3. contact-frame recovery feedforward 추가

추가 env 옵션:

- `contact_frame_low_apex_recovery_lift_gain`
- `contact_frame_low_apex_recovery_lift_max`
- `contact_frame_low_apex_recovery_velocity_gain`
- `contact_frame_low_apex_recovery_velocity_max`

동작:

- 이전 contact apex shortfall이 있을 때만 켜진다.
- 공이 descending 중이고 contact 준비도가 높을 때만 켜진다.
- base strike z offset에 extra lift를 더한다.
- target racket z velocity에도 extra upward velocity를 더한다.
- 정상 높이로 회복되면 shortfall이 0이 되어 꺼진다.

의도:

- 모든 타격을 무작정 세게 치게 만들지 않는다.
- "낮게 끝난 다음 타격"만 더 들어 올리도록 상태 의존적인 보조 primitive를 만든다.

### 4. v14 preset 조정

파일:

- `scripts/run_ppo_learning.py`

주요 변경:

- `low_apex_contact_height_threshold`: `0.20 -> 0.14`
- `low_apex_contact_grace_count`: `1 -> 3`
- `contact_apex_recovery_progress_reward_weight = 0.70`
- `contact_frame_low_apex_recovery_lift_gain = 0.024`
- `contact_frame_low_apex_recovery_lift_max = 0.045`
- `contact_frame_low_apex_recovery_velocity_gain = 0.28`
- `contact_frame_low_apex_recovery_velocity_max = 0.45`
- `contact_frame_apex_lift_gain`: `0.10 -> 0.12`
- `contact_frame_apex_lift_max`: `0.095 -> 0.110`
- `contact_frame_velocity_target_max`: `2.7 -> 2.9`
- `stable_cycle_reward_weight`: `0.80 -> 0.90`

의도:

- 진짜 죽은 낮은 타격은 여전히 종료한다.
- 다만 v13처럼 회복 가능한 낮은 타격을 바로 끊지 않고, 몇 번의 회복 시도를 학습하게 한다.
- 방향/next-intercept 개선은 유지하면서 height recovery를 추가한다.

### 5. analysis field 추가

파일:

- `scripts/run_ppo_rebound_analysis.py`

추가 기록:

- `last_projected_contact_apex_height_above_racket`
- `last_contact_apex_shortfall`
- `contact_apex_progress_term`
- `contact_apex_recovery_progress_term`
- `contact_frame_apex_lift`
- `contact_frame_low_apex_recovery_lift`
- `contact_frame_low_apex_recovery_velocity`

추가 summary:

- `mean_last_contact_apex_shortfall`
- `mean/max_contact_frame_low_apex_recovery_lift`
- `mean/max_contact_frame_low_apex_recovery_velocity`
- `mean_contact_apex_recovery_progress_term`

v14 학습 후에는 recovery가 실제로 켜졌는지, 너무 약한지, 켜져도 apex가 안 오르는지를 바로 분리해서 볼 수 있다.

## 물성 / XML 검토

학습 직전 sanity:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_material_sanity.py --scene-path assets/scene.xml
PYTHONPATH=src conda run -n mujoco_env python scripts/run_material_sanity.py --scene-path assets/scene_racket_outward.xml
```

결과:

- gravity `-9.81`
- timestep `0.002`, control dt `0.02`
- ball radius `0.02m`
- ball mass `0.0027kg`
- racket body mass `0.18kg`
- default scene effective normal restitution approx `0.968`
- outward scene effective normal restitution approx `0.965`

판단:

- 탁구공 크기/질량은 실제 탁구공과 맞다.
- racket mass도 이전 수정 후 과도한 compiled mass 문제가 사라졌다.
- restitution은 solver 기반 effective 값이므로 계속 기록해야 하지만, 현재 병목은 물성보다 낮은 apex recovery 신호/termination curriculum 쪽이다.

## 검증

문법/오타:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  scripts/run_ppo_learning.py \
  scripts/run_ppo_rebound_analysis.py \
  scripts/run_ppo_evaluation.py \
  scripts/run_material_sanity.py
```

결과: 통과.

테스트:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest \
  tests/test_scene_load.py \
  tests/test_keepup_env.py \
  tests/test_keepup_contract_features.py \
  tests/test_ppo_runs.py \
  tests/test_vector_env.py
```

결과: `110 tests` 통과.

학습 스크립트 smoke:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally_smoke \
  --run-version v14_recovery_wiring \
  --output-dir artifacts/tmp/pmk_cf_self_rally_v14_recovery_smoke \
  --reset-model \
  --smoke
```

결과:

- PPO 생성/학습 루프/저장 경로 통과
- bootstrap base accepted episodes `10`, samples `2938`
- bootstrap follow-up accepted episodes `2`, samples `212`
- smoke evaluation mean useful `0.5`, max useful `1`

분석 스크립트 smoke:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v13/pmk_cf_self_rally_v13_best_model.zip \
  --episodes 5 \
  --analysis-name pmk_cf_self_rally_v13_recovery_wiring_smoke
```

결과: 새 analysis field 포함 runtime 통과. v13 저장 config에는 recovery gain이 없으므로 recovery lift/velocity가 0으로 찍히는 것은 정상이다.

프리셋 resolve 확인:

- preset `contact_frame_self_rally_candidate`
- target ball height `0.30`
- terminate on low apex contact `True`
- low apex threshold `0.14`
- low apex grace count `3`
- apex progress reward `0.90`
- recovery progress reward `0.70`
- recovery lift gain/max `0.024 / 0.045`
- recovery velocity gain/max `0.28 / 0.45`
- velocity target max `2.9`
- stable contact/cycle weight `1.40 / 0.90`

## 다음 학습 명령

v13은 termination curriculum 자체가 바뀌었으므로 resume하지 않는다. v14는 새 run으로 reset 학습한다.

```bash
cd mujoco/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v14 \
  --reset-model \
  --total-timesteps 2000000
```

학습 완료 후 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v14/pmk_cf_self_rally_v14_best_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v14_contact_diagnosis
```

## v14 이후 판단 기준

v14가 좋아졌는지 볼 때 useful bounce만 보지 말고 아래 순서로 판단한다.

1. mean steps와 contacts가 v13보다 늘어났는가
2. failure가 `low_apex_contact=100/100`에서 벗어났는가
3. median projected apex가 `0.169m`보다 올라갔는가
4. upward apex `<0.20m` 비율이 v13의 `0.597`보다 내려갔는가
5. next intercept reachable rate가 v13의 `0.770` 근처를 유지하는가
6. recovery lift/velocity가 실제로 켜지고, 켜진 뒤 apex가 증가하는가
7. stable cycle 2+ / 3+ rate가 증가하는가

가능한 다음 처방:

- episode 길이는 늘었지만 apex가 여전히 낮으면 recovery gain/max와 velocity target max를 한 단계 더 올린다.
- apex는 올랐지만 lateral drift가 커지면 trajectory tilt / post-contact lateral velocity penalty를 다시 강화한다.
- recovery field가 거의 0이면 shortfall memory나 readiness gate가 너무 보수적인 것이다.
- recovery field가 max에 자주 닿는데도 apex가 안 오르면 contact-frame velocity target 또는 controller velocity feedback이 병목이다.
