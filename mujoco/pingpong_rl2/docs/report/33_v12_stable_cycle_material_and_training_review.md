# v12 이후 stable-cycle / 물성 / 학습 직전 검토

작성일: 2026-06-02

## 목표

최종 목표는 YouTube short처럼 탁구공을 일정한 높이와 방향으로 계속 위로 치는 self-rally다. 이번 정리는 `pmk_cf_self_rally_v12` 2M 이후, 단순 reward weight 조정보다 근본적으로 필요한 부분을 코드에 반영하고 학습 직전 검증까지 끝낸 기록이다.

## v12 진단

`artifacts/ppo_runs/pmk_cf_self_rally_v12/analysis/pmk_cf_self_rally_v12_contact_diagnosis_summary.json` 기준:

- `mean_useful_bounces = 0.40`
- `max_useful_bounces = 3`
- `one_or_more_useful_bounce_rate = 0.30`
- `two_or_more_useful_bounce_rate = 0.08`
- failure: `low_apex_contact=57`, `ball_out_of_bounds=34`, `floor_contact=4`, `time_limit=2`, `ball_speed_limit=3`
- 전체 contact `919`, useful contact rate `0.0435`
- mean projected apex `0.199m`, median `0.167m`
- upward contact 중 apex `< 0.20m` 비율 `0.543`
- upward contact 중 target apex `< 0.30m` 미달 비율 `0.780`
- terminal contact mean apex `0.147m`, terminal upward apex `< 0.20m` 비율 `0.796`

해석:

- v12는 첫 contact 일부는 useful까지 가지만, 2~3번째 contact에서 높이가 급격히 낮아진다.
- 단순히 더 오래 학습하면 해결된다고 보기 어렵다. 정책이 "낮아도 일단 맞히는" 방향으로 갈 수 있는 reward 구조가 아직 남아 있었다.
- 필요한 것은 개별 contact 품질이 아니라, `좋은 높이 -> 쉬운 다음 intercept -> 다시 좋은 높이`로 이어지는 stable cycle 자체를 objective로 만드는 것이다.

## 참고한 외부 지식

- Google DeepMind의 competitive robot table tennis 연구는 저수준 skill controller, task distribution/curriculum, sim-to-real gap을 줄이는 구조를 강조한다. 여기서는 이미 contact-frame primitive + residual action 구조가 있으므로, 새 네트워크 구조보다 이 구조 안에서 stable-cycle objective와 checkpoint selection을 강화했다.  
  Source: https://deepmind.google/research/publications/achieving-human-level-competitive-robot-table-tennis/
- Google Research의 model-free table tennis 연구도 reward와 curriculum tuning이 return rate에 핵심이라고 보고한다. 그래서 "PPO timesteps 추가"가 아니라 reward/curriculum/checkpoint 기준을 같이 바꿨다.  
  Source: https://research.google/pubs/robotic-table-tennis-with-model-free-reinforcement-learning/
- Ploeger et al.의 real-world juggling 연구는 동적 juggling에서 policy representation, initialization, optimization을 함께 설계해야 sample-efficient하고 안전하다고 설명한다. 그래서 heuristic bootstrap을 다시 켰고, multi-ball은 현재 env를 바로 늘리는 것이 아니라 별도 task로 봐야 한다.  
  Source: https://proceedings.mlr.press/v155/ploeger21a.html

## 코드 변경

### 1. stable-cycle objective 추가

파일:

- `src/pingpong_rl2/envs/keepup_env.py`
- `tests/test_keepup_env.py`

추가된 env 옵션:

- `stable_cycle_reward_weight`
- `stable_cycle_reward_cap`
- `stable_cycle_min_easy_next_ball_score`

동작:

- contact가 `success_reason == useful_keepup_bounce`이어야 한다.
- projected apex가 target apex 이상이어야 한다.
- projected apex가 `height_tolerance` window 안이어야 한다.
- next descending intercept가 reachable이어야 한다.
- 설정된 경우 `easy_next_ball_score >= stable_cycle_min_easy_next_ball_score`를 만족해야 한다.
- 만족하면 `stable_cycle_count`와 `consecutive_stable_cycle_count`를 증가시킨다.
- 연속 stable contact에는 `stable_cycle_term`을 추가한다. streak 보상은 cap까지만 증가한다.

의도:

- 기존 `stable_contact_term`은 "이번 contact가 높이와 다음 intercept를 동시에 만족하는가"를 dense하게 본다.
- 새 `stable_cycle_term`은 "그런 contact가 연속되는가"를 별도로 보상한다.
- 즉 단타 성공보다 일정 높이 self-rally 주기를 더 직접적으로 학습하게 한다.

### 2. self-rally preset v13 조정

파일:

- `scripts/run_ppo_learning.py`

주요 변경:

- checkpoint 활성화: `checkpoint_interval = 100_000`, `checkpoint_eval_episodes = 40`
- checkpoint selection이 stable-cycle 지표를 useful bounce보다 먼저 본다.
- vertical/recovery primitive 보강:
  - `controller_velocity_feedback_gain = 0.45`
  - `controller_max_velocity_step = 0.065`
  - `contact_frame_base_strike_z_boost = 0.028`
  - `contact_frame_apex_lift_gain = 0.10`
  - `contact_frame_apex_lift_max = 0.095`
  - `contact_frame_velocity_target_max = 2.7`
- low-apex 기준 강화:
  - `low_apex_contact_height_threshold = 0.20`
  - `low_apex_contact_grace_count = 1`
- 방향/다음 intercept 품질 강화:
  - `next_intercept_xy_error_penalty_weight = 1.00`
  - `post_contact_lateral_velocity_penalty_weight = 0.45`
  - `contact_apex_under_target_penalty_weight = 0.80`
  - `contact_apex_progress_reward_weight = 0.90`
  - `stable_contact_reward_weight = 1.40`
  - `stable_cycle_reward_weight = 0.80`
  - `stable_cycle_min_easy_next_ball_score = 0.45`
- heuristic bootstrap 재활성화:
  - base: 80 episodes, `post_success`, 20 epochs
  - follow-up: 10 epochs, `post_success_reachable`, min useful bounces 2

검토 결과, 20 episode bootstrap sanity에서 base dataset은 3개 episode / 598 samples / mean useful 2.33이 모였다. follow-up도 1개 episode / 105 samples가 모였다. 즉 비어 있는 teacher가 아니다.

### 3. 분석/평가 지표 추가

파일:

- `scripts/run_ppo_rebound_analysis.py`
- `scripts/run_ppo_evaluation.py`

추가 지표:

- `mean_stable_cycles`
- `max_stable_cycles`
- `one_or_more_stable_cycle_rate`
- `two_or_more_stable_cycle_rate`
- `three_or_more_stable_cycle_rate`
- contact CSV의 `stable_cycle_observed`, `stable_cycle_term`, `stable_contact_term`

이제 학습 후에는 useful bounce뿐 아니라 안정 주기 자체를 기준으로 비교할 수 있다.

### 4. XML 물성 검토 및 수정

파일:

- `assets/franka/panda.xml`
- `assets/franka/panda_racket_outward.xml`
- `scripts/run_material_sanity.py`

확인한 점:

- ITTF 2026 Statutes 기준 탁구공은 diameter `40mm`, mass `2.7g`이다. 현재 XML의 ball은 radius `0.02m`, mass `0.0027kg`로 일치한다.  
  Source: https://documents.ittf.sport/sites/default/files/public/2026-02/2026_Statutes_v1_consolidated_clean.pdf
- MuJoCo의 `solref/solimp`는 실제 "반발계수"를 직접 입력하는 값이 아니라 soft contact solver parameter다. MuJoCo 문서도 restitution은 solver/contact softness와 함께 나타나는 효과로 설명한다.  
  Source: https://mujoco.readthedocs.io/en/stable/modeling.html#restitution

발견한 문제:

- 기존 XML은 `racket_head`에 `mass="0.11"`이 있었지만, 같은 body의 visual/rim/collideless geoms가 기본 density로 body inertia에 들어가면서 compiled racket body mass가 약 `0.541kg`이었다.
- 이는 실제 탁구채/마운트로 보기에는 과도하고, collision/control dynamics를 왜곡할 수 있다.

수정:

- `racket` body에 명시적 inertial을 추가했다.
- 기본 scene compiled racket body mass: `0.541kg -> 0.18kg`
- outward scene compiled racket body mass: `0.18kg`

sanity:

- `scripts/run_material_sanity.py --scene-path assets/scene.xml --episodes 3`
  - ball radius `0.02m`, ball mass `0.0027kg`
  - racket body mass `0.18kg`
  - static racket drop effective normal restitution approx `0.968`
- `scripts/run_material_sanity.py --scene-path assets/scene_racket_outward.xml --episodes 1`
  - racket body mass `0.18kg`
  - effective normal restitution approx `0.965`

남은 caveat:

- 이 restitution은 정적 라켓 낙하 sanity이지 실제 고무/스핀/공기저항까지 보정한 값은 아니다.
- 현재 self-rally 목표에는 "너무 죽는 접촉"보다 "일정한 높이와 다음 intercept"가 더 큰 병목이므로, 이번에는 mass 오류를 수정하고 restitution은 기록/감시 대상으로 둔다.

### 5. smoke 순서 버그 수정

`--smoke` 축소가 preset 적용 전에 실행되면, preset이 bootstrap/n_steps 등을 다시 크게 덮어쓸 수 있었다. `apply_env_preset(args)` 이후 smoke 축소가 적용되도록 수정했다.

## 2개 이상 공 저글링 분석

현재 env는 하나의 `ball` freejoint, 하나의 contact trace, 하나의 next-intercept target을 전제로 한다. 2개 이상으로 가려면 아래가 별도 구현되어야 한다.

- XML에 ball body/freejoint를 여러 개 추가
- observation을 ball별 상태 + ordering/assignment invariant 구조로 변경
- contact trace가 어느 공과의 contact인지 식별
- reward가 "다음에 쳐야 하는 공"과 "기다리는 공"을 구분
- reset/curriculum: 1개 stable cycle -> 2개 alternating cycle 순서

현 구조에서 바로 가능한 최대는 1개다. 별도 multi-ball env를 만든다면 현실적으로 첫 목표는 2개다. 3개 이상은 한 팔/한 라켓/현재 control horizon에서는 연구 난도가 크게 올라가므로, 현재 self-rally 1개가 안정화된 뒤에만 고려하는 것이 맞다.

## 학습 직전 검증 결과

명령 및 결과:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m py_compile \
  src/pingpong_rl2/envs/keepup_env.py \
  scripts/run_ppo_learning.py \
  scripts/run_ppo_rebound_analysis.py \
  scripts/run_ppo_evaluation.py \
  scripts/run_material_sanity.py
```

결과: 통과.

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest \
  tests/test_scene_load.py \
  tests/test_keepup_env.py \
  tests/test_keepup_contract_features.py \
  tests/test_ppo_runs.py \
  tests/test_vector_env.py
```

결과: `108 tests` 통과.

설정 resolve 확인:

- preset: `contact_frame_self_rally_candidate`
- scene: `assets/scene.xml`
- target apex: `0.30`
- low apex threshold: `0.20`
- stable contact weight: `1.40`
- stable cycle weight: `0.80`
- stable cycle cap: `4`
- stable cycle min easy score: `0.45`
- checkpoint interval: `100000`
- checkpoint eval episodes: `40`
- bootstrap base/follow-up 활성화 확인

## 다음 학습 명령

XML 물성, reward 구조, checkpoint 기준이 바뀌었으므로 v12를 resume하지 않는다. 새 버전으로 reset 학습한다.

```bash
cd mujoco/pingpong_rl2
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v13 \
  --reset-model \
  --total-timesteps 2000000
```

학습 완료 후 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v13/pmk_cf_self_rally_v13_best_model.zip \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_v13_contact_diagnosis
```

성공 여부는 최소한 아래를 본다.

- `mean_stable_cycles`
- `two_or_more_stable_cycle_rate`
- `three_or_more_stable_cycle_rate`
- `contact_summary.stable_cycle_contact_rate`
- upward contact apex `< 0.20m` 비율 감소
- terminal contact apex 평균 증가
- low_apex_contact failure 감소
