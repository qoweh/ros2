# pingpong_rl2 keep-up phase contract implementation report

## 1. 이번 구현 범위

`09_keepup_task_rethink_plan.md`의 우선순위 1~3을 바로 코드로 옮겼다.

- env observation 보강
  - `include_task_phase_observation`
  - `include_contact_context_observation`
  - `include_next_intercept_observation`
- env info/diagnostic 보강
  - `phase_name`, `phase_one_hot`
  - `time_since_contact`
  - `next_intercept_*`
  - `easy_next_ball_score`
- 작은 event reward 후보 추가
  - `next_intercept_reachable_bonus_weight`
  - `easy_next_ball_reward_weight`
- heuristic diagnostic baseline 추가
  - `src/pingpong_rl2/controllers/heuristic_keepup.py`
  - `scripts/run_heuristic_keepup_diagnostic.py`
  - `scripts/run_viewer.py --mode heuristic`
- PPO 실험 preset 추가
  - `phase_contract_candidate`

## 2. 구현 의도

기존 env는 `prepare/strike` 정보는 비교적 강했지만, `return/recovery`는 policy가 직접 보기 어려웠다.

이번 변경의 목적은 다음 두 가지다.

1. policy가 다음 공 feasibility를 observation으로 직접 보게 하기
2. PPO 전에 heuristic baseline으로 환경/제어/물리가 반복 keep-up을 허용하는지 분리 진단하기

## 3. 새 observation / reward contract

### 3.1 observation

새 observation은 모두 opt-in이다. 기존 best model replay 호환성을 깨지 않기 위해 default는 그대로 유지했다.

- `phase_one_hot`
  - `prepare`, `strike`, `return_shaping`, `recovery`
- `time_since_contact`
- `successful_bounce_count_clipped`
- `next_intercept_relative_xy`
- `next_intercept_time`
- `next_intercept_reachable`
- `next_intercept_recovery_distance`
- `next_intercept_recovery_readiness`

### 3.2 reward

새 reward도 default off다.

- `next_intercept_reachable_bonus_weight`
  - useful contact 이후 다음 descending intercept가 strike zone 안이면 작은 bonus
- `easy_next_ball_reward_weight`
  - useful contact 이후 analysis-style `easy_next_ball_score`의 positive part만 작은 event reward로 사용

현재 preset에서는 더 안전한 첫 후보로 `reachable bonus`만 켰다.

## 4. 새 실행 경로

### 4.1 heuristic diagnostic

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --episodes 20 \
  --analysis-name heuristic_keepup_diag_v1
```

### 4.2 heuristic viewer

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_viewer.py \
  --mode heuristic \
  --episodes 3
```

### 4.3 새 PPO preset smoke / short run

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset phase_contract_candidate \
  --run-name phase_contract \
  --run-version v1 \
  --reset-model \
  --total-timesteps 100000
```

## 5. 검증 결과

### 5.1 단위 테스트

아래 focused test가 통과했다.

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest discover -s tests -p 'test_keepup*.py'
```

결과:

- `53` tests passed

### 5.2 heuristic smoke

실행:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py \
  --episodes 5 \
  --analysis-name heuristic_keepup_smoke_v1
```

요약:

- `mean_useful_bounces=0.40`
- `max_useful_bounces=1`
- `two_or_more_useful_bounce_rate=0.00`
- `failure_counts.ball_out_of_bounds=5/5`
- `next_intercept_reachable_rate=0.368`
- `useful_contact_next_intercept_reachable_rate=0.0`

해석:

- heuristic baseline은 end-to-end로 동작한다.
- 하지만 좁은 reset에서도 아직 `2+ useful bounce`를 만들지 못했다.
- 즉, 지금 문제는 단순 PPO 튜닝 이전에 여전히 `contact 이후 next-ball quality` 또는 `contact 자체의 return direction`이 부족할 가능성이 높다.

### 5.3 PPO preset smoke

실행:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --preset phase_contract_candidate \
  --smoke \
  --n-envs 2 \
  --eval-episodes 1 \
  --checkpoint-interval 512 \
  --checkpoint-eval-episodes 1 \
  --run-name phase_contract_smoke \
  --run-version v1 \
  --reset-model
```

요약:

- preset이 end-to-end로 정상 실행됐다.
- `completed_timesteps=1024`
- best checkpoint 저장까지 정상 동작했다.

## 6. 현재 판단

이번 구현으로 다음 두 질문에 대한 기반은 생겼다.

1. policy가 다음 공 feasibility를 볼 수 있는가
2. PPO 없이도 scripted baseline이 반복 keep-up을 조금이라도 만들 수 있는가

현재까지의 짧은 smoke 결과는 아래 쪽에 더 가깝다.

- observation contract는 이제 이전보다 직접적이다.
- 하지만 heuristic baseline도 아직 `2+ useful bounce`를 만들지 못한다.

따라서 다음 단계는 또 `assist weight 0.4/0.5/0.6` 미세 튜닝이 아니라, 아래 둘 중 하나를 더 직접 건드리는 것이 맞다.

1. heuristic baseline의 contact-direction control 강화
   - fixed negative pitch vs timed ramp vs stronger inward base 비교
2. contact 직전 strike target을 next-ball workspace와 더 정렬
   - 단순 anchor 점보다 workspace region 기준으로 contact point/face를 유도

## 7. 바로 이어서 할 실험

1. `run_viewer.py --mode heuristic`로 heuristic baseline failure 장면 확인
2. heuristic diagnostic에서 pitch variant를 두세 개만 비교
3. `phase_contract_candidate`로 50k~100k 짧은 PPO run 하나만 돌리고 rebound analysis로 `two+`, `reachable rate`, `easy_next_ball_score`를 같이 보기