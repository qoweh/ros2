# run_heuristic_keepup_diagnostic.py

## 한 줄 역할

`scripts/run_heuristic_keepup_diagnostic.py`는 PPO 없이 `HeuristicKeepUpPolicy`만으로 keep-up이 얼마나 되는지 확인하는 baseline/diagnostic entrypoint다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_heuristic_keepup_diagnostic.py \
  --analysis-name heuristic_keepup_diagnostic \
  --episodes 20 \
  --print-episodes
```

## 코드 흐름

1. heuristic과 env 설정을 CLI로 받는다.
   - action mode, reset range, velocity range, target height, tilt residual, contact-frame planner, reward penalty 등을 조절할 수 있다.

2. `build_env_kwargs(args)`가 `PingPongKeepUpGymEnv`용 kwargs를 만든다.
   - 관측 확장 옵션인 task phase, contact context, next intercept observation을 켠다.
   - scene path가 있으면 XML을 바꿔 A/B 진단도 가능하다.

3. `HeuristicKeepUpPolicy`를 만든다.
   - return blend, recovery blend, strike z boost, residual action을 사용한다.
   - 이 policy는 학습된 neural network가 아니라 hand-coded controller다.

4. episode loop를 실행한다.
   - 매 step `policy.predict(env.base_env)`로 action을 만든다.
   - `env.step(action)` 후 contact event가 있으면 outgoing velocity, predicted apex, next intercept, contact normal alignment 등을 기록한다.

5. summary와 CSV를 저장한다.
   - `<analysis_name>_summary.json`
   - `<analysis_name>_episodes.csv`
   - `<analysis_name>_contacts.csv`

## 주요 호출 관계

```text
run_heuristic_keepup_diagnostic.py
  -> controllers/heuristic_keepup.py  # scripted policy
  -> envs/gym_env.py                  # Gym wrapper
  -> envs/keepup_env.py               # contact info/reward/failure reason
```

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 PPO model을 전혀 로드하지 않는다. 사람이 설계한 `HeuristicKeepUpPolicy`가 현재 환경 상태를 보고 residual action을 만들고, 그 action을 같은 `PingPongKeepUpGymEnv`에 넣어 baseline 성능을 측정한다.

```text
run_heuristic_keepup_diagnostic.py
  -> parse_args()
  -> build_env_kwargs(args)
  -> PingPongKeepUpGymEnv(**env_kwargs)
  -> HeuristicKeepUpPolicy(...)
  -> for episode
       env.reset(seed)
       policy.reset()
       while not done
         action = policy.predict(env.base_env)
         env.step(action)
         contact event면 contact row 기록
       episode row 기록
  -> summary.json / episodes.csv / contacts.csv 저장
```

`build_env_kwargs()`는 CLI 옵션을 강화학습 환경 생성자 kwargs로 바꾼다. 여기에는 action mode, target height, reset 분포, contact-frame planner, velocity target, tilt assist, reward penalty, oracle contact 옵션이 들어간다. 관측 옵션으로 task phase, contact context, next intercept observation도 켠다. PPO와 같은 env 계약 안에서 heuristic만 바꿔 실행하려는 목적이다.

## heuristic policy가 하는 일

`HeuristicKeepUpPolicy`는 학습된 network가 아니다. 내부 phase, 공의 예상 위치, 다음 intercept, strike/recovery 상태를 읽고 hand-coded rule로 action vector를 채운다.

이 파일에서는 CLI로 `return_blend`, `recovery_blend`, `strike_z_boost`, `strike_time_horizon`, position residual, tilt residual, followup lift residual 등을 조정할 수 있다. 즉 "정책을 학습한다"기보다 "사람이 정한 규칙형 baseline을 여러 설정으로 실행해 본다"에 가깝다.

매 step에서 policy는 `env.base_env`를 직접 받는다. Gym wrapper의 observation만 보는 PPO와 달리 heuristic은 환경 내부 helper와 상태를 더 직접적으로 읽을 수 있다. 그래서 이 결과를 PPO 성능과 비교할 수는 있지만, "동일한 관측만 보고 결정한 학습 정책"으로 해석하면 안 된다.

## contact row와 summary가 의미하는 것

contact event가 발생하면 이 파일은 outgoing velocity error, predicted apex error, predicted next intercept error, easy next ball score, contact normal alignment, oracle contact 여부 등을 contact row에 저장한다. 이 값들은 heuristic이 단순히 공을 맞혔는지보다, 반복 타격 가능한 방향으로 보냈는지를 판단하는 데 쓰인다.

episode가 끝나면 useful bounce 수, contact 수, failure reason을 episode row로 저장한다. summary에는 평균/최대 useful bounce, threshold rate, reachable rate, useful contact rate, outgoing velocity source 비율, apex/intercept error 평균이 들어간다.

이 파일의 결과가 좋으면 환경과 hand-coded primitive가 물리적으로 가능한 타격 구조를 갖고 있다는 근거가 된다. 반대로 heuristic도 거의 못 치면 PPO 문제가 아니라 contact primitive, reward, reset 조건, 물리 파라미터 문제일 가능성을 먼저 봐야 한다.

## heuristic bootstrap과의 관계

이 파일은 bootstrap을 실행하지 않는다. 하지만 같은 `HeuristicKeepUpPolicy`를 사용한다. bootstrap에서는 heuristic rollout으로 observation/action dataset을 모아 PPO actor를 사전 모방학습시키고, 이 diagnostic에서는 heuristic 자체를 독립 baseline으로 평가한다.

따라서 "heuristic을 썼다"는 말은 두 경우를 구분해야 한다.

```text
diagnostic heuristic
  -> 이 파일처럼 PPO 없이 baseline 성능을 측정

heuristic bootstrap
  -> run_ppo_learning.py에서 새 PPO actor를 warm start
```

## 발표 때 설명 포인트

- 이 파일은 PPO 학습 결과가 아니다. “환경과 보상/접촉 모델이 물리적으로 가능한가?”를 보는 baseline이다.
- heuristic이 잘하면 task 설계가 가능하다는 증거가 되고, PPO가 못하면 exploration/representation/reward 문제를 의심할 수 있다.
- v39 학습 자체에서는 이 diagnostic script가 호출되지 않는다.
