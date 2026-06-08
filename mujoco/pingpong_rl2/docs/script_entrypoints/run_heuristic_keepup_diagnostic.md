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

## 발표 때 설명 포인트

- 이 파일은 PPO 학습 결과가 아니다. “환경과 보상/접촉 모델이 물리적으로 가능한가?”를 보는 baseline이다.
- heuristic이 잘하면 task 설계가 가능하다는 증거가 되고, PPO가 못하면 exploration/representation/reward 문제를 의심할 수 있다.
- v39 학습 자체에서는 이 diagnostic script가 호출되지 않는다.
