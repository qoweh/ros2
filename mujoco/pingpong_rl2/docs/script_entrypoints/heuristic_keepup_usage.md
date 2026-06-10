# heuristic_keepup.py 사용 여부

## 결론

`src/pingpong_rl2/controllers/heuristic_keepup.py`의 `HeuristicKeepUpPolicy`는 학습된 PPO policy가 아니다. 사람이 설계한 scripted baseline/teacher policy다.

v39 최종 run인 `keep1_v39_17d_mid_curriculum_fixed` 학습 자체에서는 직접 사용되지 않았다.

프로젝트 전체의 과거 모델 사용 이력은 [heuristic_bootstrap_audit.md](../heuristic_bootstrap_audit.md)에 별도로 정리했다.

## 코드상 사용 위치

1. `src/pingpong_rl2/training/bootstrap.py`
   - `collect_heuristic_bootstrap_dataset()` 안에서 `HeuristicKeepUpPolicy()`를 만든다.
   - `run_ppo_learning.py`가 bootstrap 조건을 만족할 때만 호출한다.

2. `scripts/run_ppo_learning.py`
   - optional bootstrap 블록에서만 관련 함수가 호출된다.
   - 조건:

```python
starting_model_path is None
and args.bootstrap_heuristic_episodes > 0
and args.bootstrap_epochs > 0
```

3. `scripts/run_heuristic_keepup_diagnostic.py`
   - PPO 없이 heuristic baseline 성능과 실패 원인을 진단한다.

4. `scripts/run_contact_feasibility_map.py`
   - heuristic parameter 조합을 grid sweep한다.

5. `scripts/run_viewer.py`
   - `--mode heuristic`일 때 MuJoCo viewer에서 heuristic 행동을 보여준다.

6. tests
   - action contract, env info, contact-frame 기능 검증에 쓰인다.

## 학습 때 쓰이는가?

항상 쓰이지 않는다. 학습에서 쓰이는 유일한 공식 경로는 `run_ppo_learning.py`의 optional bootstrap이다.

새 PPO run에서 bootstrap 옵션이 켜져 있으면:

1. heuristic policy가 env를 직접 플레이한다.
2. 관측과 행동을 dataset으로 모은다.
3. PPO actor network를 supervised 방식으로 먼저 맞춘다.
4. 그 다음 PPO reinforcement learning이 시작된다.

즉 bootstrap은 “PPO 학습 전에 actor를 따뜻하게 시작하는 과정”이지, PPO rollout policy 자체가 heuristic으로 바뀌는 것은 아니다.

코드 흐름으로 쓰면 다음과 같다.

```text
run_ppo_learning.py
  -> 새 학습이고 bootstrap 옵션이 켜져 있는지 확인
  -> collect_heuristic_bootstrap_dataset()
       -> PingPongKeepUpGymEnv 생성
       -> HeuristicKeepUpPolicy.predict(env.base_env)
       -> env.step(action)
       -> observation/action sample 저장
  -> bootstrap_actor_from_dataset()
       -> model.policy.get_distribution(obs)
       -> deterministic predicted action
       -> MSE(predicted action, heuristic action)
       -> actor parameter update
  -> model.learn()으로 PPO 시작
```

이 단계에서 heuristic은 teacher 역할을 한다. 하지만 PPO 학습이 시작된 뒤 rollout action을 매번 대신 만들어 주지는 않는다. PPO rollout에서는 neural policy가 observation을 보고 action을 낸다.

## diagnostic과 bootstrap의 차이

`scripts/run_heuristic_keepup_diagnostic.py`와 `scripts/run_contact_feasibility_map.py`도 같은 heuristic policy를 쓰지만 목적이 다르다. 이 파일들은 PPO actor를 사전학습시키지 않고, hand-coded policy 자체의 baseline 성능이나 contact primitive 가능성을 측정한다.

```text
diagnostic/feasibility
  -> heuristic action을 직접 env에 넣고 결과를 CSV/JSON으로 분석

bootstrap
  -> heuristic action을 dataset으로 모아 PPO actor 초기값을 맞춤
```

따라서 발표에서 "heuristic을 썼다"고 말할 때는 어떤 의미인지 구분해야 한다. 최종 v39 run 자체는 bootstrap을 직접 실행하지 않았고, diagnostic script 결과도 PPO 최종 성능이 아니라 baseline/검증 결과다.

## v39에서 쓰였는가?

v39 summary 기준으로는 쓰이지 않았다.

확인한 summary:

```text
artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/
  keep1_v39_17d_mid_curriculum_fixed_training_summary.json
```

핵심 값:

```text
training_mode = resume
starting_model_path = .../keep1_v36_17d_balanced_xyz012_model.zip
bootstrap_heuristic_episodes = 0
bootstrap_epochs = 0
bootstrap_followup_epochs = 0
bootstrap = null
```

따라서 v39 run의 학습 흐름은 “heuristic bootstrap -> PPO”가 아니라 “v36 PPO checkpoint -> v39 PPO 추가 학습”이다.

단, v36 이전 조상 run 어딘가에서 heuristic bootstrap을 썼을 가능성은 별개의 문제다. v39 run 자체가 heuristic을 직접 호출했는지에 대한 답은 “아니오”다.
