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
