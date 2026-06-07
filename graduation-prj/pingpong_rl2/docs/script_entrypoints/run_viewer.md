# run_viewer.py

## 한 줄 역할

`scripts/run_viewer.py`는 PPO policy, zero action, heuristic 중 하나를 MuJoCo viewer에서 직접 보는 데모 entrypoint다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_viewer.py \
  --mode policy \
  --run-name keep1 \
  --run-version v39_17d_mid_curriculum_fixed \
  --episodes 3
```

heuristic만 보고 싶으면:

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_viewer.py \
  --mode heuristic \
  --episodes 3
```

## 코드 흐름

1. action source를 고른다.
   - `policy`: 저장된 PPO 모델을 로드한다.
   - `heuristic`: `HeuristicKeepUpPolicy`를 만든다.
   - `zero_action`: action space와 같은 크기의 0 벡터를 넣는다.

2. 모델이 있으면 env 설정을 복원한다.
   - `resolve_saved_model_path()`로 모델을 찾는다.
   - `resolve_env_kwargs_for_model()`로 training summary의 환경 설정을 불러온다.
   - CLI 옵션으로 scene, reset, planner, reward penalty 일부를 덮어쓸 수 있다.

3. heuristic-only mode는 필요한 관측/행동 구성을 직접 켠다.
   - 모델 summary가 없으므로 `action_mode="position_strike"`와 task phase/contact context/next intercept 관측을 명시한다.

4. MuJoCo passive viewer loop를 실행한다.
   - `mujoco.viewer.launch_passive(sim.model, sim.data)`로 viewer를 연다.
   - 매 frame action을 계산하고 `env.step(action)`을 호출한다.
   - episode가 끝나면 return, contact, useful bounce, failure reason을 콘솔에 출력한다.

## 주요 호출 관계

```text
run_viewer.py
  -> utils/ppo_runs.py
  -> controllers/heuristic_keepup.py
  -> envs/gym_env.py
  -> mujoco.viewer
```

## 발표 때 설명 포인트

- 실제 데모/GIF 후보는 이 파일이 가장 직접적이다.
- policy mode는 최종 모델 행동 시각화, heuristic mode는 baseline 행동 시각화, zero_action은 환경/중력/접촉 sanity 비교에 쓸 수 있다.
