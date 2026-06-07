# run_ppo_evaluation.py

## 한 줄 역할

`scripts/run_ppo_evaluation.py`는 저장된 PPO 모델을 headless로 실행해 episode 단위 성능을 JSON으로 출력하는 평가 entrypoint다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_ppo_evaluation.py \
  --run-name keep1 \
  --run-version v39_17d_mid_curriculum_fixed \
  --episodes 20
```

## 코드 흐름

1. 모델 경로를 결정한다.
   - `--model-path`가 있으면 그 파일을 직접 쓴다.
   - 없으면 `--run-name`, `--run-version`을 `resolve_requested_run_name()`으로 합치고 `resolve_saved_model_path()`가 artifact 경로를 찾는다.

2. 학습 당시 환경 설정을 복원한다.
   - `resolve_env_kwargs_for_model()`이 모델 옆 training summary JSON을 찾아 `env_config`/`config` 기반 kwargs를 만든다.
   - CLI override로 reset range, ball height, success threshold, max episode steps 등을 평가 시점에만 바꿀 수 있다.

3. 단일 `PingPongKeepUpGymEnv`를 만든다.
   - 평가는 vector env가 아니라 한 episode씩 도는 단일 env다.
   - `env.training_config()`를 읽어 결과 JSON에 포함한다.

4. episode step limit를 정한다.
   - 학습 환경의 `max_episode_steps`가 무제한이면 기본으로 3600 step safety cap을 둔다.
   - `--episode-step-limit 0` 이하를 주면 평가 cap을 끌 수 있다.

5. PPO policy를 로드하고 episode loop를 돈다.
   - `PPO.load(str(model_path))`
   - 매 step `model.predict(observation, deterministic=not args.stochastic)`을 호출한다.
   - `env.step(action)` 결과가 terminated/truncated면 episode를 끝낸다.

6. 지표를 모아 출력한다.
   - episode return
   - contact count
   - useful bounce count
   - stable cycle count
   - failure reason
   - 1/2/3/10/20/30회 이상 useful bounce rate
   - 1/2/3/10/20/30회 이상 stable cycle rate

## 주요 호출 관계

```text
run_ppo_evaluation.py
  -> utils/ppo_runs.py       # 모델 경로와 env kwargs 복원
  -> envs/gym_env.py         # Gym wrapper
  -> envs/keepup_env.py      # reward/termination/info 생성
  -> stable_baselines3.PPO   # 저장된 policy 로드와 action 예측
```

## 발표 때 설명 포인트

- 이 파일은 학습하지 않는다. 저장된 policy를 같은 환경 조건에서 재생하는 평가 스크립트다.
- 최종 성능 표에 넣을 수 있는 가장 단순한 지표는 `mean_useful_bounces`, `max_useful_bounces`, `thirty_or_more_useful_bounce_rate`, `failure_counts`다.
- 더 깊은 실패 원인이나 contact 물리량은 `run_ppo_rebound_analysis.py`가 담당한다.
