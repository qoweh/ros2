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

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 학습 파일과 다르게 PPO parameter를 업데이트하지 않는다. 이미 저장된 model zip을 불러와 deterministic 또는 stochastic action을 내보내고, 단일 `PingPongKeepUpGymEnv`에서 episode를 반복 실행한 뒤 결과를 JSON으로 출력한다.

```text
run_ppo_evaluation.py
  -> resolve_saved_model_path()
  -> resolve_env_kwargs_for_model()
  -> PingPongKeepUpGymEnv(**env_kwargs)
  -> PPO.load(model_path)
  -> for episode
       observation = env.reset(seed)
       while not done
         action = model.predict(observation)
         observation, reward, terminated, truncated, info = env.step(action)
       episode summary 누적
  -> scalar summary JSON 출력
```

`resolve_env_kwargs_for_model()`이 중요한 이유는 모델 zip만으로는 당시 환경 설정을 알 수 없기 때문이다. 이 함수는 모델 옆 training summary JSON을 찾고, 거기 저장된 `env_config`와 `config`를 바탕으로 환경 kwargs를 복원한다. 그래서 평가 스크립트는 "현재 코드 기본값"이 아니라 "그 모델이 학습될 때 저장된 설정"을 기본값으로 삼는다.

## main()을 코드 순서대로 풀어보기

먼저 CLI에서 `--model-path`가 들어왔는지 확인한다. 직접 모델 경로가 있으면 그 zip을 평가한다. 없으면 `--run-name`과 `--run-version`을 합쳐 run 이름을 만들고, `artifacts/ppo_runs/<run>/<run>_model.zip` 계열 경로를 찾는다.

모델 경로가 확정되면 `infer_run_name_from_model_path()`로 run 이름을 추론한다. 이 값은 출력 summary에 들어가며, 분석 결과가 어떤 모델에서 나온 것인지 구분하는 데 쓰인다.

그 다음 `resolve_env_kwargs_for_model()`이 평가 환경 설정을 만든다. CLI에서 `--reset-xy-range`, `--ball-height`, `--max-episode-steps`, `--success-velocity-threshold` 같은 값을 주면 학습 당시 설정 위에 평가 전용 override가 덮인다. 이 방식은 "같은 policy를 조금 다른 초기 조건에서 테스트"할 때 쓴다.

환경은 vector env가 아니라 단일 `PingPongKeepUpGymEnv`로 생성된다. 평가는 병렬 학습용 rollout이 목적이 아니라 episode별 return, contact 수, useful bounce 수를 직접 집계하는 목적이기 때문이다.

`episode_step_limit`은 평가 안전장치다. 환경의 `max_episode_steps`가 `None`이면 이론적으로 episode가 계속 갈 수 있다. 이 경우 기본값으로 3600 step cap을 둔다. `--episode-step-limit 0` 이하를 주면 이 평가 전용 cap을 끌 수 있다.

episode loop 안에서는 매 step 다음 순서로 진행된다.

```text
현재 observation
  -> model.predict(observation, deterministic=not stochastic)
  -> env.step(action)
  -> reward 누적
  -> terminated/truncated 확인
```

여기서도 실제 로봇팔 움직임은 `run_ppo_learning.py`와 같은 아래 흐름을 탄다.

```text
env.step(action)
  -> PingPongKeepUpEnv.step()
  -> residual action 해석
  -> Cartesian controller
  -> joint_targets
  -> MuJoCo mj_step()
```

episode가 끝나면 마지막 `info`에서 `failure_reason`, `contact_count`, `successful_bounce_count`, `stable_cycle_count`를 꺼낸다. 반환값 `reward`는 매 step 합산해서 episode return으로 기록한다.

## 이 파일이 만들어 주는 지표의 의미

`mean_useful_bounces`는 episode당 유효 타격 수 평균이다. 단순 contact 평균이 아니라 `keepup_env.py`의 success contract를 통과한 타격 수다. 그래서 공을 라켓에 맞히기만 한 경우와, 다음 타격 가능한 방향/높이로 보낸 경우를 구분한다.

`max_useful_bounces`는 평가 episode 중 가장 길게 이어진 결과다. 발표에서 "최대 몇 번까지 가능했는가"를 말할 때 이 값을 볼 수 있다. 다만 episode 수와 step limit에 영향을 받으므로, 무제한 평가인지 3600 step cap 평가인지 같이 밝혀야 한다.

`one_or_more`, `two_or_more`, `three_or_more`, `ten_or_more`, `thirty_or_more` rate는 threshold별 성공률이다. 평균만 보면 일부 긴 episode가 결과를 끌어올릴 수 있으므로, threshold rate와 failure count를 같이 보는 것이 안전하다.

`failure_counts`는 episode가 왜 끝났는지 보여준다. `time_limit`이 많으면 정책이 오래 유지된 것이고, `floor_contact`, `ball_out_of_bounds`, `low_apex_contact`, `robot_body_contact`가 많으면 실패 유형을 따로 분석해야 한다. 그 상세 원인은 `run_ppo_rebound_analysis.py`의 contact CSV가 더 잘 보여준다.

## 발표 때 설명 포인트

- 이 파일은 학습하지 않는다. 저장된 policy를 같은 환경 조건에서 재생하는 평가 스크립트다.
- 최종 성능 표에 넣을 수 있는 가장 단순한 지표는 `mean_useful_bounces`, `max_useful_bounces`, `thirty_or_more_useful_bounce_rate`, `failure_counts`다.
- 더 깊은 실패 원인이나 contact 물리량은 `run_ppo_rebound_analysis.py`가 담당한다.
