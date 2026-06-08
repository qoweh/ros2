# run_ppo_rebound_analysis.py

## 한 줄 역할

`scripts/run_ppo_rebound_analysis.py`는 저장된 PPO policy를 실행하면서 contact event마다 반동 품질을 CSV로 남기는 상세 분석 entrypoint다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_ppo_rebound_analysis.py \
  --run-name keep1 \
  --run-version v39_17d_mid_curriculum_fixed \
  --episodes 50 \
  --analysis-name keep1_v39_rebound_50ep
```

## 코드 흐름

1. 모델과 환경 설정을 복원한다.
   - `resolve_saved_model_path()`가 모델 zip을 찾는다.
   - `resolve_env_kwargs_for_model()`이 training summary에서 env kwargs를 복원한다.
   - reset, contact-frame, reward penalty, target, planner 관련 CLI 인자로 분석용 override를 추가할 수 있다.

2. rebound override를 적용한다.
   - `apply_rebound_env_overrides(args, env_kwargs)`가 반동 분석에 필요한 환경 옵션을 덧씌운다.
   - 예를 들어 target apex, next intercept success radius, nonuseful contact 종료 여부 같은 조건을 평가 시점에 바꿀 수 있다.

3. 단일 평가 env와 PPO policy를 준비한다.
   - `PingPongKeepUpGymEnv(**env_kwargs)`를 만든다.
   - 무제한 horizon이면 `_UNLIMITED_ANALYSIS_STEP_LIMIT`로 safety cap을 둔다.
   - `PPO.load(str(model_path))`로 policy를 로드한다.

4. episode loop를 실행한다.
   - 각 step에서 `model.predict()`로 action을 받고 `env.step(action)`을 실행한다.
   - `info["contact_event_during_step"]`가 true인 순간만 contact row를 만든다.
   - episode가 끝나면 episode row를 만든다.

5. contact row에 기록하는 핵심 데이터
   - contact 위치와 contact 시점 ball velocity
   - desired/actual outgoing velocity와 error
   - projected apex 위치와 apex target error
   - next intercept reachable/error
   - racket velocity, racket face normal, contact normal alignment
   - reward term별 값
   - applied action 0번부터 16번까지의 실제 값
   - contact-frame planner, tilt, tracking residual 관련 내부 info

6. summary를 만든다.
   - `summarize_contacts()`
   - `summarize_episode_apex_targets()`
   - `summarize_episode_next_intercepts()`
   - `summarize_episode_outgoing_velocities()`
   - `summarize_terminal_contacts()`

7. 산출물을 저장한다.
   - `<analysis_name>_episodes.csv`
   - `<analysis_name>_contacts.csv`
   - `<analysis_name>_summary.json`

## 주요 호출 관계

```text
run_ppo_rebound_analysis.py
  -> utils/ppo_runs.py             # 모델/env 설정 복원
  -> analysis/rebound_env.py       # 분석용 env override
  -> envs/gym_env.py               # 평가 env
  -> stable_baselines3.PPO         # 저장 policy
  -> analysis/rebound_metrics.py   # next intercept/contact quality 계산
  -> analysis/rebound_summary.py   # CSV rows -> summary JSON
  -> analysis/csv_io.py            # CSV 저장
```

## 발표 때 설명 포인트

- `run_ppo_evaluation.py`가 “성공률/평균 성능”이라면, 이 파일은 “왜 성공하거나 실패했는가”를 설명하는 근거다.
- action dimension별 사용량, apex error, next intercept reachable rate, failure reason 분포를 시각화하기 좋다.
- 실패 사례 분석, policy behavior 비교, 반동 품질 그래프는 이 파일의 CSV를 기반으로 만들 수 있다.
