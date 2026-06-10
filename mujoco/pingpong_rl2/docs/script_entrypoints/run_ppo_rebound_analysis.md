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

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 `run_ppo_evaluation.py`보다 훨씬 자세한 평가 파일이다. episode 결과만 보는 것이 아니라, contact event가 발생한 순간마다 공과 라켓의 위치, 속도, 목표 outgoing velocity, 실제 outgoing velocity, 예상 apex, 다음 intercept, reward term, action dimension 사용량을 CSV row로 저장한다.

```text
run_ppo_rebound_analysis.py
  -> resolve_saved_model_path()
  -> resolve_env_kwargs_for_model()
  -> apply_rebound_env_overrides()
  -> PingPongKeepUpGymEnv(**env_kwargs)
  -> PPO.load(model_path)
  -> for episode
       observation = env.reset(seed)
       while not done
         action = model.predict(observation)
         observation, reward, terminated, truncated, info = env.step(action)
         if info["contact_event_during_step"]
           contact_row 생성
       episode_row 생성
  -> summarize_contacts()
  -> summarize_episode_apex_targets()
  -> summarize_episode_next_intercepts()
  -> summarize_episode_outgoing_velocities()
  -> summarize_terminal_contacts()
  -> episodes.csv / contacts.csv / summary.json 저장
```

핵심은 `info`다. `PingPongKeepUpEnv.step()`은 단순히 reward만 반환하지 않고, contact 시점의 여러 내부 물리량과 planner/controller 관련 값을 `info`에 넣는다. rebound analysis는 이 `info`를 최대한 풀어서 CSV로 남긴다.

## main()을 코드 순서대로 풀어보기

처음에는 평가와 마찬가지로 model path를 결정한다. `--model-path`가 있으면 직접 쓰고, 없으면 `--run-name`, `--run-version`으로 저장된 모델을 찾는다. 모델이 없으면 바로 `FileNotFoundError`를 낸다.

그 다음 `resolve_env_kwargs_for_model()`이 학습 당시 환경 설정을 복원한다. 여기까지는 평가 스크립트와 비슷하다. 차이는 그 다음 `apply_rebound_env_overrides(args, env_kwargs)`가 분석용 override를 추가로 적용한다는 점이다. 이 override에는 target apex, next intercept 성공 반경, nonuseful contact 종료 여부, contact-frame planner, controller gain, reward penalty 관련 옵션이 포함된다.

환경을 만든 뒤에는 분석용 상수를 미리 꺼낸다. `gravity_z`와 `gravity_magnitude`는 apex time과 ballistic trajectory 계산에 쓰인다. `strike_plane_offset`과 `strike_zone_xy_radius`는 contact 이후 공이 다음 타격 가능한 영역으로 돌아오는지 계산하는 데 쓰인다.

episode loop는 평가와 같은 모양이지만 contact event에서 분기한다.

```text
action = model.predict(observation)
observation, reward, terminated, truncated, info = env.step(action)
if contact_event_during_step:
    contact_row 생성
```

contact row를 만들 때 이 파일은 먼저 `apex_target_xy_candidates()`로 여러 apex target 후보를 만든다. 예를 들어 controller anchor, racket home, 현재 racket position, target position 같은 기준이 후보가 된다. `--apex-target`은 이 중 어떤 기준을 primary projected apex error로 볼지 정한다.

그 다음 `info`에서 contact 위치와 ball velocity를 꺼낸다. 이 값들이 모두 있으면 projected apex를 직접 계산한다.

```text
projected_apex_time = max(ball_velocity_z, 0) / gravity
projected_apex_x = contact_x + ball_velocity_x * projected_apex_time
projected_apex_y = contact_y + ball_velocity_y * projected_apex_time
projected_apex_xy_error = ||projected_apex_xy - selected_target_xy||
```

이 계산은 "공이 현재 outgoing velocity를 유지하고 중력만 받는다면 최고점 XY가 어디쯤인가"를 보는 것이다. 그래서 단순히 contact가 있었는지가 아니라, 그 contact가 다음 타격 가능한 위치로 공을 보내는지 판단하는 근거가 된다.

## contact row가 담는 정보

contact row는 이 파일의 핵심 산출물이다. 크게 다섯 묶음으로 볼 수 있다.

첫째, 성공/안정성 정보다. `success_reason`, `is_useful_contact`, `stable_cycle_observed`, `stable_cycle_count`, reward term들이 들어간다. 여기서 `success_reason == "useful_keepup_bounce"`인 contact만 유효 타격으로 본다.

둘째, action 사용량 정보다. `applied_action_0_radial`부터 `applied_action_16_tracking_vy_residual`까지 저장한다. 최종 17D action mode에서 policy가 어떤 축을 실제로 얼마나 쓰는지 그래프로 만들 때 이 값들이 필요하다.

셋째, contact-frame planner/controller 정보다. `contact_frame_vz_scale`, outgoing residual, racket residual, tilt scale, target apex z residual, strike plane z residual, tracking residual, planner contact position, planner desired velocity 같은 값이 들어간다. PPO action이 단독으로 끝나는 게 아니라 hand-coded contact-frame 계획을 어떻게 보정했는지 볼 수 있다.

넷째, 물리량 정보다. contact 위치, contact 전후 ball velocity, racket velocity, racket face normal, MuJoCo contact normal, relative velocity, lateral speed, vertical speed 등이 저장된다. 라켓이 공을 위로 보냈는지, 옆으로 너무 날렸는지, 라켓 면 방향이 접촉 normal과 잘 맞았는지 확인할 수 있다.

다섯째, ballistic 품질 정보다. desired outgoing velocity와 actual outgoing velocity error, projected apex error, next intercept reachable, easy next ball score가 들어간다. 이 값들이 "그럴듯하게 맞혔지만 다음 타격이 어려운 contact"와 "반복 타격 가능한 contact"를 구분한다.

## summary는 CSV를 어떻게 압축하나

episode loop가 끝나면 `episode_rows`와 `contact_rows`를 바탕으로 summary JSON을 만든다. 기본적으로 평균 return, 평균 useful bounce, 최대 useful bounce, threshold rate, failure counts를 계산한다. 여기에 contact row 기반 요약이 붙는다.

`summarize_contacts()`는 contact 단위 품질을 요약한다. useful contact 비율, apex error, next intercept error, outgoing velocity error 같은 contact-level 통계를 만든다.

`summarize_episode_apex_targets()`는 episode별 첫 contact/마지막 contact 기준 apex target error를 볼 수 있게 한다. `--compare-apex-targets`가 켜져 있으면 여러 target 후보별 error도 비교한다.

`summarize_episode_next_intercepts()`는 contact 이후 다음 intercept가 reachable했는지, episode별로 어떤 차이가 있었는지 요약한다. 이건 "한 번 맞히는 것"과 "반복 타격 가능한 곳으로 보내는 것"의 차이를 설명할 때 중요하다.

`summarize_terminal_contacts()`는 episode가 끝나기 직전 contact의 품질을 모아 실패 원인을 보는 데 도움을 준다. 예를 들어 마지막 contact에서 apex가 낮았는지, outgoing velocity가 틀어졌는지, next intercept가 멀었는지 확인할 수 있다.

## 평가 파일과의 차이

`run_ppo_evaluation.py`는 결과표용이다. 평균 몇 번 쳤는지, 최대 몇 번 쳤는지, 실패 이유가 무엇인지 빠르게 본다.

`run_ppo_rebound_analysis.py`는 원인 분석용이다. 공을 맞힌 순간의 물리량과 policy action을 모두 남겨서, 왜 성공했는지 또는 왜 다음 타격으로 이어지지 않았는지 설명한다. 발표에서 그래프나 표를 만들 때는 이 파일의 `contacts.csv`가 가장 중요한 원자료가 된다.

## 발표 때 설명 포인트

- `run_ppo_evaluation.py`가 “성공률/평균 성능”이라면, 이 파일은 “왜 성공하거나 실패했는가”를 설명하는 근거다.
- action dimension별 사용량, apex error, next intercept reachable rate, failure reason 분포를 시각화하기 좋다.
- 실패 사례 분석, policy behavior 비교, 반동 품질 그래프는 이 파일의 CSV를 기반으로 만들 수 있다.
