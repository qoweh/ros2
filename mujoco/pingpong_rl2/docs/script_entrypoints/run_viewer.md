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

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 사람이 눈으로 움직임을 확인하기 위한 렌더링 entrypoint다. 평가/분석처럼 CSV를 주로 만드는 것이 아니라 MuJoCo passive viewer를 띄우고, 매 frame마다 action을 골라 `env.step(action)`을 실행한다.

```text
run_viewer.py
  -> mode 결정: policy / heuristic / zero_action
  -> 필요하면 model path와 env kwargs 복원
  -> CLI override를 env_kwargs에 반영
  -> PingPongKeepUpGymEnv(**env_kwargs)
  -> action source 준비
       policy    -> PPO.load()
       heuristic -> HeuristicKeepUpPolicy()
       zero      -> np.zeros(action_space.shape)
  -> mujoco.viewer.launch_passive()
  -> viewer loop
       action 선택
       env.step(action)
       viewer.sync()
       episode 종료 시 요약 출력
```

policy mode에서는 저장된 PPO policy가 observation을 보고 action을 낸다. 이 action은 평가 파일과 동일하게 `PingPongKeepUpEnv.step()`으로 들어가 residual action으로 해석되고, Cartesian controller를 거쳐 MuJoCo actuator target으로 변환된다.

heuristic mode에서는 neural network를 쓰지 않는다. `HeuristicKeepUpPolicy.predict(env.base_env)`가 현재 환경 내부 상태를 읽고 hand-coded residual action을 만든다. 모델 경로 없이 heuristic만 볼 때는 `action_mode="position_strike"`와 task phase/contact/next intercept observation을 명시적으로 켠다. 이건 저장된 training summary가 없기 때문에 viewer가 직접 기본 구성을 잡아주는 것이다.

zero_action mode에서는 action space와 같은 크기의 0 벡터를 넣는다. 이 모드는 정책 성능을 보려는 것이 아니라, 환경의 기본 planner/중력/접촉/termination이 어떻게 동작하는지 비교하는 sanity check에 가깝다.

## viewer loop에서 실제로 일어나는 일

viewer가 열린 뒤에는 `frame_sleep = sim.model.opt.timestep * sim.n_substeps`로 물리 control step에 맞춘 sleep 시간을 잡는다. 매 loop마다 action source에 따라 action을 하나 고르고 `env.step(action)`을 호출한다. 그 다음 `viewer.sync()`로 현재 MuJoCo state를 화면에 반영한다.

episode가 `terminated` 또는 `truncated`가 되면 console에 return, contact count, useful bounce count, failure reason을 출력한다. 아직 지정한 episode 수가 남아 있으면 env를 reset하고 다음 episode를 viewer에서 계속 보여준다. 마지막 episode가 끝나면 `hold_final_seconds`만큼 마지막 장면을 유지한다.

## 평가/분석 파일과의 차이

`run_ppo_evaluation.py`는 수치 요약을 빠르게 얻기 위한 headless 평가다. `run_ppo_rebound_analysis.py`는 contact별 CSV를 만드는 상세 분석이다. `run_viewer.py`는 같은 policy라도 실제 라켓이 어느 위치로 가고, 공이 어떻게 튀고, episode가 어떤 느낌으로 끝나는지 눈으로 확인하는 파일이다.

그래서 발표용으로는 최종 모델의 동작 영상이나 heuristic baseline 비교 영상을 만들 때 이 파일이 가장 직접적이다.

## 발표 때 설명 포인트

- 실제 데모/GIF 후보는 이 파일이 가장 직접적이다.
- policy mode는 최종 모델 행동 시각화, heuristic mode는 baseline 행동 시각화, zero_action은 환경/중력/접촉 sanity 비교에 쓸 수 있다.
