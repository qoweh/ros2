# pingpong_rl2 Python 코드 맵

이 문서는 `pingpong_rl2` 디렉토리 안의 모든 Python 파일이 무엇을 담당하는지, 그리고 코드가 어떤 순서로 흘러가는지 발표 준비용으로 정리한 것이다. `pingpong_rl3`는 범위에서 제외한다.

## 리팩토링 후 구조 요약

기존에는 `scripts/run_ppo_learning.py` 하나에 학습 프리셋, CLI 파싱, config 적용, 환경 kwargs 생성, reset curriculum, heuristic bootstrap, PPO 초기화, 학습, 평가가 모두 들어 있었다. 지금은 실행 스크립트는 학습 순서만 남기고, 세부 책임은 `src/pingpong_rl2/training/`으로 분리했다.

기존 `scripts/run_ppo_rebound_analysis.py`도 CSV 저장, ballistic 계산, contact quality 계산, 요약 통계, env override가 한 파일에 붙어 있었다. 지금은 분석 보조 로직을 `src/pingpong_rl2/analysis/`로 분리했다.

`src/pingpong_rl2/envs/keepup_env.py`는 여전히 핵심 환경 클래스라 크지만, 상태/행동 모드 정의와 observation layout 정의는 `envs/action_modes.py`, `envs/observation_layout.py`로 분리했다.

## 전체 코드 흐름

### 학습 흐름

```text
scripts/run_ppo_learning.py
  -> training/cli_config.py: CLI/config 파일/--set 파싱
  -> training/env_config.py: preset 적용, env_kwargs 생성
  -> envs/gym_env.py: Gymnasium 호환 wrapper 생성
  -> envs/keepup_env.py: MuJoCo 기반 keep-up 환경 생성
  -> training/vector_env.py: SB3용 vector env adapter 생성
  -> training/curriculum.py: reset 분포 curriculum callback 적용
  -> training/bootstrap.py: 선택 시 heuristic policy로 actor 사전학습
  -> stable_baselines3.PPO.learn()
  -> training/evaluation.py: deterministic evaluation
  -> artifacts/ppo_runs/<run_name>/: model.zip, monitor.csv, training_summary.json 저장
```

### 환경 step 흐름

```text
PingPongKeepUpGymEnv.step(action)
  -> PingPongKeepUpEnv.step(action)
    -> action_mode별 목표 위치/tilt/velocity 해석
    -> RacketCartesianController.compute_joint_targets()
    -> PingPongSim.step()
    -> contact trace, failure reason, success reason 계산
    -> reward_terms 계산
    -> observation() 생성
  -> Gymnasium step tuple 반환
```

### 분석/평가 흐름

```text
scripts/run_ppo_evaluation.py 또는 scripts/run_ppo_rebound_analysis.py
  -> utils/ppo_runs.py: 모델 경로와 env_config 복원
  -> PingPongKeepUpGymEnv 생성
  -> PPO.load()
  -> episode loop 실행
  -> info/reward/contact 데이터를 집계
  -> CSV/JSON summary 저장
```

### 시각화 흐름

```text
docs/rl_presentation_pack/scripts/generate_visuals.py
  -> artifacts/ppo_runs, docs/rl_presentation_pack/data 읽기
  -> timeline, failure mode, long target, action usage, ablation, observation/action diagram 생성
  -> docs/rl_presentation_pack/assets 저장
```

## 패키지 핵심 파일

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `src/pingpong_rl2/__init__.py` | 패키지 최상위 export. `PingPongSim`, `PingPongKeepUpEnv`, `PingPongKeepUpGymEnv`를 노출한다. | 외부 코드가 `from pingpong_rl2 import ...`를 할 때 `envs` 패키지의 주요 클래스를 가져온다. |
| `src/pingpong_rl2/defaults.py` | 기본 학습값, 기본 모델/run 이름, 기본 reset 범위, 기본 reward/penalty 상수를 둔다. | 학습 스크립트와 유틸에서 기본 run 이름과 PPO hyperparameter를 참조한다. |

## 환경 파일

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `src/pingpong_rl2/envs/__init__.py` | 환경 관련 class export. | `PingPongSim`, `PingPongKeepUpEnv`, `PingPongKeepUpGymEnv`를 한 번에 import할 수 있게 한다. |
| `src/pingpong_rl2/envs/action_modes.py` | action mode와 관련 mode 그룹을 정의한다. | `keepup_env.py`가 이 상수를 import해서 action size, action 해석, success contract 조건을 결정한다. |
| `src/pingpong_rl2/envs/observation_layout.py` | observation 구성 요소와 slice layout을 만든다. | `build_observation_layout()`이 action mode와 observation flag를 받아 component 목록, slice map, 전체 observation 크기를 반환한다. |
| `src/pingpong_rl2/envs/pingpong_sim.py` | MuJoCo scene 로딩, joint/ball/racket 상태 접근, simulation step, contact trace 수집을 담당한다. | `PingPongSim` 생성 시 XML scene을 로드하고, env reset/step에서 ball spawn, joint target 적용, contact 검사를 수행한다. |
| `src/pingpong_rl2/envs/keepup_env.py` | 강화학습 환경의 본체. reset, observation, step, reward, success/failure 판정, contact-frame controller 보조 로직을 포함한다. | `__init__`에서 환경 파라미터 검증과 action/observation 공간을 구성한다. `reset()`은 공 위치/속도/target을 초기화한다. `step()`은 action을 target position/tilt/velocity로 해석하고 controller와 sim을 진행한 뒤 reward/info/termination을 만든다. |
| `src/pingpong_rl2/envs/gym_env.py` | `PingPongKeepUpEnv`를 Gymnasium API로 감싼다. | SB3가 요구하는 `reset`, `step`, `action_space`, `observation_space` 형식을 제공하고 vector env에서 안전한 info 형태로 변환한다. |

## 컨트롤러 파일

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `src/pingpong_rl2/controllers/__init__.py` | controller export. | `RacketCartesianController`, `HeuristicKeepUpPolicy`를 노출한다. |
| `src/pingpong_rl2/controllers/ee_pose_controller.py` | racket end-effector 목표 위치/tilt/velocity를 Franka joint target으로 바꾼다. | target position/tilt/velocity를 저장하고, Jacobian 기반 position/orientation/velocity error를 joint delta로 변환한다. nullspace posture와 body clearance 보정도 여기서 더한다. |
| `src/pingpong_rl2/controllers/heuristic_keepup.py` | PPO 사전학습이나 진단용 heuristic policy. | env phase를 읽고 position residual, tilt residual, lift residual, tracking residual을 action vector로 채운다. contact-frame action mode도 지원한다. |

## 학습 모듈

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `src/pingpong_rl2/training/__init__.py` | vector env helper export. | `make_sb3_async_vector_env` 등을 노출한다. |
| `src/pingpong_rl2/training/presets.py` | 학습 preset과 preset이 관리하는 기본 인자 목록. | `contact_frame_self_rally_v...` 계열 preset을 정의하고 `cli_config.py`, `env_config.py`에서 사용한다. |
| `src/pingpong_rl2/training/cli_config.py` | CLI 인자 파싱, JSON config 로딩, `--set KEY=VALUE` override 처리. | parser를 만들고, 명시 CLI 인자를 추적한 뒤 config 파일 값과 override를 `argparse.Namespace`에 반영한다. |
| `src/pingpong_rl2/training/env_config.py` | preset 적용, tilt profile 해석, `env_kwargs` 생성. | `apply_env_preset()`이 preset과 CLI 충돌을 검사하고, `env_kwargs_from_args()`가 `PingPongKeepUpGymEnv(**env_kwargs)`에 들어갈 dict를 만든다. |
| `src/pingpong_rl2/training/run_paths.py` | 학습 run directory, model path, resume/new training 판단. | run name을 바탕으로 artifact directory를 만들고 기존 model zip이 있으면 resume 대상으로 잡는다. |
| `src/pingpong_rl2/training/vector_env.py` | Gymnasium vector env를 SB3 VecEnv 계약에 맞게 감싼다. | `make_gym_vector_env()`가 Async/SyncVectorEnv를 만들고, `SB3AsyncVectorEnvAdapter`가 SB3의 reset/step/done/info 형식으로 변환한다. |
| `src/pingpong_rl2/training/curriculum.py` | reset 분포 curriculum callback. | 학습 progress에 따라 reset XY 범위, velocity 범위, spin 범위를 보간하고 vector env의 `set_reset_distribution()`을 호출한다. |
| `src/pingpong_rl2/training/bootstrap.py` | heuristic policy로 dataset을 모으고 PPO actor를 MSE로 사전학습한다. | heuristic episode를 실행해 observation/action sample을 모은 뒤 actor output과 heuristic action의 MSE loss로 policy parameter를 업데이트한다. |
| `src/pingpong_rl2/training/policy_init.py` | PPO policy 초기화 보조 함수. | action limit 비율로 per-action log_std를 계산하고, 필요하면 PPO policy의 `log_std`를 직접 초기화한다. `learn_model()`은 timestep 0 처리까지 포함한 얇은 wrapper다. |
| `src/pingpong_rl2/training/evaluation.py` | 학습 후 deterministic evaluation. | model을 deterministic으로 실행하고 return, useful bounce, stable cycle, failure reason 비율을 summary dict로 만든다. |

## 분석 모듈

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `src/pingpong_rl2/analysis/__init__.py` | 분석 패키지 marker. | 분석 helper 모듈들이 `pingpong_rl2.analysis.*`로 import될 수 있게 한다. |
| `src/pingpong_rl2/analysis/csv_io.py` | CSV 저장 helper. | row dict들의 key 순서를 수집하고 `csv.DictWriter`로 저장한다. |
| `src/pingpong_rl2/analysis/rebound_env.py` | rebound analysis용 env override 정리. | CLI args에서 값이 있는 direct field, tuple field, true/false flag를 `env_kwargs`에 반영한다. |
| `src/pingpong_rl2/analysis/rebound_metrics.py` | contact 이후 ballistic/quality metric 계산. | contact position/velocity로 다음 descending intercept, easy-next-ball score, relative/tangential contact speed, apex target 후보를 계산한다. |
| `src/pingpong_rl2/analysis/rebound_summary.py` | contact/episode CSV row들을 aggregate summary로 바꾼다. | contact rows에서 useful/stable/apex/velocity/penalty 통계를 만들고, episode별 first-contact metric gap과 terminal contact summary를 계산한다. |

## 유틸 파일

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `src/pingpong_rl2/utils/__init__.py` | path/run-name 유틸 export. | `paths.py`, `ppo_runs.py`의 주요 함수를 한 곳에서 import하게 한다. |
| `src/pingpong_rl2/utils/paths.py` | 프로젝트 root, artifact root, 입력/출력 path 해석. | 상대 경로를 `pingpong_rl2` root 기준으로 해석하고 output path 부모 디렉토리를 만든다. |
| `src/pingpong_rl2/utils/ppo_runs.py` | PPO run name/model path/env_config 복원 유틸. | action mode별 기본 run name을 고르고, model zip에서 run name을 추론하며, training summary JSON에서 env_config를 불러온다. |

## 실행 스크립트

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `scripts/run_ppo_learning.py` | PPO 학습 실행 진입점. | args 파싱, preset/config 적용, env/vector env/model 생성, bootstrap, learn, evaluation, summary 저장 순서만 담당한다. 세부 로직은 `training/` 모듈로 분리됐다. |
| `scripts/run_ppo_evaluation.py` | 저장된 PPO model을 간단히 평가한다. | model/env_config를 복원하고 episode loop를 돌려 return/useful bounce/failure count를 출력한다. |
| `scripts/run_ppo_rebound_analysis.py` | 저장된 PPO model의 contact/rebound quality를 깊게 분석한다. | model/env 복원, episode loop, contact row 생성, episode/contact CSV 저장, summary JSON 저장을 수행한다. metric/summary/env override는 `analysis/` 모듈로 분리됐다. |
| `scripts/expand_ppo_action_space.py` | 기존 PPO model을 더 큰 action space 모델로 이식한다. | source/target env를 만들고 policy network 앞부분 weight를 복사해 action dimension 확장 transfer를 준비한다. |
| `scripts/run_heuristic_keepup_diagnostic.py` | heuristic policy를 실행해 contact와 episode 진단 CSV를 만든다. | env kwargs를 만들고 heuristic action으로 episode를 실행하며 contact event와 terminal info를 기록한다. |
| `scripts/run_contact_feasibility_map.py` | reset 조건/target 조건 조합별 contact feasibility를 평가한다. | 여러 XY/height/velocity 설정을 sweep하고 heuristic 또는 env behavior 기반 성공 가능성 row를 만든다. |
| `scripts/run_viewer.py` | MuJoCo viewer에서 정책 또는 heuristic을 눈으로 확인한다. | model 또는 heuristic action을 선택해 env를 step하면서 viewer에 scene을 렌더링한다. |
| `scripts/run_bounce_sanity.py` | 순수 simulation bounce/contact sanity check. | `PingPongSim`에서 공을 떨어뜨리고 contact trace와 bounce 결과를 확인한다. |
| `scripts/run_material_sanity.py` | MuJoCo geom/material/contact 설정 sanity check. | geom 속성 요약과 정적 racket drop 실험을 통해 contact 물성을 확인한다. |
| `scripts/benchmark_vector_env.py` | vector env 성능 측정. | 여러 env를 만들고 reset/step throughput을 측정한다. |

## 발표 자료 시각화 스크립트

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `docs/rl_presentation_pack/scripts/generate_visuals.py` | 발표용 그래프와 CSV를 생성한다. | training/eval summary와 presentation data를 읽고 timeline, failure mode, long target, apex distribution, action usage, ablation, observation/action diagram, monitor curve를 `assets/`에 저장한다. |

## 테스트 파일

| 파일 | 용도 | 코드 흐름 |
|---|---|---|
| `tests/conftest.py` | pytest 실행 시 `src` 경로를 import path에 넣는다. | 테스트가 설치 없이 local package를 import하게 한다. |
| `tests/test_scene_load.py` | MuJoCo scene 로딩과 contact trace 필드 검증. | `PingPongSim`을 만들고 필수 body/geom/site와 ball spawn/contact trace를 확인한다. |
| `tests/test_vector_env.py` | vector env와 SB3 adapter 계약 검증. | async/sync vector env reset shape, done/reset 처리, SB3 step return 형식을 확인한다. |
| `tests/test_ppo_runs.py` | PPO run path/config/log_std/curriculum 유틸 검증. | model summary에서 env_config를 찾는지, action std scaling과 reset curriculum 보간이 맞는지 확인한다. |
| `tests/test_keepup_contract_features.py` | task phase, next intercept, useful contact contract, heuristic policy의 핵심 feature 테스트. | observation flag, reachable next intercept, success requirement, nonuseful termination, heuristic action clipping을 검증한다. |
| `tests/test_keepup_env.py` | keep-up env 전체 동작에 대한 가장 큰 regression suite. | reset 분포, observation slice, action mode별 action space, tilt/strike/contact-frame 보조 로직, reward term, success/failure 조건, Gym wrapper space를 폭넓게 검증한다. |

## 핵심 질문 대비 포인트

1. 환경의 중심은 `PingPongKeepUpEnv`다. MuJoCo state를 observation으로 만들고, action을 racket target으로 바꾸며, contact 이후 success/reward/failure를 계산한다.
2. Gym/SB3 연결은 `PingPongKeepUpGymEnv`와 `SB3AsyncVectorEnvAdapter`가 담당한다.
3. PPO 학습 자체는 SB3가 수행하고, 프로젝트 코드는 reset curriculum, preset, bootstrap, 평가 summary를 제공한다.
4. 최종 17D action mode는 `envs/action_modes.py`의 `position_contact_frame_velocity_tilt_lateral_apex_tracking_residual` 계열 정의와 `keepup_env.py`의 contact-frame helper들에서 해석된다.
5. Q-value heatmap은 PPO 특성상 직접 Q-value가 아니라 critic value function heatmap으로 설명해야 한다.

## 검증 상태

`python3 -m compileall -q pingpong_rl2/src/pingpong_rl2 pingpong_rl2/scripts pingpong_rl2/tests`는 통과했다.

현재 시스템 Python에는 `pytest`, `gymnasium`, `mujoco`, `numpy`, `stable_baselines3`, `torch`가 설치되어 있지 않아 실행형 unit test는 import 단계에서 중단된다. `pingpong_rl2/pyproject.toml` 기준 필요한 의존성은 `gymnasium`, `mujoco`, `numpy`, `stable-baselines3`이다.
