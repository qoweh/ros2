# `pingpong_rl2` minimal keep-up 상태 보고서

## 1. 목적

이 문서는 `pingpong_rl`의 reward/assist/curriculum 얽힘을 피하면서, `pingpong_rl2`를 최소 구조의 keep-up baseline으로 다시 세운 작업과 현재 상태를 정리한다.

핵심 목표는 아래 세 가지였다.

- 문제 원인을 reward 추가 없이 분리할 수 있는 구조 만들기
- simulation-only keep-up baseline을 다시 학습 가능한 수준으로 정리하기
- 이후 control A/B와 limited tilt branch를 명시적으로 비교할 수 있는 실험 표면 만들기

## 2. 이번까지 반영된 구조 변경

### 2.1 프로젝트 분리와 최소 baseline 재구성

`pingpong_rl2`는 아래 최소 stack으로 정리했다.

- `PingPongSim`
  - MuJoCo scene, contact trace, failure reason
- `RacketCartesianController`
  - site Jacobian 기반 EE task-space controller
- `PingPongKeepUpEnv`
  - minimal keep-up RL env
- `PingPongKeepUpGymEnv`
  - Gymnasium wrapper
- `scripts/run_ppo_learning.py`
  - PPO 학습 entrypoint

초기 원칙은 아래와 같았다.

- reward shaping은 최소화
- heuristic assist는 기본 baseline에서 섞지 않음
- hidden controller state가 있으면 observation contract에 드러냄
- 긴 A/B는 반드시 별도 run 디렉토리로 분리

### 2.2 학습/실험 표면 정리

다음 항목이 현재 코드에 반영돼 있다.

- `--reset-model` / `--resume-from` semantics
- session별 `monitor_XXX.monitor.csv` 파일 분리
- `racket_velocity` observation 복원
- 기본 action envelope의 XY/Z 비대칭화
  - 기본값 `XY +/-0.03`, `Z +/-0.04`
- versioned run-name 지원
  - `--run-name` + `--run-version`
  - 예: `ppo_keepup_position_v001`, `ppo_keepup_tilt_v001`
- 평가/뷰어가 저장된 training summary의 `env_config`를 다시 읽어서 동일 계약으로 실행

### 2.3 최신 control 확장

이번 턴에서 `position_tilt` branch를 추가했다.

- action mode: `position`
  - action = `[dx, dy, dz]`
- action mode: `position_tilt`
  - action = `[dx, dy, dz, pitch_delta, roll_delta]`

`position_tilt`에서 추가된 계약은 아래다.

- action size: `5`
- 추가 observation:
  - `racket_face_normal`
  - `target_tilt`
- 기본 tilt limit:
  - target tilt limit = `(+/-0.18, +/-0.18)`
  - per-step tilt action limit = `0.05`

이 변경은 baseline reward를 늘리지 않고 rebound direction 제어 표현력을 늘리기 위한 최소 control 확장이다.

## 3. 현재 학습 상태 요약

현재 가장 중요한 기준 모델은 아래다.

- run: `ppo_minimal_keepup`
- training budget: `2,000,000` steps
- action mode: `position`

학습 summary 기준 5-episode evaluation:

- `mean_return = -1.886`
- `mean_useful_bounces = 0.2`
- `max_useful_bounces = 1`
- failure counts:
  - `ball_out_of_bounds = 4`
  - `ball_speed_limit = 1`

하지만 5-episode summary는 노이즈가 커서, 별도 50-episode evaluation을 기준으로 판단했다.

50-episode evaluation 기준:

- `mean_return = -1.555`
- `mean_useful_bounces = 0.5`
- `max_useful_bounces = 2`
- failure counts:
  - `ball_out_of_bounds = 35`
  - `floor_contact = 7`
  - `ball_speed_limit = 6`
  - `robot_body_contact = 2`

해석은 명확하다.

- 정책은 이미 `아예 못 맞추는 상태`는 지났다.
- useful bounce를 반복적으로 한 번 이상 만드는 episode가 의미 있게 늘었다.
- 그러나 여전히 주 실패는 `ball_out_of_bounds`다.
- 즉 병목은 `contact 부재`보다 `contact 이후 rebound 방향 품질`이다.

## 4. 최신 rebound 분석 결과

현재 baseline에 대해 새로 추가한 `scripts/run_ppo_rebound_analysis.py`를 50 episode로 실행했다.

출력 artifact:

- `artifacts/ppo_runs/ppo_minimal_keepup/analysis/ppo_minimal_keepup_rebound_50ep_episodes.csv`
- `artifacts/ppo_runs/ppo_minimal_keepup/analysis/ppo_minimal_keepup_rebound_50ep_contacts.csv`
- `artifacts/ppo_runs/ppo_minimal_keepup/analysis/ppo_minimal_keepup_rebound_50ep_summary.json`

요약 수치:

- `mean_return = -1.601`
- `mean_useful_bounces = 0.54`
- `max_useful_bounces = 1`
- failure counts:
  - `ball_out_of_bounds = 38`
  - `ball_speed_limit = 5`
  - `floor_contact = 4`
  - `robot_body_contact = 3`

contact summary:

- `total_contacts = 101`
- `useful_contact_rate = 0.2673`
- 전체 contact 평균 lateral speed = `0.4673`
- 전체 contact 평균 lateral/vertical ratio = `12.3366`
- useful contact 평균 lateral speed = `0.3150`
- useful contact 평균 lateral/vertical ratio = `0.0820`

이 결과는 중요한 방향성을 준다.

- useful contact 자체는 이미 비교적 vertical rebound에 가깝다.
- 문제는 `좋은 contact를 충분히 자주 만들지 못하는 것`이다.
- 따라서 다음 우선순위는 reward 추가보다 `rebound direction을 직접 제어할 수 있는 control 표현력 확장`이다.

## 5. 현재 판단

현재 `pingpong_rl2`는 아래 정도까지 왔다.

- 최소 baseline은 실제로 학습된다.
- useful first bounce는 꽤 자주 나온다.
- 다만 장기 keep-up attractor는 아직 형성되지 않았다.
- 주 실패는 여전히 `one-bounce-then-out` 계열이다.

따라서 지금 단계의 판단은 아래다.

- reward를 여러 항으로 더 붙이는 것은 보류
- control 쪽 실험은 유지
- limited tilt branch는 충분히 시도할 가치가 있다
- A/B 비교는 반드시 별도 versioned run-name으로 진행해야 한다

## 6. 권장 실험 규칙

앞으로는 아래 패턴을 기본으로 쓴다.

- position baseline
  - `python scripts/run_ppo_learning.py --run-name ppo_keepup_position --run-version v001 --reset-model --total-timesteps 300000`
- position_tilt baseline
  - `python scripts/run_ppo_learning.py --action-mode position_tilt --run-name ppo_keepup_tilt --run-version v001 --reset-model --total-timesteps 300000`
- rebound analysis
  - `python scripts/run_ppo_rebound_analysis.py --run-name ppo_keepup_position --run-version v001 --episodes 50`

이렇게 하면 모델, monitor, summary, analysis 출력이 모두 run 디렉토리 기준으로 분리되어 비교가 쉬워진다.

## 7. 결론

현재 `pingpong_rl2`는 `최소 구조 + 관측/제어 계약 명시 + versioned A/B`라는 디버깅 가능한 상태로 정리됐다.

다음 핵심 질문은 아래 하나다.

- limited tilt가 `ball_out_of_bounds`를 줄이고 useful contact 비율을 올릴 수 있는가

이 질문에 답하기 전에는 reward를 더 복잡하게 만들지 않는 것이 맞다.