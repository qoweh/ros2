# `position_tilt` branch와 rebound analysis 보고서

## 1. 목적

이 문서는 이번 턴에서 추가한 두 가지를 정리한다.

1. versioned run-name 기반 A/B 실험 표면
2. limited pitch/roll branch와 rebound analysis 도구

목표는 `stable keep-up`에 필요한 다음 control 실험을 명시적으로 비교 가능한 형태로 만드는 것이다.

## 2. 구현 내용

### 2.1 versioned run-name 지원

`scripts/run_ppo_learning.py`는 이제 아래를 지원한다.

- `--run-name`
- `--run-version`
- `--action-mode`
- `--lateral-action-limit`
- `--vertical-action-limit`
- `--tilt-action-limit`
- `--target-tilt-limit`

run-name 결정 규칙은 아래다.

- position mode default
  - `ppo_minimal_keepup`
- position_tilt mode default
  - `ppo_position_tilt_keepup`
- smoke + position_tilt default
  - `ppo_position_tilt_smoke`
- 여기에 `--run-version v001`을 주면 최종 run-name은 `..._v001`

예시:

- `ppo_keepup_position_v001`
- `ppo_keepup_tilt_v001`
- `ppo_position_tilt_smoke_v001_smoke`

이 방식은 baseline과 A/B branch가 같은 디렉토리를 덮어쓰는 문제를 막는다.

### 2.2 evaluation/viewer의 계약 복원

`scripts/run_ppo_evaluation.py`와 `scripts/run_viewer.py`는 이제 저장된 training summary의 `env_config`를 다시 읽는다.

의미:

- tilt 모델을 evaluation할 때 action/observation shape mismatch가 줄어든다.
- lateral/vertical action limit 같은 control 설정도 같이 복원된다.
- `--run-name` + `--run-version`만으로 해당 실험의 모델과 env 계약을 다시 열 수 있다.

### 2.3 rebound analysis 스크립트 추가

새 스크립트:

- `scripts/run_ppo_rebound_analysis.py`

출력:

- per-episode CSV
- per-contact CSV
- summary JSON

핵심 기록 항목:

- contact 시 ball velocity `(x, y, z)`
- contact 시 ball lateral speed
- contact 시 lateral/vertical ratio
- contact 시 racket velocity `(x, y, z)`
- contact XY alignment error
- projected contact apex height
- useful contact 여부

이 도구는 `contact를 했는가`가 아니라 `어떤 contact였는가`를 보는 용도다.

## 3. limited tilt branch 세부 계약

`position_tilt`에서 정책이 직접 다루는 값은 아래다.

- `[dx, dy, dz]`
- `[pitch_delta, roll_delta]`

추가 observation은 아래다.

- `racket_face_normal`
- `target_tilt`

이 선택의 이유:

- tilt도 hidden controller state처럼 누적되므로 policy가 `target_tilt`를 봐야 한다.
- 실제 racket orientation 오차는 `racket_face_normal`로 볼 수 있다.
- full orientation control보다 탐색 부담이 낮다.

기본 제한값:

- `target_tilt_limit = (0.18, 0.18)`
- `tilt_action_limit = 0.05`

현재 수준에서는 이 정도가 `rebound direction residual`로서 적당한 상한이다.

## 4. 실행 검증 결과

### 4.1 tilt smoke train

검증 명령:

- `python scripts/run_ppo_learning.py --action-mode position_tilt --run-version v001_smoke --smoke --reset-model`

생성된 run:

- `ppo_position_tilt_smoke_v001_smoke`

확인 사항:

- versioned run-name 정상 생성
- training summary 정상 저장
- `env_config.action_mode = position_tilt`
- `env_config.target_tilt_limit = [0.18, 0.18]`

smoke 결과 자체는 아직 학습 의미가 없고, 표면 검증용이다.

### 4.2 tilt evaluation script 검증

검증 명령:

- `python scripts/run_ppo_evaluation.py --run-name ppo_position_tilt_smoke --run-version v001_smoke --episodes 2`

확인 사항:

- evaluation script가 versioned tilt run을 정상 해석
- 저장된 `env_config`를 복원해서 shape mismatch 없이 실행

즉 tilt branch는 이제 학습/평가/뷰어 표면까지 연결된 상태다.

## 5. 현재 baseline rebound 분석 해석

현재 `ppo_minimal_keepup` 2M 모델에 대해 50-episode rebound analysis를 수행했다.

요약:

- failure는 여전히 `ball_out_of_bounds`가 지배적이다.
- useful contact rate는 `26.7%`다.
- useful contact의 평균 lateral/vertical ratio는 `0.082`로 매우 낮다.
- 반면 전체 contact 평균 ratio는 `12.34`로 훨씬 크다.

해석:

- policy가 “좋은 vertical contact”를 전혀 못 만드는 것은 아니다.
- 문제는 그런 contact가 episode 전반에서 충분히 안정적으로 반복되지 않는다는 것이다.
- 따라서 limited tilt는 `새 보상 항 추가`보다 우선해 볼 가치가 있다.

## 6. 현재 판단

현재 단계에서 권장하는 실험 순서는 아래다.

1. position baseline versioned run
2. position_tilt versioned run
3. 각 run마다 50-episode evaluation
4. 각 run마다 rebound analysis

이후에도 `one-bounce-then-out`이 유지되면, 그때 reward를 한 항만 추가한다.

가장 작은 후보는 아래다.

- contact 시 lateral rebound를 약하게 누르는 항 1개

하지만 현재는 아직 그 단계가 아니다.

## 7. 결론

이번 턴의 핵심 성과는 성능 숫자 하나보다 `비교 가능한 실험 구조`를 만든 것이다.

- position과 tilt를 명시적으로 분리해 학습 가능
- run-name/version으로 artifact가 섞이지 않음
- evaluation/viewer가 saved env contract를 따라감
- rebound direction을 per-contact 수준에서 분석 가능

즉 다음 단계부터는 “어떤 변경이 실제로 rebound 품질을 바꿨는가”를 더 정직하게 비교할 수 있다.