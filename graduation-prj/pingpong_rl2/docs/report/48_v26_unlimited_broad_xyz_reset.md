# 48. v26 Unlimited Horizon and Broad XYZ Reset

작성일: 2026-06-04

## 목표

v25는 30 useful bounce 기준으로는 발표 가능한 수준까지 올라왔지만, 웹 서비스/시연 관점에서는 `max_episode_steps=1800`에 도달하면 실패하지 않아도 episode가 끝난다. v26의 목적은 다음 두 가지다.

- 시연/웹서비스: 실패할 때까지 계속 실행되는 무제한 episode 지원
- 학습: v25 best policy를 유지하면서, 타격 가능한 범위 안에서 시작 XY/Z를 더 다양화

## 구현 내용

- `max_episode_steps <= 0` 또는 `None`을 시간 제한 없음으로 해석하도록 `PingPongKeepUpEnv`를 수정했다.
- `reset_xy_sampling="disk"`를 추가했다. 기존 `square`는 x/y 각각 uniform이고, `disk`는 반경 안에서 0~360도 방향을 균일하게 샘플링한다.
- `reset_ball_height_bounds=(low, high)`를 추가했다. 기존 `ball_height +/- reset_ball_height_range` 대신 라켓 위 기준 z 높이를 직접 샘플링할 수 있다.
- 저장된 `env_config` 복원, viewer, headless evaluation, rebound analysis가 새 reset 옵션을 받을 수 있게 연결했다.
- 무제한 env를 평가/분석할 때 프로그램이 끝나지 않는 문제를 막기 위해 evaluation/rebound analysis에는 기본 3600 step safety cap을 추가했다.

## v26 preset

새 preset: `contact_frame_self_rally_v26_unlimited_broad_xyz`

핵심 설정:

- `max_episode_steps=0`
- `reset_xy_sampling="disk"`
- `reset_xy_range=0.075`
- `reset_ball_height_bounds=(0.24, 0.48)`
- `reset_velocity_xy_range=0.025`
- `reset_velocity_z_range=(-0.08, 0.02)`
- checkpoint/final deterministic evaluation safety cap: `evaluation_step_limit=3600`

범위는 일부러 보수적으로 잡았다. v25의 안정성을 유지하면서 일반화를 넓히는 첫 단계이며, 성공하면 다음 버전에서 `xy 0.10m` 또는 더 넓은 z 범위로 키우는 식이 안전하다.

## 학습 명령

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v26_unlimited_broad_xyz.json
```

이 config는 `pmk_cf_self_rally_v25_best_model.zip`에서 `pmk_cf_self_rally_v26`으로 이어서 500k 학습한다.

## 확인 명령

시연처럼 실패할 때까지 viewer로 실행:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_viewer.py \
  --run-name pmk_cf_self_rally \
  --run-version v26 \
  --best-model \
  --episodes 1 \
  --max-episode-steps 0
```

학습 후 rebound 분석:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --run-name pmk_cf_self_rally \
  --run-version v26 \
  --episodes 100
```

무제한 모델에서 분석이 너무 오래 걸리면 기본 3600 step에서 `time_limit`으로 끊긴다. 더 길게 보고 싶으면 `--episode-step-limit 7200`, 정말 제한 없이 분석하고 싶으면 `--episode-step-limit 0`을 사용한다.

## 검증

- `py_compile`: `keepup_env.py`, `ppo_runs.py`, learning/evaluation/rebound/viewer 스크립트 통과
- v26 config 0-step smoke: preset/config/resume/env_config 정상 확인
- `python -m unittest tests.test_keepup_env -v`: 125 tests passed
- `python -m unittest discover -s tests -v`: 143 tests passed

## 해석

“제한 없이 학습”은 가능하지만, 평가/분석까지 무제한이면 좋은 정책일수록 스크립트가 끝나지 않는다. 그래서 학습 env는 실패 전까지 계속 가게 만들고, deterministic evaluation에는 별도의 safety cap을 둔 구조가 가장 현실적이다.

v26은 v25의 30 bounce 안정성 위에 일반화 과제를 얹는 실험이다. 성능이 바로 떨어질 수 있는데, 그 경우 실패가 아니라 curriculum이 넓어진 것이다. v26 결과를 볼 때는 기존 v25 조건의 bounce 수와 broad xyz 조건의 bounce 수를 따로 비교해야 한다.
