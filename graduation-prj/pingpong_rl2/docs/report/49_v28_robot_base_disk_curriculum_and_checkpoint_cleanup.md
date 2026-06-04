# v28 robot-base disk curriculum and checkpoint cleanup

Date: 2026-06-04

## 1. 기준 변경

이번 변경의 핵심은 reset 원반의 중심을 `탁구채 중심`이 아니라 `로봇팔이 바닥에 고정된 베이스 중심`으로 바꾼 것이다.

기존 v26/v27:

- `reset_xy_origin = racket`
- 공 시작 XY = `racket_position[:2] + sampled_offset`
- 즉, 12cm disk는 탁구채 홈 위치 주변 12cm였다.

v28:

- `reset_xy_origin = robot_base`
- `reset_robot_base_xy = (0.0, 0.0)`
- 공 시작 XY = `robot_base_xy + sampled_offset`
- 즉, disk는 월드 원점에 있는 로봇 베이스 중심 기준이다.

현재 홈 탁구채 중심은 대략 `[0.5545, 0.1650]m`이고, 베이스 기준 거리는 약 `0.579m`다. 따라서 기존의 "racket 중심 12cm"와 "base 중심 12cm"는 완전히 다른 분포다.

## 2. v27_fast 상태

사용자가 학습한 모델:

`artifacts/ppo_runs/pmk_cf_self_rally_v27_fast/pmk_cf_self_rally_v27_fast_best_model.zip`

이전 확인 기준:

- final 20 episode eval: mean useful bounce 약 `16.05`, max `31`
- best interim model: mean useful bounce 약 `24.88`, max `34`
- v27_fast는 12cm 수준의 racket-centered/random disk에는 꽤 적응했지만, robot-base 360도 disk를 직접 의미하지는 않는다.

## 3. 구현 변경

추가된 환경 인자:

- `reset_xy_origin`: `"racket"` 또는 `"robot_base"`
- `reset_robot_base_xy`: robot-base origin의 world XY
- `ball_x_bounds`, `ball_y_bounds`: 넓은 360도 reset에서 시작하자마자 `ball_out_of_bounds`가 나지 않도록 실패 판정 영역 조정

추가된 v28 preset:

- `contact_frame_self_rally_v28_robot_base_disk_curriculum`
- `reset_xy_origin = robot_base`
- `reset_xy_sampling = disk`
- `reset_xy_curriculum_start = 0.12`
- `reset_xy_curriculum_end = 0.68`
- `reset_xy_curriculum_fraction = 0.90`
- `target_offset_low = (-1.25, -0.85, -0.04)`
- `target_offset_high = (0.20, 0.55, 0.12)`
- `ball_x_bounds = (-0.85, 0.85)`
- `ball_y_bounds = (-0.85, 0.85)`

중요한 추가 수정:

- `reset_xy_origin == robot_base`일 때 pre-contact target guard가 기존 홈 위치 주변 14cm 제한으로 다시 잘리지 않게 했다.
- 기존 제한이 남아 있으면 공은 base disk에 생성되지만 로봇은 여전히 홈 탁구채 주변만 따라가서 학습이 성립하지 않는다.

## 4. Checkpoint cleanup

`run_ppo_learning.py`에서 중간 checkpoint 저장, 중간 checkpoint evaluation, best checkpoint 선택, early-stop 경로를 제거했다.

현재 학습 흐름:

1. 필요하면 `--resume-from` 모델을 로드한다.
2. `model.learn(total_timesteps=...)`를 한 번 수행한다.
3. 마지막 모델만 `<run_name>_model.zip`으로 저장한다.
4. final evaluation과 training summary만 저장한다.

active code/config/test 영역에서 checkpoint 관련 키와 코드는 제거했다. 과거 report 문서의 checkpoint 언급은 당시 실험 기록이므로 그대로 둔다.

## 5. v28 학습 명령

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_learning.py \
  --config-file configs/pmk_cf_self_rally_v28_robot_base_disk_curriculum.json
```

학습이 끝나면 final model은 아래에 저장된다.

```text
artifacts/ppo_runs/pmk_cf_self_rally_v28_robot_base_disk/pmk_cf_self_rally_v28_robot_base_disk_model.zip
```

중간 checkpoint나 best model은 새 코드에서 생성되지 않는다.

## 6. Viewer command

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2

PYTHONPATH=src conda run -n mujoco_env python scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/pmk_cf_self_rally_v28_robot_base_disk/pmk_cf_self_rally_v28_robot_base_disk_model.zip
```

모델 summary에 v28 env config가 저장되므로 viewer는 robot-base disk 설정을 자동 복원한다.

## 7. 주의점

v28은 쉬운 fine-tune이 아니라 distribution 자체를 크게 바꾸는 실험이다. 시작 위치가 기존 탁구채 주변에서 로봇 베이스 중심 360도 원반으로 바뀌므로, 처음에는 성능이 떨어지는 것이 정상이다.

만약 v28이 너무 불안정하면 다음 staged curriculum 후보가 더 현실적이다.

- radius `0.12 -> 0.35`
- radius `0.35 -> 0.50`
- radius `0.50 -> 0.68`

다만 이번 config는 사용자가 자리를 비운 동안 한 번의 run으로 최대한 넓혀 보려는 목적이라 `0.12 -> 0.68`을 한 번에 넣었다.

## 8. Refactor candidates

정리 우선순위:

1. `run_ppo_learning.py` preset들을 JSON/YAML config registry로 분리
2. 오래된 bootstrap candidate preset 정리 또는 archival config로 이동
3. training/evaluation/reporting 함수를 별도 모듈로 분리
4. action-mode별 reward/config defaults를 dataclass로 묶기
5. historical report는 유지하되, 현재 발표용 문서는 `docs/report/46`, `47`, `48`, `49` 중심으로 별도 index 만들기

이번 변경에서는 학습 안정성에 직접 필요한 reset origin, bounds, pre-contact guard, checkpoint cleanup만 처리했다.
