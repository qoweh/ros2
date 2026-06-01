# 28. outward racket scene + state-dependent tilt 정리

작성일: 2026-06-01

## 목적

`pmk_cf_self_rally_v3`에서 남은 핵심 문제는 다음 세 가지로 정리된다.

- 공이 라켓/로봇팔 안쪽으로 들어오면 `link5/link6/hand`에 맞고 episode가 깨진다.
- 공을 너무 낮게 띄우는 contact가 남아 있어 다음 타격 주기가 짧고 불안정하다.
- tilt가 없던 것이 아니라, 기존 self-rally preset이 과거 candidate에서 물려받은 고정 pitch 성분 때문에 상태 기반 보정보다 한쪽 기울기 포화가 먼저 생겼다.

이번 작업은 값을 조금씩 바꾸는 것이 아니라, 최종 목표인 self-rally에 맞춰 geometry와 tilt 의미를 다시 정렬하는 것이다.

## viewer 자세 해석

`python -m mujoco.viewer --mjcf assets/scene.xml`에서 보이는 팔을 위로 편 자세는 실제 학습 reset 자세가 아니다.

- raw XML 기본값: `qpos[:7] = [0, 0, 0, 0, 0, 0, 0]`
- 실제 환경 reset: `assets/franka/panda.xml`의 `home` keyframe
- `PingPongSim.reset()`은 `mujoco.mj_resetDataKeyframe(..., 0)`을 호출한다.

따라서 정적 viewer에서 보이는 첫 화면으로 학습 시작 자세를 판단하면 안 된다. 실제 reset/정책 움직임은 `scripts/run_viewer.py` 또는 환경을 직접 생성해서 확인해야 한다.

## geometry 변경

기존 scene은 그대로 두고 비교용 scene을 추가했다.

- 새 파일: `assets/franka/panda_racket_outward.xml`
- 새 파일: `assets/scene_racket_outward.xml`
- 변경: 라켓 head/site 중심을 local x 기준 `0.12m -> 0.18m`로 이동
- 효과: reset 기준 hand와 racket center의 XY 거리 `0.125m -> 0.185m`

라켓을 반대편으로 뒤집지는 않았다. 현재 문제는 라켓 방향 자체보다 공이 안쪽으로 drift할 때 팔 body와 가까워지는 구조가 더 직접적이다. 반대편으로 단순히 뒤집으면 오히려 공 목표점이 팔 위/안쪽으로 들어갈 수 있다. 그래서 먼저 라켓을 손에서 더 바깥쪽으로 빼는 variant를 만들었다.

## scene_path 연결

이제 환경이 `scene_path`를 받을 수 있다.

- `PingPongKeepUpEnv(scene_path=...)`
- `PingPongSim(scene_path=...)`
- training summary의 `env_config.scene_path`에 실제 사용 scene 저장
- `run_viewer.py`, `run_ppo_rebound_analysis.py`는 저장된 model summary에서 scene을 자동 복원
- 필요하면 `--scene-path assets/scene_racket_outward.xml`로 수동 override 가능

이렇게 해야 새 geometry로 학습한 모델을 기존 scene으로 잘못 viewer/eval하는 일을 막을 수 있다.

## self-rally preset 변경

`contact_frame_self_rally_candidate`는 이제 새 outward scene을 기본으로 사용한다.

고정 tilt 성분을 제거했다.

- `strike_tilt_ramp_pitch = None`
- `followup_strike_target_tilt = None`
- `contact_frame_base_tilt_residual = None`

대신 상태 기반 tilt만 남겼다.

- `contact_frame_trajectory_tilt_gain = 1.0`
- `contact_frame_trajectory_tilt_limit = (0.06, 0.06)`
- `contact_frame_centering_tilt_limit = (0.04, 0.05)`
- `target_tilt_limit = (0.12, 0.12)`
- `tilt_action_limit = 0.008`

의도는 다음과 같다.

- planner/primitive가 다음 목표 apex/intercept를 고정한다.
- trajectory tilt는 원하는 outgoing velocity 방향에 맞춰 라켓 면을 기울인다.
- centering tilt는 예측 contact XY가 keep-up target에서 벗어날 때만 보정한다.
- RL은 큰 tilt 자체가 아니라 작은 residual만 건드린다.

## 낮은 타격 보정

목표 apex를 더 명확하게 올렸다.

- `target_ball_height = 0.30`

또한 나쁜 contact penalty를 정리했다.

이전에는 non-useful contact penalty가 대체로 "어쨌든 위로 충분히 튄 contact"에만 걸렸다. 이제 약하게 맞아서 높이가 부족한 contact도 `success_reason is None`이면 `nonuseful_contact_penalty`를 받는다.

추가 preset 보강:

- `nonuseful_contact_penalty_weight = 1.25`
- `trajectory_match_reward_weight = 0.35`
- `trajectory_error_penalty_weight = 0.50`

의도는 "무조건 공을 맞히기"가 아니라 "다음에도 치기 쉬운 높이/방향/속도로 맞히기"를 더 강하게 학습시키는 것이다.

## 추천 학습 명령

긴 학습은 아래처럼 새 run으로 분리해서 돌리는 것을 권장한다.

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally_outward \
  --run-version v1 \
  --reset-model \
  --total-timesteps 2000000
```

기존 이름을 유지하고 싶으면 `--run-name pmk_cf_self_rally --run-version v4`로 돌려도 된다.

## 확인 명령

학습 후 viewer:

```bash
python scripts/run_viewer.py \
  --run-name pmk_cf_self_rally_outward \
  --run-version v1 \
  --episodes 100
```

학습 후 rebound 분석:

```bash
python scripts/run_ppo_rebound_analysis.py \
  --run-name pmk_cf_self_rally_outward \
  --run-version v1 \
  --episodes 100 \
  --analysis-name pmk_cf_self_rally_outward_v1_quality100
```

정적 geometry만 보고 싶으면:

```bash
python -m mujoco.viewer --mjcf assets/scene_racket_outward.xml
```

단, 정적 viewer의 첫 자세는 raw XML qpos라 실제 reset 자세가 아니다.

## 성공 판단 기준

다음 모델은 단순 평균 reward보다 아래 지표로 봐야 한다.

- `robot_body_contact` failure rate가 v3보다 낮아지는가
- first/useful contact 이후 `predicted_next_intercept_xy_error`가 줄어드는가
- useful contact의 projected apex height가 `0.30m` 근처로 모이는가
- `target_tilt`가 한쪽 pitch로 계속 포화되지 않고, contact 상황에 따라 pitch/roll 양쪽이 움직이는가
- viewer에서 공이 arm 안쪽으로 빨려 들어가는 비율이 줄었는가

이번 변경은 최종 모델 완성이 아니라, "로봇팔 충돌 구조"와 "고정 tilt 포화"를 끊고 self-rally 학습이 의미 있게 일어나도록 만드는 다음 학습용 기준점이다.
