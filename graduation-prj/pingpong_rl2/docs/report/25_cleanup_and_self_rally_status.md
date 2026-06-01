# 25. Artifact Cleanup and Self-Rally Status

작성일: 2026-06-01

최신 `pmk_cf_self_rally_v2` 2M 학습 결과, 병렬 PPO 로그 해석, checkpoint 비활성화, 현재 실패 원인은 `26_learning_runtime_parallel_and_v2_diagnosis.md`를 우선 참고한다. 이 문서는 직전 artifact 정리와 `v1` 기준 상태를 남긴 기록이다.

## Artifact 정리

정리 전 `artifacts`는 약 221MB였다. 발표/비교용 모델만 남기는 기준으로 정리했다.

남긴 PPO run:

- `artifacts/ppo_runs/pmk_cf_self_rally_v1`
  - 최신 self-rally planner/primitive 후보 모델
- `artifacts/ppo_runs/pmk_cf_zero_init_eval_v2`
  - 이전에 비교용으로 괜찮다고 판단했던 모델

삭제한 것:

- `artifacts/ppo_runs` 아래 나머지 오래된 run
- 이름에 `smoke`가 들어간 산출물
- 모든 checkpoint 디렉토리
- 오래된 benchmark sweep CSV/JSON

정리 후 `artifacts`는 약 1.7MB다.

## Checkpoint 판단

checkpoint는 중간 모델을 되돌리거나 best checkpoint를 뽑기 위한 파일이다. 최종 발표/시연에는 보통 `*_best_model.zip`, `*_model.zip`, training summary만 있으면 된다.

이번 정리에서는 checkpoint 디렉토리를 삭제했다. `pmk_cf_self_rally_v1_best_model.zip`과 `pmk_cf_self_rally_v1_model.zip`은 남겼다.

새 self-rally preset은 checkpoint overhead를 줄이도록 바꿨다.

- 기존: 10k마다 50 episode 평가, early-stop 4회
- 중간 변경: 50k마다 20 episode 평가, early-stop 끔
- 최신 변경: 기본 checkpoint 끔. 필요할 때만 `--checkpoint-interval`을 명시한다.

## TensorBoard event 파일

`tb/PPO_1/events.out.tfevents...` 파일은 Stable-Baselines3/TensorBoard가 쓰는 binary event log다. 텍스트로 읽는 파일이 아니라 TensorBoard로 확인한다.

```bash
tensorboard --logdir artifacts/ppo_runs/pmk_cf_self_rally_v1/tb
```

학습 곡선이 필요 없으면 삭제해도 모델 실행에는 영향이 없다. 이번에는 최신 run의 TB 파일만 작게 남겨두었다.

## 최신 모델 상태

사용자가 1M으로 실행한 `pmk_cf_self_rally_v1`은 실제로는 150k에서 early-stop으로 멈췄다.

- requested timesteps: 1,000,000
- completed timesteps: 150,000
- best checkpoint: 110,000
- final evaluation:
  - mean_useful_bounces: 0.51
  - two_or_more_rate: 0.19
  - three_or_more_rate: 0.03
  - max_useful_bounces: 3

추가 100 episode rebound analysis 결과:

- mean_useful_bounces: 0.69
- max_useful_bounces: 7
- two_or_more_rate: 0.17
- total contacts: 536
- useful contact rate: 0.129
- all-contact mean next_intercept_xy_error: 0.093m
- useful-contact mean next_intercept_xy_error: 0.020m
- useful-contact mean lateral speed: 0.045m/s
- useful-contact mean projected apex height: 0.404m

해석:

- useful로 인정된 contact의 XY 방향은 좋아졌다.
- 하지만 전체 contact 중 useful 비율이 낮다.
- useful contact의 apex가 목표 `0.25m`보다 평균적으로 너무 높다.
- 따라서 "어거지로 여러 번 맞는" 상황이 아직 남아 있고, 안정적인 keep-up 완성 상태는 아니다.

## 이번 보강

너무 높게 튄 공도 useful로 인정되는 문제를 줄이기 위해 success contract를 강화했다.

- `require_apex_height_window_for_success` 추가
- self-rally preset에서 `require_apex_height_window_for_success=True`
- self-rally preset에서 `min_easy_next_ball_score_for_success=0.35`
- self-rally preset에서 `easy_next_ball_reward_weight=1.0`

즉 다음 학습부터는 공이 라켓 근처로 돌아오는 것뿐 아니라, 목표 높이 주변으로 안정적으로 돌아오는 접촉만 useful success로 인정된다.

## 다음 권장 학습

```bash
conda activate mujoco_env
python scripts/run_ppo_learning.py \
  --preset contact_frame_self_rally_candidate \
  --run-name pmk_cf_self_rally \
  --run-version v2 \
  --reset-model \
  --total-timesteps 1000000
```

`v1`은 150k early-stop 모델이므로, 강화된 contract와 checkpoint 설정으로 `v2`를 새로 학습하는 것이 맞다.

## 다음 판단 기준

학습 결과를 볼 때 reward 평균보다 아래 지표를 먼저 본다.

- `mean_useful_bounces >= 2.0`
- `two_or_more_rate >= 0.50`
- `three_or_more_rate >= 0.30`
- `useful_contact_mean_next_intercept_xy_error <= 0.025m`
- `useful_contact_mean_projected_apex_height`가 `0.25m +/- 0.10m` 안에 들어오는지
- `useful_contact_mean_ball_lateral_speed <= 0.08m/s`
- `robot_body_contact_rate`와 `ball_out_of_bounds_rate`가 함께 줄어드는지

이 기준을 못 넘으면 단순 timesteps 추가보다 controller/body-clearance와 contact primitive 속도 추종을 먼저 봐야 한다.
