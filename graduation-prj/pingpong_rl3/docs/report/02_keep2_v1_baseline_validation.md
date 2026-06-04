# keep2_v1 baseline validation

작성일: 2026-06-04

## 목적

`pingpong_rl3`의 첫 2-ball keep-up baseline이 학습 스크립트까지 실제로 돌아가는지 확인하고, 다음 학습으로 넘어가기 전에 작은 계약을 보강했다.

## 변경

- `max_episode_steps=0`을 time limit 비활성으로 처리했다.
- 라켓 접촉이 step 시작 시점부터 이미 활성인 경우 새 contact reward로 세지 않게 했다.
- `training_config()`가 reward, bounds, scheduler, termination 파라미터를 함께 기록하게 했다.
- `analyze.py`에 mean return, mean steps, max steps, max contacts, seed, deterministic 여부를 추가했다.
- `analyze.py`와 `viewer.py`에 `--device` 옵션을 추가했다.
- smoke test에 reset 분포, contact-start 필터, vector env autoreset 계약을 추가했다.

## 검증

`mujoco_env`에서 아래를 확인했다.

```bash
PYTHONPATH=pingpong_rl3/src /Users/pilt/miniforge3/envs/mujoco_env/bin/python -m compileall -q pingpong_rl3/src pingpong_rl3/scripts pingpong_rl3/tests
```

```bash
PYTHONPATH=pingpong_rl3/src /Users/pilt/miniforge3/envs/mujoco_env/bin/python pingpong_rl3/scripts/train.py \
  --config pingpong_rl3/configs/keep2_v1.json \
  --run-name smoke_keep2_after_contact_filter \
  --total-steps 4 \
  --num-envs 1 \
  --device cpu
```

짧은 PPO smoke run은 `n_steps=512` 때문에 실제 `total_timesteps=512`까지 수집했고 정상 저장됐다.

## 다음 작업

- `keep2_v1`을 본 학습으로 돌린다.
- 100 episode 분석에서 useful bounce, time limit, floor/out-of-bounds 비율을 본다.
- 결과가 불안정하면 먼저 `reset_xy_range`, `slot_xy_offsets`, `reachable_radius`를 좁히고, 안정적이면 curriculum으로 넓힌다.
