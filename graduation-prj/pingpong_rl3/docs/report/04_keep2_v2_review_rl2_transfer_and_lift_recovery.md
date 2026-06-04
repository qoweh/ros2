# keep2_v2 검토, rl2 전이 가능성, lift recovery 적용

작성일: 2026-06-04

## keep2_v2 결과

`keep2_v2_staggered`는 두 공의 초기 phase gap을 벌렸지만 `keep2_v1`보다 좋아지지 않았다.

| run | mean useful | max useful | mean steps | mean contacts |
| --- | ---: | ---: | ---: | ---: |
| keep2_v1 | 4.71 | 24 | 59.70 | 20.61 |
| keep2_v2_staggered | 3.96 | 13 | 54.38 | 12.53 |

v2 contact 계측:

- 전체 contact outgoing `vz` median: `0.233m/s`
- non-useful contact outgoing `vz` median: `0.046m/s`
- useful contact projected apex median: `0.161m`
- target apex: `0.24m`
- contact 순간 racket `vz` median: `-0.002m/s`

판단:

- phase gap만 벌려서는 충분하지 않다.
- 라켓이 contact 순간 위로 움직이지 못하는 문제가 계속 남아 있다.
- useful 기준의 `min_useful_apex_height=0.10m`가 너무 낮아 낮은 통통 루프도 보상을 받는다.

## pingpong_rl2에서 효과가 있었던 것

rl2의 안정 모델 계열은 다음 순서로 좋아졌다.

1. 낮은 apex를 바로 실패로만 보지 않고 recovery progress reward를 추가했다.
2. 낮은 apex 이후 다음 타격에 extra lift와 upward target velocity를 넣었다.
3. 낮은 안정 루프가 lateral/steady reward를 받지 못하게 height-qualified reward를 넣었다.
4. action에 target apex residual과 strike plane residual을 추가했다.
5. controller velocity step과 contact-frame velocity target을 충분히 크게 열었다.
6. spin/넓은 reset을 처음부터 넣지 않고 staged distribution으로 넓혔다.

현재 keep2 병목은 1, 2, 3, 5, 6에 해당한다. 4는 이미 keep2 action에 `target_apex_z_residual`과 `contact_z_residual`이 있으므로 새 action dimension을 늘릴 필요는 아직 없다.

## rl2 모델 zip을 직접 가져올 수 있나?

직접 resume은 불가능하다.

- rl2 `keep1_v30/v26/v31`: observation `(55,)`, action `(15,)`, MLP `64x64`
- rl3 `keep2_v1/v2`: observation `(42,)`, action `(13,)`, MLP `256x256`

SB3는 observation/action space가 다르면 `PPO.load(..., env=...)` resume을 거부한다. policy layer shape도 달라서 weight를 그대로 이식할 수 없다.

가능한 전이는 모델 파일 자체보다 아래 방식이다.

- rl2의 recovery primitive/reward/controller 설정을 keep2에 이식한다.
- rl3의 기존 keep2 모델은 observation/action space가 같으므로 v3로 resume fine-tune한다.
- 더 공격적으로 하려면 나중에 rl2 policy를 teacher로 두고 target ball만 synthetic 1-ball observation으로 매핑하는 distillation script를 따로 만든다. 지금은 구조 대비 효과가 불확실하다.

## keep2_v3_lift_recovery

새 config:

- `configs/keep2_v3_lift_recovery.json`

주요 변경:

- target apex: `0.24 -> 0.28`
- min useful apex: `0.10 -> 0.20`
- reset spin: `20 -> 0`
- reset XY/velocity 난이도 축소
- controller max velocity step: `0.028 -> 0.080`
- target velocity max: `2.2 -> 3.0`
- min useful outgoing `vz`: `0.35 -> 0.45`
- min useful racket `vz`: `0.00 -> 0.03`
- low-apex recovery lift/velocity 추가
- under-min apex penalty, progress reward, recovery progress, potential shaping 추가

코드 변경:

- `TwoBallKeepUpEnv`에 per-ball last apex/shortfall memory 추가
- target ball의 이전 apex shortfall이 있으면 다음 descent에서 strike plane lift와 upward target velocity를 추가
- contact reward에 apex progress/recovery/potential/under-min penalty 추가
- controller gain/velocity step을 config에서 조절 가능하게 노출

## 권장 학습

v3는 action/observation space가 v2와 같으므로 v2에서 이어서 fine-tune한다.

```bash
cd /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl3
conda activate mujoco_env
python scripts/train.py \
  --config configs/keep2_v3_lift_recovery.json \
  --resume-from artifacts/ppo_runs/keep2_v2_staggered/keep2_v2_staggered_model.zip
```

v3가 더 나빠지면 v1에서 resume해서 비교한다.

```bash
python scripts/train.py \
  --config configs/keep2_v3_lift_recovery.json \
  --run-name keep2_v3_from_v1 \
  --resume-from artifacts/ppo_runs/keep2_v1/keep2_v1_model.zip
```

분석에서 볼 것:

- useful contact projected apex median이 `0.20m` 위로 올라가는지
- contact 순간 racket `vz` median/p75가 양수로 올라가는지
- floor contact가 줄고, speed/out-of-bounds가 과하게 늘지 않는지
- `recovery_lift`, `recovery_velocity`, `last_contact_racket_vz`가 info에 기록되는지

## 검증

- `compileall` 통과
- smoke test 함수 직접 실행 통과
- v3 scratch PPO smoke 통과
- v2 model에서 v3 config로 resume PPO smoke 통과
