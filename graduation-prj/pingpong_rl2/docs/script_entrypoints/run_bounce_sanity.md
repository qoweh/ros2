# run_bounce_sanity.py

## 한 줄 역할

`scripts/run_bounce_sanity.py`는 Gym/RL 없이 `PingPongSim`만으로 공이 라켓 또는 바닥에 먼저 닿는지 확인하는 최소 물리 sanity check다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_bounce_sanity.py \
  --episodes 5 \
  --ball-height 0.3
```

## 코드 흐름

1. `PingPongSim()`을 만든다.
2. episode마다 simulation을 reset한다.
3. `reset_ball_above_racket()`으로 공 위치와 초기 속도를 지정한다.
4. `run_episode()`에서 MuJoCo substep을 진행한다.
5. 처음 닿은 대상이 `racket_head`인지 `floor`인지 기록한다.
6. failure reason, step 수, peak height, final ball position을 출력한다.

## 주요 호출 관계

```text
run_bounce_sanity.py
  -> envs/pingpong_sim.py
```

## 발표 때 설명 포인트

- RL 이전 단계에서 XML/중력/접촉이 정상인지 확인하는 스크립트다.
- 문제 정의나 환경 sanity check 슬라이드에 “학습 전에 물리 접촉부터 검증했다”는 근거로 쓸 수 있다.
