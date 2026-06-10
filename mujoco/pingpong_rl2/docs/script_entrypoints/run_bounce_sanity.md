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

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 Gym 환경도 PPO도 쓰지 않는다. `PingPongSim`만 직접 만들어서 공을 라켓 위에 놓고 MuJoCo step을 진행한다. 그래서 강화학습 이전 단계에서 XML scene, 중력, 공 위치, 라켓 contact가 최소한 정상인지 보는 가장 낮은 수준의 sanity check다.

```text
run_bounce_sanity.py
  -> PingPongSim()
  -> for episode
       sim.reset()
       sim.reset_ball_above_racket(height, xy_offset, velocity)
       run_episode(sim, max_steps)
         for step
           sim.step(n_substeps=1)
           ball/racket 또는 ball/floor 첫 contact 확인
           sim.failure_reason() 확인
       episode summary 출력
  -> racket_first 비율 출력
```

`run_episode()`는 공의 peak height를 계속 갱신하고, 첫 target contact가 `racket_head`인지 `floor`인지 기록한다. `sim.failure_reason()`이 나오면 즉시 episode를 끝낸다. max step까지 아무 failure가 없으면 `failure_reason="max_steps"`로 반환한다.

## 이 파일이 확인하는 것

여기서는 policy action, reward, useful bounce 같은 개념이 없다. 오직 MuJoCo scene에서 공이 라켓에 닿는지, floor에 먼저 닿는지, 공 위치가 out of bounds가 되는지 같은 물리 기본 동작만 본다.

학습이 이상할 때 이 파일이 정상이라면 "기본 XML 로딩과 contact geometry는 살아 있다"고 볼 수 있다. 반대로 여기서 공이 라켓에 닿지 않거나 contact가 이상하면 PPO나 reward를 보기 전에 scene/material/contact 설정부터 확인해야 한다.

## 발표 때 설명 포인트

- RL 이전 단계에서 XML/중력/접촉이 정상인지 확인하는 스크립트다.
- 문제 정의나 환경 sanity check 슬라이드에 “학습 전에 물리 접촉부터 검증했다”는 근거로 쓸 수 있다.
