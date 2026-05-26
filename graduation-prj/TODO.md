로봇팔이 탁구채를 잡고 탁구공을 떨어지지 않게 계속 하늘 위로 튕기는 강화학습
[agent-RULE](./RULE.md)


## 세팅

- 노트북 정보 :
      Model Identifier: MacBookPro18,1
      Model Number: Z14W000QXKH/A
      Chip: Apple M1 Pro
      Total Number of Cores: 10 (8 Performance and 2 Efficiency)
      Memory: 32 GB
- 시뮬레이터 : MuJoCo 3.8 (conda activate mujoco_env) python 3.10버전, conda env mujoco_env 에 존재
- 로봇팔 : Franka Emika Panda (위치: ~/mujoco_menagerie/franka_emika_panda)
- 로봇팔과 탁구채, 탁구공 존재
- 카메라 없이 공의 위치나 정보들을 알고 있음을 가정
- Sim2Real까지는 안 하고 simulation 환경에서만 RL


## 보완 사항
- total_timesteps를 1M 돌려본 결과 `/Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2/artifacts/ppo_runs/ppo_minimal_keepup`디렉토리에 모델과 로그가 있고 mjpython `/Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2/scripts/run_viewer.py`으로 확인해보니, 로봇팔이 탁구공을 올려치는 모션 학습이 잘 안 됨.
- 어떻게 해야 로봇팔이 탁구채로 탁구공을 위로 계속 올려치게 학습시킬 수 있을지 잘 모르겠음.
- 고려해야 할 것들이 어떤 것들인지, 차근차근 어떤 것들을 하나씩 수정 및 추가 및 삭제해야 할 지 모르겠음.

## agent 작업 진행 방향

### 1차 수정: 첫 바운스 이후 학습 신호 복구

- 기존 환경은 `contact_count > 0`이 된 뒤 `tracking_term`을 0으로 꺼서, 두 번째 바운스부터는 공 아래로 다시 들어가는 dense reward가 거의 없었다.
- `tracking_term`을 첫 접촉 이후에도 매 descending strike window마다 살아 있게 수정했다.
- strike window 안에서도 준비 높이에 가까울수록 reward가 커지도록 vertical score를 곱했다.
- 접촉 이후에도 공이 다시 내려오기 전까지 target Z가 과하게 위로 유지되지 않도록 strike guard를 반복 적용한다.
- 접촉 시점의 `ball_height_above_racket`, `xy_alignment_error`를 `contact_trace`에 기록해서 useful bounce 판정과 로그 해석을 더 직접적으로 볼 수 있게 했다.

### 2차 확인 순서

1. 단위 테스트로 reward/guard 동작이 유지되는지 확인한다.
2. `run_ppo_learning.py --smoke`로 학습 루프가 깨지지 않는지 확인한다.
3. 100k~300k timesteps 정도의 짧은 run에서 `mean_useful_bounces`, `failure_counts`, `ball_out_of_bounds` 비율을 본다.
4. 짧은 run에서 첫 바운스는 늘지만 여전히 밖으로 많이 나가면, 그때만 rebound direction 또는 lateral velocity penalty를 최소 항으로 추가한다.
5. useful contact 자체가 거의 안 늘면, curriculum보다 먼저 reset 범위와 strike guard timing을 좁게 조정한다.

### 아직 바로 넣지 않을 것

- heuristic keep-up controller를 env 내부 assist로 섞지 않는다.
- tilt action은 position-only baseline 결과를 더 본 뒤 판단한다.
- single-bounce-out 전용 penalty는 현재 수정 이후에도 exploit가 반복될 때만 다시 검토한다.
