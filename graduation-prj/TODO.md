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
