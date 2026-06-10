- 모든 작업을 할 때는 [원칙](../mujoco/md-docs/RULE.md) 파일을 항상 지켜.

- 터미널에서 명령어로 ros_activate 치면 자동으로 zsh에서 
`ros_activate() {
        conda activate ros_env
        eval "$(register-python-argcomplete ros2)"
        eval "$(register-python-argcomplete colcon)"
}`가 실행되게 해놨어. `conda list -n ros_env`로 설치된 패키지들을 확인하면 돼. 버전 충돌 안 일어나게 조심하고, 고려해야 할 것이 있으면 나한테 물어봐.

- 현재 ROS 2/Gazebo 작업은 `gazebo-ros2`에서 진행하고, MuJoCo 강화학습 작업은 `mujoco`에서 진행해.

<!-- - 필요한 것들을 적절히 활용해서 so-101를 gazebo에서 실행시켜서 로봇팔을 움직이게 해보고 싶어. 어떤 식으로 개발할 지 방법이 다양하잖아? 여러 후보군들을 claude-proposa.md로 만들어. -->

1. 로봇팔을 어떤 시뮬레이터에서 실행할지 선택해야 함. 시뮬레이터를 gazebo에 한정하지 말고 mujoco나 다른 것도 고려를 해봐야 함. 정리해봐.
2. 로봇팔도 굳이 so-101이 아니라 학습시킬 종류에 따라 다른 것들도 고려해야 함. 개발 용이, 오류 등등. 정리해봐.
3. 어떤 학습을 시킬지에 따라 최종적으로 시뮬레이터와 학습 종류, 개발 방향이 정해짐. 로봇팔로 어떤 작업을 학습시킬지 고민인데, 학습이 잘 되는 행동이 있나? 탁구공 안 떨어지게 위로 계속 탁구채로 튕기는 행동은 어떤지 생각해보고 다른 어떤 행동을 학습시킬지도 추천해봐 - 각 행동에 어떤 학습을 시켜야 하는지도.
4. claude-proposal2.md로 만들어.
