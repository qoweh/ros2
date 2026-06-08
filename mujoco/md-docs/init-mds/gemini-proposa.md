# SO-101 로봇팔 Gazebo 컨트롤 개발 후보군 (Proposals)

ROS 2(Jazzy)와 Gazebo(Harmonic) 환경에서 SO-101 로봇팔을 시뮬레이션하고 제어하기 위한 개발 방법 후보군입니다. 현재 `gazebo-ros2`와 `mujoco` 구조 안에는 SO-101 모델(URDF/Xacro)이 없는 상태이므로, 모델링부터 제어까지의 전체 파이프라인 구축을 전제로 작성했습니다.

## 후보 1: ros2_control + MoveIt2 (가장 표준적이고 강력한 방법)
ROS 2 생태계에서 로봇 매니퓰레이터를 제어하는 가장 정석적인 방법입니다.

*   **장점:**
    *   역기구학(IK), 충돌 회피, 모션 플래닝 등 복잡한 제어를 MoveIt2가 자동으로 처리해줌.
    *   `gz_ros2_control` 플러그인을 통해 Gazebo와 완벽하게 통합됨.
    *   향후 실제 하드웨어 (SO-101 실제 로봇) 연동 시 컨트롤러만 교체하면 코드를 그대로 재사용 가능함.
*   **단점:**
    *   설정 파일(URDF, SRDF, controllers.yaml, moveit configs 등)이 많고 초기 학습 곡선이 높음.
    *   시스템이 무거울 수 있음.
*   **작업 흐름:** SO-101 URDF 작성 -> `ros2_control` 태그 추가 -> MoveIt Setup Assistant로 설정 패키지 생성 -> Gazebo 브릿지 및 컨트롤러 런치 파일 작성.

## 후보 2: 단순 ROS 2 Topic/Service 제어 (JointTrajectoryController 수동 사용)
MoveIt2의 복잡한 모션 플래닝 기능을 빼고, `ros2_control`의 `JointTrajectoryController`나 `PositionController`만 사용하여 직접 joint 각도를 쏴주는 방법입니다.

*   **장점:**
    *   MoveIt2 설정 없이 비교적 빠르게 시뮬레이션 제어 환경 구축 가능.
    *   단순한 Pick & Place나 직접 지정한 각도로의 이동(FK)만 필요할 때 직관적이고 가벼움.
*   **단점:**
    *   역기구학(IK)이나 충돌 회피를 사용자가 직접 파이썬/C++ 노드로 계산해서 넘겨줘야 함.
*   **작업 흐름:** SO-101 URDF 작성 -> `ros2_control` 태그 추가 -> 컨트롤러 설정 및 Gazebo 실행 -> 사용자가 직접 `JointTrajectory` 메시지를 publish하는 ROS 2 노드 작성.

## 후보 3: Gazebo (Ignition) Joint Controller 직접 사용 (ROS 2 플러그인 최소화)
ROS 2의 `ros2_control`을 거치지 않고, Gazebo(Harmonic) 자체의 Joint Controller 플러그인을 모델에 부착한 뒤, `ros_gz_bridge`를 통해 ROS 2 토픽으로 제어 명령만 브릿징하는 방법입니다.

*   **장점:**
    *   초기 구조가 가장 단순함. URDF (또는 SDF) 모델에 Gazebo 플러그인만 한 줄 추가하면 됨.
*   **단점:**
    *   ROS 2 표준인 `ros2_control` 인터페이스를 따르지 않기 때문에 향후 실제 하드웨어 연동이나 MoveIt2 확장이 매우 번거로워짐.
*   **작업 흐름:** SO-101 SDF/URDF 작성 -> Gazebo Joint Position Controller 플러그인 추가 -> `ros_gz_bridge` 설정 -> 토픽 퍼블리시 노드 작성.

---

### 질문 및 다음 단계
1.  **SO-101 모델 유무:** 현재 모델(URDF, 메쉬 파일 등)을 이미 가지고 계신가요? 혹은 오픈소스 링크가 있다면 알려주세요.
2.  **어떤 옵션으로 진행할까요?** 로봇팔의 목적(단순 티칭용, 복잡한 궤적 생성, 실제 하드웨어 전환 여부)에 따라 1번이나 2번을 추천합니다. 선택해주시면 해당 방향으로 첫 번째 패키지 생성을 진행하겠습니다.
