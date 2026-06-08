# SO-101 로봇팔 Gazebo 시뮬레이션 개발 방법 제안

## 환경 정보 (현재 설치 상태)

| 항목 | 버전/상태 |
|------|-----------|
| ROS 2 | Jazzy (robostack-jazzy, conda) |
| Gazebo | Harmonic (gz-sim8) |
| ros_gz bridge | 설치됨 (`ros-jazzy-ros-gz-bridge`, `ros-jazzy-ros-gz-sim`) |
| rviz2 | 설치됨 |
| xacro / urdf tools | 설치됨 |
| joint_state_publisher | 설치됨 |
| ros2_control | **미설치** |
| MoveIt 2 | **미설치** |

## SO-101 공식 시뮬레이션 파일 위치

- 공식 URDF: `TheRobotStudio/SO-ARM100` 리포 → `Simulation/SO101/so101_new_calib.urdf`
- 공식 MJCF(MuJoCo): `Simulation/SO101/so101_new_calib.xml` + `scene.xml`
- 메쉬 파일: `Simulation/SO101/assets/`
- 6-DOF, STS3215 서보 6개 구성

---

## 후보안 비교

| 후보 | 복잡도 | 추가 설치 | Gazebo 사용 | 관절 제어 방식 |
|------|--------|-----------|-------------|----------------|
| A: URDF + ros_gz + 직접 토픽 제어 | ⭐ 낮음 | 없음 | ✅ | ROS 토픽 직접 publish |
| B: URDF + ros_gz + ros2_control | ⭐⭐ 중간 | ros2_control, ros2_controllers | ✅ | JointTrajectoryController |
| C: URDF + ros_gz + MoveIt 2 | ⭐⭐⭐ 높음 | MoveIt 2 전체 스택 | ✅ | 태스크 공간 계획 |
| D: MJCF + MuJoCo Python (Gazebo 없음) | ⭐ 낮음 | `mujoco` pip | ❌ | Python API 직접 |

---

## 후보 A: URDF + ros_gz + 직접 토픽 제어 (최소 접근)

### 개요
공식 URDF를 그대로 사용하고, `ros_gz_sim`으로 Gazebo에 스폰한 뒤, `ros_gz_bridge`를 통해 ROS 토픽으로 관절 명령을 보내는 가장 단순한 방법.

### 구성 요소
```
mujoco/
  so101_description/       ← URDF + meshes + robot_state_publisher
  so101_gazebo/            ← launch 파일, world 파일
  so101_teleop/            ← joint command publisher (Python 노드)
```

### 동작 흐름
1. SO-ARM100 리포에서 URDF + assets 복사
2. `robot_state_publisher` 노드로 `/tf` 발행
3. `ros_gz_sim`의 `create` 서비스로 Gazebo에 로봇 스폰
4. `ros_gz_bridge`로 Gazebo joint 토픽 ↔ ROS 연결
5. Python 노드에서 `/model/so101/joint_state` 또는 `JointVelocity` 토픽으로 관절 이동

### 장점
- 추가 패키지 설치 없음 (현재 환경으로 바로 가능)
- 코드량 최소 (launch 파일 1개 + 제어 노드 1개)
- 빠른 프로토타이핑 가능

### 단점
- 위치 제어 루프가 없어 물리적으로 정밀하지 않음
- 충돌 방지, 경로 계획 없음
- Gazebo의 `GZ_SIM_PHYSICS`와 직접 인터페이스해야 할 수 있음

### 소요 단계 (예시)
```bash
# 1. 리포 클론
git clone https://github.com/TheRobotStudio/SO-ARM100 ~/so_arm100

# 2. 패키지 생성 및 URDF 복사
ros2 pkg create so101_description --build-type ament_cmake
cp -r ~/so_arm100/Simulation/SO101/ so101_description/

# 3. launch 파일 작성 → gz_sim + ros_gz_bridge + robot_state_publisher

# 4. 빌드 및 실행
colcon build
ros2 launch so101_gazebo so101_gazebo.launch.py
```

---

## 후보 B: URDF + ros_gz + ros2_control (표준 접근)

### 개요
`ros2_control` + `JointTrajectoryController`를 사용해 ROS 2 표준 방식으로 관절 궤적 명령을 보내는 방법. 실제 로봇에서도 동일 인터페이스를 쓸 수 있어 확장성이 높음.

### 구성 요소
```
mujoco/
  so101_description/       ← URDF (ros2_control 태그 추가)
  so101_gazebo/            ← gz_ros2_control 플러그인 연결 + launch
  so101_moveit_config/     ← (선택) MoveIt 없이 직접 JointTrajectory action 사용
  so101_controller/        ← JointTrajectory action client (Python)
```

### 추가 설치 필요
```bash
conda install -n ros_env ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-gz-ros2-control
```
> 버전 충돌 가능성: robostack-jazzy에서 지원되는지 먼저 확인 필요

### 동작 흐름
1. URDF에 `<ros2_control>` 태그 + `gz_ros2_control` 플러그인 추가
2. Gazebo 실행 + `controller_manager` 스폰
3. `JointTrajectoryController` 활성화
4. Python 노드에서 `FollowJointTrajectory` action으로 목표 관절각 전송

### 장점
- 실제 SO-101 하드웨어와 동일 인터페이스 재사용 가능
- 부드러운 궤적 보간 (속도/가속도 제한 포함)
- 졸업 프로젝트 데모에 적합한 수준

### 단점
- `ros-jazzy-gz-ros2-control` conda 패키지 존재 여부 불확실 → 소스 빌드 필요 가능
- 설정 파일(yaml) 작업량 증가
- URDF 수정이 필요 (inertia, ros2_control 태그)

---

## 후보 C: URDF + ros_gz + MoveIt 2 (풀 스택)

### 개요
MoveIt 2를 사용해 태스크 공간(Cartesian) 목표 지정, 충돌 회피 경로 계획까지 지원하는 가장 완성도 높은 방법.

### 추가 설치 필요
```bash
conda install -n ros_env ros-jazzy-moveit ros-jazzy-ros2-control \
  ros-jazzy-gz-ros2-control
```
> robostack-jazzy에서 `ros-jazzy-moveit`가 제공되나 용량 크고 충돌 위험 있음

### 동작 흐름
1. 후보 B와 동일하게 ros2_control 설정
2. `moveit_setup_assistant` 또는 수동으로 MoveIt config 패키지 생성
3. `move_group` 노드 실행
4. Python MoveIt API(`MoveGroupInterface`)로 end-effector 목표 지정

### 장점
- 가장 강력한 계획/제어 파이프라인
- 추후 실물 로봇 연동 시 동일 코드 사용 가능
- 연구/졸업 논문 수준의 완성도

### 단점
- 설치 및 설정 복잡도 매우 높음
- conda robostack에서 MoveIt 안정성 보장 어려움
- URDF inertia/collision 품질에 민감 (현재 URDF는 collision 부분 수정됨)

---

## 후보 D: MJCF + MuJoCo Python (Gazebo 없이)

### 개요
공식 MuJoCo 파일(`so101_new_calib.xml`)을 그대로 사용하여 MuJoCo Python API로 시뮬레이션하는 방법. Gazebo를 사용하지 않으므로 시뮬레이션 목적에 맞지 않을 수 있음.

### 추가 설치
```bash
pip install mujoco
```

### 장점
- 설치 가장 간단, 공식 파일 바로 사용
- LeRobot 라이브러리와 직접 연동 가능

### 단점
- **Gazebo가 아님** → TODO 요구사항 불충족
- ROS 2와의 연동이 없어 ROS 생태계 활용 불가

---

## 추천

**단계적 접근을 권장:**

```
1단계 → 후보 A  (지금 당장 실행 가능, 설치 없음)
2단계 → 후보 B  (안정적이면 conda로 ros2_control 추가)
```

**후보 A로 먼저 시작하는 이유:**
- 현재 ros_gz, rviz2, urdf 도구가 이미 설치되어 있어 추가 설치 없이 바로 진행 가능
- Gazebo에서 SO-101 URDF를 보고 관절을 움직이는 최소 목표를 빠르게 달성
- ros2_control/MoveIt 설치 시 conda 버전 충돌 리스크를 피할 수 있음

---

## 질문

진행 방향을 결정하기 전에 확인이 필요한 사항:

1. **목표 수준**: "움직이게 한다"가 단순 관절각 변화(demo 수준)인지, 경로 계획 포함(연구 수준)인지?
2. **실물 연동 계획**: 나중에 실제 SO-101 하드웨어와 연결할 예정인지?
3. **conda 패키지 추가 가능 여부**: ros2_control/MoveIt 설치 시도해볼 의향이 있는지? (버전 충돌 가능성 있음)
