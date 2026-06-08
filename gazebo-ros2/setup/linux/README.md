# Linux ROS 2 Foxy Setup

Linux Mint 20.x 또는 Ubuntu 20.04에서 ROS 2 Foxy와 Gazebo 11을 설치하던 절차를 단계별 스크립트로 정리한 디렉터리다. 원문 환경 재현용 legacy 경로이며, 새 개발 기준은 아니다.

## 대상 환경

- OS: Linux Mint 20.x Cinnamon 64-bit 또는 Ubuntu 20.04 LTS
- ROS: ROS 2 Foxy Fitzroy
- Simulator: Gazebo 11
- DDS: Fast DDS 또는 Cyclone DDS
- IDE: Visual Studio Code

## 실행 순서

```bash
cd /Users/pilt/project-collection/ros2/gazebo-ros2/setup/linux
bash 10_add_ros2_repository.sh
bash 20_install_ros2_foxy.sh
bash 40_install_dev_tools.sh
bash 50_init_robot_ws.sh
```

검증은 두 터미널에서 실행한다.

```bash
bash 30_verify_talker.sh
```

```bash
bash 31_verify_listener.sh
```

셸 자동 설정은 `60_bashrc_snippet.txt` 내용을 확인한 뒤 `~/.bashrc`에 필요한 줄만 추가한다. VS Code가 설치되어 있으면 `70_install_vscode_extensions.sh`로 확장 목록을 맞춘다.

## 주의

- Foxy는 지원 종료된 배포판이라 패키지 mirror 상태에 따라 설치가 실패할 수 있다.
- Linux Mint 20.x는 Ubuntu focal 패키지를 사용한다.
- 현재 MuJoCo 강화학습 실험과 직접 연결되는 경로는 아니다.
