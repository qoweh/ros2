# ROS 2 / Gazebo Setup Notes

이 디렉터리는 ROS 2와 Gazebo 설치 절차를 플랫폼별로 나눈 메모다. 현재 강화학습 본 실험은 `mujoco/`에서 진행하지만, ROS 2/Gazebo 환경 재현이나 로봇팔 제어 연동을 다시 확인할 때 이 자료를 사용한다.

| 경로 | 기준 환경 | 용도 |
| --- | --- | --- |
| `linux/` | Linux Mint 20.x 또는 Ubuntu 20.04 + ROS 2 Foxy + Gazebo 11 | 원문 설치 절차를 재시작 가능한 단계로 분리 |
| `mac/conda/` | macOS, 특히 Apple Silicon + RoboStack Jazzy | 현재 macOS에서 가장 재현성 높은 ROS 2/Gazebo 경로 |
| `mac/native/` | macOS + Homebrew + source build | 비-conda 네이티브 설치가 꼭 필요할 때 |

## 권장 순서

1. Linux에서 원문을 재현하려면 `linux/README.md`부터 읽는다.
2. macOS에서 ROS 2/Gazebo를 쓰려면 `mac/README.md`를 읽고 보통 `mac/conda/`를 선택한다.
3. 설치 후 talker/listener 검증 스크립트를 먼저 통과시킨다.
4. Gazebo/Gz Sim 검증은 ROS 2 검증이 끝난 뒤 진행한다.

Foxy는 이미 EOL인 legacy 환경이다. 새 실험이나 Apple Silicon에서는 Jazzy/RoboStack 경로를 우선한다.
