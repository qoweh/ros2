# macOS ROS 2 / Gazebo Setup

macOS에서 ROS 2와 Gazebo 계열 도구를 쓰기 위한 두 가지 경로를 보관한다. Apple Silicon에서는 `conda/` 경로가 기본 선택이다.

| 경로 | 선택 기준 |
| --- | --- |
| `conda/` | RoboStack 기반 Jazzy/Gz Sim 설치. 현재 macOS에서 가장 실용적 |
| `native/` | Homebrew와 source build를 이용한 host native 설치. 문제 해결 비용이 큼 |

## 권장 경로

```bash
cd /Users/pilt/project-collection/ros2/gazebo-ros2/setup/mac/conda
bash 10_create_ros_env.sh
bash 20_install_ros_jazzy_and_gz.sh
bash 30_verify_talker.sh
bash 31_verify_listener.sh
bash 32_start_gz_server.sh
```

셸 설정은 `conda/40_shell_notes.md`를 참고한다. VS Code 확장은 상위 `70_install_vscode_extensions.sh`를 사용한다.

## native 경로

`native/`는 비-conda 설치가 꼭 필요할 때만 사용한다. 같은 셸 세션에서 `conda/`와 `native/` 환경 변수를 섞지 않는다.
