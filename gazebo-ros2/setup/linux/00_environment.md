# Linux environment from the post

Source post:

- Title: `001 ROS 2 개발 환경 구축`
- Subtitle: `ROS 2 Foxy 설치, IDE 및 개발 환경 구축`
- Created: `2020-07-13`
- Modified: `2022-02-03`
- Revision: `34`

Environment summary extracted from the post:

| Category | Recommended | Optional |
| --- | --- | --- |
| Base OS | Linux Mint 20.x | Ubuntu 20.04.x LTS |
| ROS | ROS 2 Foxy Fitzroy | ROS 2 Rolling Ridley |
| Architecture | amd64 | amd64, arm64 |
| IDE | Visual Studio Code | QtCreator |
| Language | Python 3 (3.8.0), C++14 | newer Python and C++ |
| Simulator | Gazebo 11.x | Ignition Citadel |
| DDS | Fast DDS | Cyclone DDS |
| Other | CMake 3.16.3, Qt 5.12.5, OpenCV 4.2.0 | |

Important notes from the post:

- Install the OS language as `English`.
- Linux is the recommended first environment for ROS beginners.
- The post does not recommend Docker or VM-first setup for beginners.
- Linux Mint 20.x uses Ubuntu 20.04 `focal` packages for ROS 2.

Manual step before the scripts in this folder:

- Install Linux Mint 20.x Cinnamon 64-bit or Ubuntu 20.04.x LTS.
- Reboot into the new system.
- Open a terminal and continue with `10_add_ros2_repository.sh`.