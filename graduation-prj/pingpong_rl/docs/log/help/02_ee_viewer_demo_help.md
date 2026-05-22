# EE Viewer Demo 도움말

이 문서는 `mjpython /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl/scripts/run_viewer.py` 경로에서 EE(task-space) 데모를 어떻게 쓰는지와, 왜 현재 구조를 이렇게 잡았는지를 짧게 정리한다.

## 1. 왜 기본 실행은 그대로 두고 EE demo를 옵션으로 분리했는가

선택지는 두 가지였다.

- 기본 viewer 실행부터 바로 EE 데모가 계속 움직이게 만들기
- 기본 viewer 실행은 현재처럼 home 자세 유지로 두고, 필요할 때만 EE 데모를 켜기

이번에는 두 번째를 선택했다.

이유:
- 기존 viewer 사용 습관을 덜 깨뜨린다.
- grip/pause 수정이 정상인지 먼저 확인하기 쉽다.
- task-space 제어만 따로 검증할 수 있다.

즉 지금은 `run_viewer.py`의 기본 동작과 `EE demo`를 분리해 두었다.

## 2. 현재 EE demo는 무엇을 제어하는가

현재 EE demo는 Franka hand 원점이 아니라 `racket_center` site를 기준으로 움직인다.

의미:
- 강화학습에서 실제로 중요한 기준점은 손목보다는 paddle strike 중심에 더 가깝다.
- 공과의 접촉, spawn, 보상 설계도 결국 `racket_center` 기준으로 정리하는 편이 일관된다.

현재 controller는 다음만 다룬다.

- 위치(position) 3축
- orientation은 아직 다루지 않음

즉 지금 단계는 “라켓 중심점을 task-space로 움직일 수 있는가”만 확인하는 최소 구현이다.

## 3. 내부적으로 어떻게 동작하는가

`RacketCartesianController`는 `mj_jacSite`로 `racket_center`의 위치 Jacobian을 구한다.

그 다음:
- 현재 `racket_center` 위치와 target 위치의 차이를 계산
- damping이 들어간 DLS 방식으로 joint delta를 계산
- joint limit 안으로 clip
- 그 값을 actuator target으로 보낸다

정리하면:
- 입력: 원하는 `racket_center` 위치
- 출력: 7개 arm joint target

## 4. viewer에서 어떻게 켜는가

기본 실행:

```bash
mjpython /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl/scripts/run_viewer.py
```

EE demo 실행 예시:

```bash
mjpython /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl/scripts/run_viewer.py --demo-controller ee --ee-axis z --demo-amplitude 0.03 --demo-frequency 0.5
```

의미:
- `--demo-controller ee`: joint demo 대신 EE demo 사용
- `--ee-axis z`: `racket_center`를 z축으로 흔듦
- `--demo-amplitude 0.03`: 진폭 3cm
- `--demo-frequency 0.5`: 0.5Hz

joint demo가 필요하면 기존처럼 아래를 쓰면 된다.

```bash
mjpython /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl/scripts/run_viewer.py --demo-controller joint --demo-joint 4
```

## 5. 왜 이제 space pause가 동작하는가

passive viewer에서는 Python 쪽 루프가 계속 `mj_step`을 호출하면 viewer 안에서 pause를 눌러도 체감상 멈추지 않는다.

현재는 viewer 내부 `Simulate.run` 플래그를 읽어서:
- run 상태면 step 진행
- pause 상태면 step 생략

으로 바꿨다.

즉 지금의 pause는 “그림만 멈추는 것”이 아니라 실제 Python step 루프도 멈춘다.

## 6. 현재 한계

- orientation 제어는 아직 없음
- 목표 위치는 sine wave 데모 수준임
- IK 실패/특이점 회피 전략은 아직 최소 수준임

즉 이 코드는 학습용 최종 controller가 아니라, EE 경로가 성립하는지 확인하는 중간 검증 도구다.
