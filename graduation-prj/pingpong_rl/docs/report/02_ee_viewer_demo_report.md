# EE Viewer Demo 2차 작업 보고

## 1. 작업 목표

이번 작업의 목표는 두 가지였다.

- `mjpython /Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl/scripts/run_viewer.py` 경로에서 grip/pause 동작을 정상화
- 같은 viewer 경로에서 EE(task-space) 제어 데모를 바로 확인할 수 있게 만들기

## 2. 이번 작업에서 구현한 내용

### 2.1 grip 초기상태 보정
- `home` keyframe에서 finger opening을 `0.012`로 줄였다.
- gripper actuator target도 `76.5`로 맞췄다.
- reset 이후 Python 코드가 gripper를 다시 `255`로 강제로 열던 경로를 제거했다.
- racket body를 손가락 pad 기준으로 약간 정렬했다.

### 2.2 passive viewer pause 수정
- 기존 passive loop는 viewer 내부 pause 상태와 무관하게 매 step마다 `mj_step`을 호출했다.
- 현재는 viewer 내부 `Simulate.run` 플래그를 확인해, pause 상태일 때는 physics step을 건너뛴다.

### 2.3 EE demo viewer 경로 추가
- `RacketCartesianController`를 viewer에 연결했다.
- 새 인자:
  - `--demo-controller {hold,joint,ee}`
  - `--ee-axis {x,y,z}`
- 기본 실행은 기존처럼 `hold` 상태를 유지한다.
- `--demo-controller ee`를 주면 `racket_center`가 선택한 축으로 sine motion을 하도록 target 위치를 만든다.

## 3. 검증 결과

### 3.1 단위 테스트
검증 명령:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
python -m unittest discover -s pingpong_rl/tests -p 'test_scene_load.py'
```

결과:
- 10개 테스트 통과
- gripper home target 유지 확인
- passive viewer pause helper 확인
- EE demo target helper 확인
- `RacketCartesianController`가 위치 오차를 줄이는 것 확인

### 3.2 수치 확인
home 상태 확인:
- `home_gripper_target = 76.5`
- `finger_qpos = 0.012, 0.012`

racket local frame 기준 pad/handle 정렬:
- left pad: `[-0.005, 0.0175, 0.0]`
- right pad: `[-0.005, -0.0175, 0.0]`
- handle grip center: `[-0.005, 0.0, 0.0]`

즉 handle 중심이 좌우 finger pad 사이 중앙에 오도록 맞춰졌다.

### 3.3 headless EE sanity check
짧은 EE 데모 루프를 돌린 결과:
- 시작 `racket_center`: `[0.5545, 0.1250, 0.5245]`
- 종료 `racket_center`: `[0.5549, 0.1248, 0.5320]`
- 변화량: `[+0.00034, -0.00019, +0.00748]`

즉 viewer 없이도 task-space target이 실제로 `racket_center`를 움직이는 것을 확인했다.

## 4. 현재 상태 요약

현재 viewer 경로는 다음을 만족한다.

- 기본 실행에서 home pose와 ball reset이 적용된다.
- grip이 완전히 벌어진 상태로 시작하지 않는다.
- space pause가 실제 step loop에 반영된다.
- 같은 경로에서 opt-in 방식으로 EE demo를 실행할 수 있다.

## 5. 다음 작업 제안

### 5.1 바로 이어서 할 것
- EE action을 `racket_center` 기준 delta position으로 고정할지 결정
- observation 벡터 초안 작성
- reward/termination 초안 작성

### 5.2 이후 작업
- EE orientation 제어 추가 여부 판단
- Gymnasium wrapper 연결
- scripted EE baseline 작성
