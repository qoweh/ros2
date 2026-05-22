# MuJoCo 환경 구축과 RL 파이프라인 기준 문서

이 문서는 현재 프로젝트에서 헷갈리기 쉬운 결정 사항, 이미 끝난 작업, 다음 작업을 한 곳에 모은 기준 문서다.

## 1. 현재 확정된 방향

### 1.1 시뮬레이터 계층
- 초기 환경은 순수 MuJoCo Python 바인딩 기반으로 진행한다.
- 이유:
  - Viewer로 물리와 충돌을 바로 확인하기 쉽다.
  - RL wrapper를 얹기 전에 scene 자체를 안정화하기 좋다.
  - contact/solver 튜닝을 낮은 레벨에서 직접 볼 수 있다.

### 1.2 현재 코드 구조
- 프로젝트 루트 기준 핵심 경로:
  - `pingpong_rl/assets`: MuJoCo scene과 asset 저장
  - `pingpong_rl/src/pingpong_rl`: Python 패키지 코드
  - `pingpong_rl/scripts`: 수동 실행 스크립트
  - `pingpong_rl/tests`: 기본 검증 코드
- 현재는 표준 `src` 레이아웃을 사용한다.

### 1.3 라켓 부착 방식
- 현재 구현 선택: Franka의 `hand` body에 라켓 body를 자식으로 직접 부착했다.
- 장점:
  - 가장 단순하다.
  - weld 제약 없이 안정적으로 hand를 따라간다.
  - 초기 scene 검증 단계에서 디버깅이 쉽다.
- 단점:
  - 나중에 라켓 교체/옵션화를 하려면 robot asset 수정 범위가 생긴다.
- 대안:
  - separate body + weld equality
  - 장점: 라켓 자산 분리가 쉽다.
  - 단점: 초기에 좌표계와 제약 설정이 더 번거롭다.

## 2. 현재 구현 완료 상태

### 2.1 완료
- Franka Menagerie asset을 프로젝트 내부로 복사
- 하나의 MuJoCo scene에 다음 요소 통합
  - Franka Panda
  - 탁구채
  - 탁구공
  - 바닥
- MuJoCo scene 로딩 검증 완료
- ball freejoint 기반 spawn/reset 구현 완료
- joint target 기반 arm control API 구현 완료
- editable install 패키징 설정 완료
- interactive/passive viewer 분리 완료
- 라켓 contact geom을 flat paddle 기준으로 1차 튜닝 완료
- contact query API와 기본 unittest 추가 완료
- failure/reset 판정 API 완료
- headless scripted bounce baseline 추가 완료

### 2.2 현재 구현 파일
- scene: `pingpong_rl/assets/scene.xml`
- robot asset: `pingpong_rl/assets/franka/panda.xml`
- environment wrapper: `pingpong_rl/src/pingpong_rl/envs/pingpong_env.py`
- joint controller: `pingpong_rl/src/pingpong_rl/controllers/joint_controller.py`
- viewer script: `pingpong_rl/scripts/run_viewer.py`
- bounce baseline script: `pingpong_rl/scripts/run_bounce_baseline.py`
- test: `pingpong_rl/tests/test_scene_load.py`

## 3. 지금 바로 실행 가능한 것

### 3.1 기본 테스트
```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
python -m unittest discover -s pingpong_rl/tests -p 'test_scene_load.py'
```

### 3.2 Viewer 실행
```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
pingpong-rl-viewer --demo-joint 4
```

설명:
- `--demo-joint 4`는 4번 관절에 작은 sine motion을 줘서 joint control이 적용되는지 확인하는 용도다.
- 기본 interactive mode는 stock MuJoCo viewer CLI에 scene 경로를 넘겨 실행한다.
- custom loop 확인이 필요할 때만 `--mode passive`를 사용한다.

### 3.3 Headless baseline 실행
```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
python pingpong_rl/scripts/run_bounce_baseline.py --episodes 3 --max-steps 900
```

설명:
- 각 episode마다 공을 `racket_center` 위에서 다시 떨어뜨린다.
- 출력에는 `first_target_contact`, `failure_reason`, `steps`, `time`이 포함된다.
- 현재 기준 baseline 결과는 `racket_first=3/3`, 실패 원인은 `floor_contact`다.

## 4. 현재 물리/제어 기준값

### 4.1 Physics
- timestep: `0.002`
- gravity: `0 0 -9.81`
- ball radius: `0.02`
- ball mass: `0.0027`

### 4.2 Control
- control_dt: `0.02`
- physics timestep 대비 약 10 substeps로 step 진행
- Panda actuator는 기존 menagerie position-style actuator 설정을 그대로 사용

## 5. 다음 작업 체크리스트

### 5.1 바로 이어서 할 것
- end-effector 제어 계층을 넣을지 joint target baseline으로 먼저 갈지 결정
- `PingPongEEDeltaEnv` 기준 success threshold와 reward logging 정책 정리
- baseline rollout에서 쓸 성공 조건과 reward 초안 정의
- passive viewer와 baseline 출력을 연결할 간단한 디버그 HUD 또는 로깅 포맷 정리
- `run_ee_rollout_analysis.py`로 multi-episode distribution을 먼저 수집하고 threshold 조정은 그 다음에 검토
- `run_ppo_baseline.py`로 rollout analysis와 같은 schema의 PPO smoke/baseline 로그를 수집

### 5.2 그 다음 단계
- end-effector 제어 계층 추가 여부 결정
- scripted baseline controller 추가
- Gymnasium wrapper 연결
- reward/observation 설계 구체화

## 6. 이후 RL 설계 윤곽

### 6.1 현재 Observation 계약
- public observation은 flat vector `(26,)`
- 순서:
  - joint position `(7,)`
  - joint velocity `(7,)`
  - racket position `(3,)`
  - target position `(3,)`
  - ball position `(3,)`
  - ball velocity `(3,)`

### 6.2 Action 후보
- Option A: joint position target
  - 장점: 현재 구현과 직접 연결됨
  - 단점: 탐색 공간이 큼
- Option B: end-effector position control
  - 장점: 탁구 태스크에 더 직관적임
  - 단점: IK 또는 OSC 계층이 필요함

### 6.3 Reward 후보
- racket과 ball 접촉 여부
- 공 높이 유지
- 라켓 중심 타격
- action smoothness penalty

### 6.4 Episode 경계 계약
- `terminated`: failure reason 또는 success reason이 생기면 `True`
- `truncated`: failure 없이 `step_count >= max_episode_steps`면 `True`
- 현재 기본 `max_episode_steps = 300`
- 현재 기본 success 조건: racket contact + `ball_velocity_z > 0.5`

## 7. 주의할 점
- 현재 goal은 RL이 아니라 scene 안정화다.
- contact 품질과 reset 일관성이 학습 성능보다 먼저다.
- ball spawn은 현재 `racket_center` 정중앙 위 `0.22m`로 정렬되어 있고, 회귀 테스트 기준 첫 접촉은 floor보다 racket이 먼저다.
- 현재 기본 실패 원인은 `floor_contact`, `ball_out_of_bounds`, `nonfinite_state`, `ball_speed_limit`로 정리되어 있다.
- 다음 단계의 핵심은 spawn 보정보다 controller/observation 인터페이스를 고정하는 것이다.