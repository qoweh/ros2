# MuJoCo 씬 구성 도움말

이 문서는 현재 구성한 MuJoCo 씬에서 처음 보면 낯선 부분만 짧게 정리한 설명 문서다.

## 1. 왜 `panda.xml`의 `meshdir`를 바꿨는가

Franka menagerie의 `panda.xml`은 원래 자기 파일 기준으로 `assets` 폴더를 찾는다.

하지만 이번 프로젝트에서는 상위 scene 파일인 `pingpong_rl/assets/scene.xml`에서 `franka/panda.xml`을 include했다. 이 경우 top-level scene 기준으로 상대 경로가 해석되어 메시 파일을 잘못 찾을 수 있다.

그래서 현재 프로젝트 복사본에서는 다음처럼 바꿨다.

- 기존: `meshdir="assets"`
- 변경: `meshdir="franka/assets"`

의미:
- top-level scene 기준으로도 Franka 메시 파일을 정확히 찾게 만든 것이다.

## 2. 왜 라켓을 `hand` body 자식으로 붙였는가

선택지는 크게 두 가지였다.

### 2.1 자식 body로 직접 부착
- 장점:
  - 단순하다.
  - hand를 그대로 따라간다.
  - 초기 디버깅이 쉽다.
- 단점:
  - robot asset 내부를 직접 수정하게 된다.

### 2.2 별도 body + weld constraint
- 장점:
  - 라켓 자산을 더 독립적으로 다룰 수 있다.
- 단점:
  - 초기 위치/자세 설정이 더 번거롭다.
  - scene 안정화 전에는 오히려 복잡도를 높인다.

현재 단계는 물리 확인이 우선이라 direct child 방식을 사용했다.

## 3. 공 spawn/reset은 어떻게 동작하는가

탁구공은 `freejoint`를 가진 body다.

freejoint는 상태를 크게 두 묶음으로 가진다.

- `qpos`
  - 위치 3개
  - 자세 quaternion 4개
- `qvel`
  - 선속도 3개
  - 각속도 3개

그래서 ball reset 시에는 아래를 함께 바꿔야 한다.

- 위치를 원하는 spawn 지점으로 설정
- quaternion을 `[1, 0, 0, 0]`으로 초기화
- 속도를 원하는 값으로 설정
- 각속도는 0으로 초기화

## 4. joint control은 어떤 방식인가

Franka menagerie asset에는 actuator가 이미 들어 있다.

현재 코드는 `data.ctrl`에 joint target을 넣는 방식으로 arm을 제어한다.

즉 지금 단계는 다음을 먼저 확인하는 단계다.

- 원하는 joint target이 들어가는가
- viewer에서 arm이 안정적으로 움직이는가
- ball과 racket의 충돌이 기본적으로 성립하는가

아직 end-effector 제어기나 RL wrapper는 붙이지 않았다.

## 5. 지금 남아 있는 핵심 튜닝 포인트

### 5.1 라켓 contact 면
- 초기에 `racket_head`는 `ellipsoid`였다.
- centered drop에서도 곡면 때문에 공이 옆으로 흐르기 쉬워서 현재는 얇은 `cylinder` contact geom으로 바꿨다.
- 이 변경 후 기준 rollout에서 공의 첫 target contact는 floor가 아니라 racket으로 유지되고, 첫 floor contact 시점도 더 늦어졌다.

### 5.2 contact 반발감
- MuJoCo에서 실제 반발 느낌은 단순한 restitution 한 값으로 끝나지 않는다.
- `solref`, `solimp`, `friction`, 질량 조합이 같이 영향을 준다.
- 현재는 paddle을 flat하게 만든 1차 튜닝만 끝난 상태고, 다음엔 scripted baseline에 맞춰 반발 궤적을 더 정교하게 볼 단계다.

### 5.3 reset robustness
- 공이 바닥에 닿았을 때
- 유효 workspace 밖으로 멀어졌을 때
- 비정상 속도/자세로 드리프트할 때

이런 경우 reset 조건을 더 보강해야 한다.

## 8. 왜 라켓이 집게 아래에 매달린 것처럼 보였는가

초기 구현에서는 `racket` body 원점을 paddle head 쪽에 가깝게 두었다.

그 상태에서 `reset_ball_above_racket()`가 `racket` body 원점 기준으로 공을 spawn하니 다음 문제가 같이 생겼다.

- 라켓이 손가락 사이가 아니라 집게 아래에 매달린 것처럼 보임
- 공도 손에 가까운 이상한 위치에 있는 것처럼 보임

이번 수정에서는 역할을 분리했다.

- `racket` body 원점:
  - 실제로 손가락이 잡는 손잡이 위치
- `racket_center` site:
  - paddle head 중심
  - 공 spawn과 타격 기준점으로 사용

즉 지금은 grip 위치와 strike 위치를 분리해서 해석한다.

정리하면:
- 손가락 사이에 있는 것은 `racket` body 원점이다.
- 공은 `racket_center` 위에서 생성된다.
- 예전에 집게 아래 공처럼 보이던 것은 실제 공이 아니라 paddle head가 섞여 보인 영향이 컸다.

## 9. 현재 라켓 배치 원칙

이번 수정 후 라켓은 다음 원칙으로 배치한다.

- 손잡이 축:
  - 지면과 거의 평행
  - 손가락 사이를 통과
- paddle head:
  - 손잡이 한쪽에 연결된 한 덩어리
  - grip 원점과 분리된 `racket_center`를 가짐
- 공 초기 위치:
  - `racket_center` 바로 위
  - reset 직후 로봇/라켓과 접촉하지 않음

즉 지금은 “집게가 손잡이를 옆에서 물고 있는 형태”로 정리한 상태다.

## 6. 왜 viewer 키가 달랐는가

MuJoCo viewer에는 크게 두 가지 실행 방식이 있다.

### 6.1 interactive mode
- 현재 프로젝트에서는 stock MuJoCo viewer CLI를 subprocess로 실행한다.
- macOS 환경에서 in-memory `launch(model, data)` 경로보다 scene path를 직접 넘기는 쪽이 더 안정적이었다.
- 기본 키 동작을 기대할 때 이쪽이 맞다.

### 6.2 passive mode
- `mujoco.viewer.launch_passive(...)`
- Python 코드가 바깥에서 직접 step을 돌린다.
- 그래서 pause/run 같은 일부 viewer 키 동작이 직관과 다를 수 있다.
- 대신 자동 reset, scripted joint motion 같은 커스텀 루프를 넣기 쉽다.

이번 수정 후 기본 viewer 실행은 stock interactive viewer 위임이고, custom Python loop 확인이 필요할 때만 `--mode passive`를 쓴다.

## 7. 왜 `pip install -e .`를 추가했는가

기존에는 루트에 `pyproject.toml`이나 `setup.py`가 없어서 editable install이 불가능했다.

이번에는 루트에 `pyproject.toml`을 추가해서 아래가 가능해졌다.

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
python -m pip install -e .
```

설치 후에는 아래처럼 실행할 수 있다.

```bash
pingpong-rl-viewer
python -m pingpong_rl.viewer
```