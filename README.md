# ROS2 / MuJoCo Project Collection

이 저장소는 ROS 2/Gazebo 환경 구축 메모와 MuJoCo 기반 로봇 탁구 강화학습 실험을 함께 보관한다. 이전 단일 프로젝트 구조는 현재 아래처럼 분리되어 있다.

| 경로 | 역할 |
| --- | --- |
| `gazebo-ros2/` | ROS 2와 Gazebo 설치, 검증, VS Code 설정 메모 |
| `mujoco/` | Franka Panda 탁구 keep-up/self-rally 강화학습 실험 |
| `mujoco/pingpong_rl/` | 첫 번째 MuJoCo keep-up 실험. 현재는 참고용 legacy 코드 |
| `mujoco/pingpong_rl2/` | 현재 기준 핵심 강화학습 코드와 v26~v40 실험 artifacts |
| `mujoco/md-docs/` | 발표, handoff, 초기 제안서, 최종 정리 문서 |

웹 시연/배포 코드는 sibling 저장소인 `/Users/pilt/project-collection/pingpong`에서 관리한다.

## 현재 기준

- 주력 실험 패키지: `mujoco/pingpong_rl2`
- 주력 task: 로봇팔 끝 라켓으로 공을 계속 위로 받아 올리는 keep-up/self-rally
- 주력 모델 계열: `keep1_v39_17d_mid_curriculum_fixed`
- 후속 polish 후보: `keep1_v40_17d_v39_polish`
- 권장 Python 환경: conda `mujoco_env`

## 빠른 시작

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
conda activate mujoco_env
python -m pip install -e .
```

전체 테스트:

```bash
PYTHONPATH=src conda run -n mujoco_env python -m unittest discover -s tests
```

viewer 확인:

```bash
PYTHONPATH=src mjpython scripts/run_viewer.py \
  --model-path artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --episodes 5
```

headless 분석:

```bash
PYTHONPATH=src conda run -n mujoco_env python scripts/run_ppo_rebound_analysis.py \
  --model-path artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_model.zip \
  --episodes 20 \
  --episode-step-limit 7200 \
  --analysis-name keep1_v39_readme_check
```

## 주요 문서

- `mujoco/pingpong_rl2/README.md`: 현재 학습 패키지 실행과 구조
- `mujoco/pingpong_rl2/docs/report/00_index.md`: 실험 보고서 인덱스
- `mujoco/pingpong_rl2/docs/rl_presentation_pack/README.md`: 발표용 자료 묶음
- `mujoco/pingpong_rl2/docs/script_entrypoints/README.md`: 실행 스크립트별 역할
- `gazebo-ros2/setup/README.md`: ROS 2/Gazebo 설치 흐름

## 검증 메모

경로 변경 후 아래 항목을 기준으로 확인한다.

- `mujoco/pingpong_rl2` unit test 전체
- legacy `mujoco/pingpong_rl` scene/controller smoke test
- Python 파일 compile
- sibling `pingpong` 저장소의 backend preflight와 frontend build

생성물 경로는 가능한 한 패키지 루트 기준 상대경로를 사용한다. 절대경로가 필요한 기록성 파일에서는 `/Users/pilt/project-collection/ros2/mujoco/...` 기준을 사용한다.
