# pingpong_rl Legacy

`pingpong_rl`은 MuJoCo 기반 로봇 탁구 keep-up을 처음 구성한 legacy 패키지다. 현재 주력 실험은 sibling 디렉터리 `../pingpong_rl2`이며, 이 디렉터리는 장면 구성, EE delta 환경, 초기 PPO 로그 구조를 참고할 때 사용한다.

## 포함 내용

| 경로 | 역할 |
| --- | --- |
| `assets/scene.xml` | Franka Panda, 라켓, 공, 테이블을 포함한 MuJoCo scene |
| `src/pingpong_rl/envs/` | scene load, bounce/contact 판정, EE delta RL 환경 |
| `src/pingpong_rl/controllers/` | joint target 보관기와 Jacobian 기반 EE pose controller |
| `src/pingpong_rl/training/` | PPO episode/step/contact logging |
| `scripts/` | viewer, baseline, PPO 학습/렌더링, rollout 분석 |
| `tests/test_scene_load.py` | scene, controller, env contract smoke test |
| `docs/` | 초기 실험 기록과 보고서 |

## 실행

```bash
cd mujoco/pingpong_rl
conda activate mujoco_env
PYTHONPATH=src python -m unittest discover -s tests
```

장면 확인:

```bash
PYTHONPATH=src mjpython scripts/run_viewer.py
```

물리 sanity check:

```bash
PYTHONPATH=src python scripts/run_bounce_baseline.py
PYTHONPATH=src python scripts/run_keepup_baseline.py
```

초기 PPO 학습:

```bash
PYTHONPATH=src python scripts/run_ppo_baseline.py --device cpu
```

저장 모델 viewer:

```bash
PYTHONPATH=src mjpython scripts/run_ppo_render.py \
  --model-path docs/etc/ppo_runs/ppo_active_hit/ppo_active_hit_model.zip
```

## 현재 판단

- `pingpong_rl`은 keep-up 문제의 초기 구조를 이해하는 데 유용하다.
- 최신 action space, curriculum, long-horizon 평가, v39/v40 artifacts는 `../pingpong_rl2`에 있다.
- 기존 training summary 안의 절대경로는 현재 `mujoco/pingpong_rl` 위치 기준으로 정리되어 있다.
