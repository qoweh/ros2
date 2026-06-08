# run_material_sanity.py

## 한 줄 역할

`scripts/run_material_sanity.py`는 scene XML의 ball/racket material, friction, solref/solimp와 라켓 반발 실험 결과를 JSON으로 확인하는 sanity script다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_material_sanity.py \
  --scene-path assets/scene.xml \
  --episodes 5
```

## 코드 흐름

1. scene path를 resolve한다.
2. `PingPongSim(scene_path=scene_path)`를 만든다.
3. `geom_summary()`로 `ball_geom`, `racket_head`의 크기, 질량, friction, solref, solimp를 읽는다.
4. `run_static_racket_drop()`로 라켓 위에 공을 떨어뜨린다.
5. 접촉 전후 normal 방향 상대속도에서 effective restitution을 계산한다.
6. scene/material 정보와 drop test 결과를 JSON으로 출력한다.

## 주요 호출 관계

```text
run_material_sanity.py
  -> envs/pingpong_sim.py
  -> utils/pathing.py 또는 utils export
```

## 발표 때 설명 포인트

- 보상보다 먼저 물리 모델이 타당한지 설명할 때 쓸 수 있다.
- 반발계수나 접촉 파라미터를 바꾼 실험이 있다면 baseline 대비 차이를 표로 만들기 좋다.
