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
  -> utils.resolve_input_path
```

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 contact 물성을 확인하는 sanity script다. `PingPongSim`을 만들고 XML에 정의된 `ball_geom`, `racket_head`의 geom/material 관련 값을 읽은 뒤, 고정된 라켓 위로 공을 떨어뜨려 유효 반발계수를 계산한다.

```text
run_material_sanity.py
  -> resolve_input_path(scene_path)
  -> PingPongSim(scene_path)
  -> geom_summary(sim, "ball_geom")
  -> geom_summary(sim, "racket_head")
  -> for episodes
       run_static_racket_drop()
         sim.reset()
         sim.reset_ball_above_racket()
         while max_substeps
           sim.step(joint_targets=home_joint_targets, n_substeps=1)
           ball/racket contact 시작 감지
           contact 종료 감지
           normal 방향 상대속도 전후 비교
  -> JSON summary 출력
```

`geom_summary()`는 MuJoCo model에서 geom size, body mass, friction, solref, solimp를 읽는다. 이 값들은 XML material/contact 설정이 실제 모델에 어떻게 들어갔는지 확인하는 용도다.

`run_static_racket_drop()`은 contact 시작 직전의 공-라켓 상대속도와 contact 종료 직후의 상대속도를 contact normal 방향으로 비교한다. 계산식은 다음과 같이 볼 수 있다.

```text
effective_normal_restitution
  = post_contact_relative_normal_speed
    / abs(pre_contact_relative_normal_speed)
```

이 값은 XML에 적힌 restitution 값을 그대로 읽은 것이 아니라, 실제 MuJoCo step을 돌렸을 때 관측되는 유효 반발 정도다. contact solver, solref/solimp, timestep, substep 설정의 영향을 함께 받는다.

## 학습 코드와의 관계

PPO 학습에서 reward가 아무리 잘 설계되어도 contact 물성이 지나치게 이상하면 공이 원하는 방식으로 튀지 않는다. 이 파일은 그런 문제를 학습 전에 분리해서 보는 용도다. 즉, policy가 못해서 공이 안 뜨는지, contact 물성이 낮아서 공이 안 튀는지 구분하는 데 도움이 된다.

## 발표 때 설명 포인트

- 보상보다 먼저 물리 모델이 타당한지 설명할 때 쓸 수 있다.
- 반발계수나 접촉 파라미터를 바꾼 실험이 있다면 baseline 대비 차이를 표로 만들기 좋다.
