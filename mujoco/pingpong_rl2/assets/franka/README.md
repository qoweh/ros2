# Franka Panda MJCF Asset

이 디렉터리는 `pingpong_rl2/assets/scene.xml`과 `scene_racket_outward.xml`에서 include하는 Franka Emika Panda MuJoCo asset이다. 원본 Panda MJCF 설명, mesh, license, changelog를 프로젝트 안에 vendoring한 형태다.

## 프로젝트에서 쓰는 파일

- `panda.xml`, `panda_nohand.xml`, `panda_racket_outward.xml`: Panda arm과 라켓 방향 실험용 MJCF
- `assets/*.obj`, `assets/*.stl`: visual/collision mesh
- `scene.xml`: asset package 자체의 기본 scene
- `panda.png`: 원 asset preview

학습에 쓰는 탁구 scene은 상위 `../scene.xml` 또는 `../scene_racket_outward.xml`이다. 이 디렉터리의 mesh나 Panda XML을 바꾸면 두 scene의 include 관계와 웹 프로젝트의 `rl/assets/franka` 복사본도 함께 확인한다.

## 검증

```bash
cd /Users/pilt/project-collection/ros2/mujoco/pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python -m unittest discover -s tests
```

## 출처와 라이선스

Franka Emika Panda model은 공개 Franka description에서 파생된 MJCF asset이며 Apache-2.0 License로 배포된다. 자세한 원 asset 이력은 `CHANGELOG.md`와 MuJoCo Menagerie의 Panda 문서를 참고한다.
