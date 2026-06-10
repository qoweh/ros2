# Franka Panda MJCF Asset

이 디렉터리는 `pingpong_rl/assets/scene.xml`에서 include하는 Franka Emika Panda MuJoCo asset이다. 원본은 MuJoCo Menagerie 계열의 Panda MJCF 설명을 기반으로 하며, `LICENSE`와 `CHANGELOG.md`를 함께 보관한다.

## 프로젝트에서 쓰는 파일

- `panda.xml`, `panda_nohand.xml`: Panda arm MJCF 본체
- `scene.xml`: asset package의 기본 scene
- `assets/*.obj`, `assets/*.stl`: visual/collision mesh
- `panda.png`: 원 asset preview

`pingpong_rl`의 실제 탁구 장면은 상위 `../scene.xml`이다. 이 asset 내부 파일을 바꿀 때는 `../scene.xml`의 include 경로와 mesh path가 함께 유지되는지 확인한다.

## 검증

```bash
cd mujoco/pingpong_rl
PYTHONPATH=src conda run -n mujoco_env python -m unittest discover -s tests
```

## 출처와 라이선스

Franka Emika Panda model은 공개 Franka description에서 파생된 MJCF asset이며 Apache-2.0 License로 배포된다. 자세한 변환 이력은 `CHANGELOG.md`와 원본 MuJoCo Menagerie 문서를 참고한다.
