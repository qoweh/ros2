# pingpong_rl3 keep2_v1 설계 정리

`pingpong_rl3`는 `pingpong_rl2`의 1-ball 실험 누적 코드를 그대로 복사하지 않고, 공 2개 keep-up만 남긴 새 프로젝트다.

## 가져온 것

- Franka/racket MuJoCo asset
- 라켓 end-effector Cartesian controller
- Gymnasium/SB3 vector env adapter
- 탁구공 질량, 크기, 접촉 파라미터는 `rl2/assets/scene.xml` 기준을 유지

## 제외한 것

- 긴 preset 목록
- checkpoint 저장 로직
- 1-ball 실험별 action mode 분기
- 과거 분석/진단 스크립트 묶음
- 쓰지 않는 CLI 옵션 대부분

## 2-ball 환경 핵심

- XML에 `ball_0`, `ball_1` freejoint 공을 둔다.
- reset 때 두 공을 서로 다른 높이/slot에서 떨어뜨려 phase 차이를 만든다.
- 환경은 예측 낙하지점과 낙하 시간을 기준으로 먼저 칠 공을 고른다.
- RL action은 hand-coded scheduler 위에 residual을 더한다.

## RL action 13D

1. `contact_x_residual`
2. `contact_y_residual`
3. `contact_z_residual`
4. `tilt_pitch`
5. `tilt_roll`
6. `racket_vx_residual`
7. `racket_vy_residual`
8. `racket_vz_residual`
9. `outgoing_vx_residual`
10. `outgoing_vy_residual`
11. `target_apex_z_residual`
12. `ball_0_priority_bias`
13. `ball_1_priority_bias`

## Reward 기준

- 공이 라켓에 닿으면 contact reward를 준다.
- useful bounce는 projected apex가 높이/xy 범위에 들어오고, outgoing z 속도가 충분할 때만 센다.
- 두 공을 번갈아 useful contact하면 alternating bonus를 준다.
- floor/out-of-bounds/robot-body/ball-ball contact는 실패로 종료한다.

## 현재 의도

`keep2_v1`은 완성 모델이 아니라 공 2개 keep-up을 학습시킬 수 있는 깨끗한 baseline이다. 처음에는 reset 범위를 좁게 두고 두 공 phase를 강하게 분리했으며, 이후 학습 결과를 보고 `reset_xy_range`, `slot_xy_offsets`, `target_apex_height`, `reachable_radius`를 넓히는 식으로 커리큘럼을 잡는 것이 안전하다.
