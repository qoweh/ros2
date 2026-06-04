# keep2_v1 학습 분석과 stagger reset 계획

작성일: 2026-06-04

## 학습 결과 요약

100 episode deterministic 분석 결과:

- mean useful bounces: `4.71`
- max useful bounces: `24`
- mean steps: `59.7`
- max steps: `197`
- mean contacts: `20.61`
- max contacts: `140`

실패 원인:

- out-of-bounds: `71/100`
- speed limit: `12/100`
- robot body contact: `8/100`
- floor contact: `5/100`
- ball-ball contact: `4/100`

## 위로 올려치는 모션

정책 rollout을 instrument해서 contact 순간을 확인했다.

- 전체 contact outgoing `vz` median: `0.145 m/s`
- non-useful contact outgoing `vz` median: `0.085 m/s`
- useful contact outgoing `vz` median: `0.656 m/s`
- 전체 contact projected apex median: `0.144 m`
- useful contact projected apex median: `0.161 m`
- 목표 apex: `0.24 m`
- contact 순간 racket `vz` median: `0.017 m/s`
- contact 순간 racket `vz` p75: `0.089 m/s`

따라서 viewer에서 보이는 "위로 치는 힘이 약하다"는 관찰이 맞다. 정책 command는 평균적으로 upward target velocity를 만들고 있지만, 실제 contact 순간 라켓 vertical velocity가 충분히 올라오지 않는다. 또 action의 `target_apex_z_residual` 평균이 약간 음수라 정책도 적극적으로 더 높게 치는 방향을 고르지 않는다.

## 현재 reset phase 문제

현재 reset base height는 `[0.40, 0.67]`이고, 이론상 첫/두 번째 strike plane 도달 시간 차이는 대략 `0.10s`다.

학습된 정책 rollout 중 두 공 모두 valid intercept를 가진 순간의 phase gap:

- mean: `0.127s`
- p10: `0.039s`
- p50: `0.093s`
- p75: `0.134s`

control dt가 `0.02s`이므로 median gap은 약 5 control step뿐이다. 2공을 번갈아 치기에는 너무 빡빡하고, 라켓이 다음 공을 향해 이동/상향 가속할 시간이 부족하다.

## stagger reset 후보

true delayed spawn도 가능하지만, inactive ball mask와 termination 예외 처리가 필요하다. 다음 실험은 먼저 두 공을 모두 spawn하되 높이 차이를 크게 벌리는 방식이 안전하다.

base height 후보별 초기 intercept gap:

- `[0.40, 0.67]`: median `0.099s`
- `[0.36, 0.85]`: median `0.170s`
- `[0.35, 1.00]`: median `0.213s`
- `[0.35, 1.15]`: median `0.248s`
- `[0.35, 1.30]`: median `0.280s`

추천 다음 실험:

- `reset_base_heights = [0.35, 1.15]`
- config: `configs/keep2_v2_staggered.json`
- 초기에는 `reset_spin_range`를 줄이거나 0으로 두고, useful bounce가 늘면 다시 spin을 켠다.
- target apex는 `0.24m` 유지로 시작한다. 높이 보상/목표를 동시에 바꾸면 phase 개선 효과와 섞인다.

## 판단

하나가 먼저 떨어지고 다음 공이 뒤따라 떨어지는 방식으로 가는 것이 맞다. 다만 첫 구현은 true delayed spawn보다 height-stagger curriculum이 좋다. 모델/observation 구조를 바꾸지 않고 phase gap만 늘릴 수 있어서 원인 분리가 쉽다.
