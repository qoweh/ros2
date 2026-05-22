# Reward Logging과 Rollout Analysis 도움말

이 문서는 이번 단계에서 추가한 두 가지를 설명한다.

- 왜 reward decomposition logging이 지금 필수인가
- headless rollout export를 어떻게 써서 success/failure/contact 분포를 볼 수 있는가

## 1. 왜 reward decomposition logging을 지금 넣는가

현재 단계의 위험은 reward 숫자만 보면 좋아 보이는데 실제 behavior는 전혀 원하는 방향이 아닐 수 있다는 점이다.

대표적인 경우:
- 공을 실제로 치지 않음
- success는 거의 없음
- 그런데 height term만 계속 쌓여 episode reward가 높게 보임

그래서 이번 단계에서는 reward 식을 바꾸지 않고, reward가 어디서 나왔는지만 바로 보이게 만들었다.

현재 `info`에는 아래가 들어간다.

- `reward_total`
- `reward_height`
- `reward_distance`
- `reward_contact`
- `reward_success`
- `reward_failure`

중요한 점:
- `reward_distance`는 현재 reward 식에 없어서 `0.0`
- `reward_success`도 현재 success bonus가 없어서 `0.0`

즉 이번 변경은 reward semantics 변경이 아니라, 이후 항이 추가되더라도 log schema가 흔들리지 않게 미리 칸을 고정한 것이다.

## 2. episode 종료 원인은 어떻게 같이 보나

현재 `info`에는 아래 종료 메타데이터도 같이 들어간다.

- `terminated`
- `truncated`
- `success_reason`
- `failure_reason`
- `time_limit_reached`
- `episode_steps`

의미:
- 실패 종료면 `terminated=True`, `failure_reason` 기록
- 성공 종료면 `terminated=True`, `success_reason` 기록
- 시간 종료면 `truncated=True`, `time_limit_reached=True`

즉 rollout log만 봐도 episode가 왜 끝났는지 바로 분해해서 볼 수 있다.

## 3. contact velocity distribution은 왜 substep trace로 잡는가

env 한 step은 MuJoCo physics substep 여러 개를 묶는다.

그래서 단순히 step 마지막 상태만 보면:
- contact가 step 중간에 잠깐 있었더라도
- 마지막 substep에서 contact가 풀려 있으면
- contact를 놓칠 수 있다.

이번에는 이 문제를 피하려고 `PingPongSim.step_with_contact_trace()`를 추가했다.

이 trace는 reward/success 판정을 바꾸지 않고, logging용으로만 아래를 잡는다.

- `contact_observed_during_step`
- `contact_substep`
- `contact_ball_velocity_x`
- `contact_ball_velocity_y`
- `contact_ball_velocity_z`
- `contact_ball_speed_norm`

즉 reward semantics는 그대로 두고, 분석용 관측만 더 촘촘하게 만든 것이다.

## 4. 새 rollout analysis 스크립트는 무엇을 내보내는가

실행 스크립트:

```bash
conda activate mujoco_env
cd /Users/pilt/project-collection/ros2/graduation-prj
python pingpong_rl/scripts/run_ee_rollout_analysis.py --episodes 10 --max-episode-steps 300
```

기본 출력 위치:

- `pingpong_rl/docs/etc/rollout_analysis`

기본 산출물:

- `*_episodes.csv`
- `*_steps.csv`
- `*_contacts.csv`
- `*_summary.json`

## 5. 각 파일의 역할

`episodes.csv`
- episode 단위 집계
- 종료 원인, step 수, contact 수, reward 누적 합 확인용

`steps.csv`
- step 단위 reward decomposition
- 어떤 reward 항이 실제로 누적되는지 확인용

`contacts.csv`
- contact 순간의 velocity 표본
- threshold 조정 전에 분포를 먼저 보는 용도

`summary.json`
- contact velocity 분포 요약
- 현재는 다음 통계를 자동 계산한다.
  - `p50`
  - `p75`
  - `p90`
  - `max`

## 6. 이번 단계에서 일부 항이 0으로 남는 이유

사용자 제약 때문에 이번 단계에서는 다음을 하지 않았다.

- reward coefficient 수정
- reward 식 변경
- success threshold 자동 수정
- physics parameter 수정

그래서 현재 로그에 있는 일부 항은 비어 있거나 0일 수 있다.

하지만 그 상태가 오히려 중요하다.

예를 들어:
- `reward_contact=0.0`
- `reward_success=0.0`
- `reward_height`만 큼

이면 policy가 실제로는 성공하지 못하고 있다는 걸 바로 확인할 수 있다.

## 7. 지금 이 로그로 바로 볼 수 있는 것

- success가 실제로 늘고 있는지
- 실패 원인이 주로 무엇인지
- time limit으로만 끝나는지
- contact 순간 `ball_velocity_z`가 threshold `0.5`보다 어느 정도 위아래에 있는지

즉 이 단계의 목적은 튜닝이 아니라, 튜닝 전에 관측해야 할 신호를 빠짐없이 모으는 것이다.