# RL이 담당한 부분과 hand-coded control이 담당한 부분

작성 기준: 2026-06-07 로컬 저장소 기준. 최종 발표 기준 모델은 `keep1_v39_17d_mid_curriculum_fixed`다.

## 한 줄 결론

이 프로젝트의 최종 v39는 “PPO가 7개 관절을 처음부터 끝까지 알아서 움직이는 순수 end-to-end 로봇 제어”가 아니다. 사람이 설계한 contact-frame primitive, planner, Cartesian controller 위에서 PPO가 17차원 residual action을 학습하는 hybrid 구조다.

그래서 “RL이 한 일이 작다”고 느끼는 것은 어느 정도 맞다. 다만 RL이 의미 없다는 뜻은 아니다. RL은 전체 로봇 제어 스택을 담당하지 않았고, 공을 계속 살리기 위해 필요한 타격 위치, 라켓 자세, 속도, apex/timing, tracking 보정값을 담당했다.

## 내가 처음 생각했던 구조와 실제 구조

처음 기대했던 구조:

```text
공/로봇 상태 관측
  -> PPO policy
  -> 7개 관절 위치/속도/토크 직접 출력
  -> 로봇팔이 스스로 탁구 랠리 학습
```

실제 최종 v39 구조:

```text
공/로봇 상태 관측
  -> PPO policy가 17D residual action 출력
  -> hand-coded contact-frame planner가 기본 타격 목표 계산
  -> residual action이 그 목표를 조금씩 수정
  -> hand-coded Cartesian controller가 라켓 target pose/velocity를 7개 관절 target으로 변환
  -> MuJoCo 시뮬레이션 step
  -> reward/failure/curriculum으로 PPO 업데이트
```

즉 PPO가 직접 “joint1은 얼마, joint2는 얼마...”를 정한 것이 아니라, “기본 타격 구조 위에서 얼마나 보정할지”를 정했다.

## 담당 범위 비교

| 영역 | 담당 주체 | 설명 |
| --- | --- | --- |
| 공, 라켓, Panda 로봇팔 물리 | MuJoCo 환경 | 물리 시뮬레이션과 contact를 계산한다. |
| episode reset, curriculum | hand-coded 환경 설정 | 공 시작 위치/속도 범위, 난이도 확장, 종료 조건을 사람이 설계했다. |
| contact time/position 예측 | hand-coded planner | 공이 하강할 때 어디서 맞힐지 계산한다. |
| desired outgoing velocity | hand-coded primitive | 공을 어느 방향/높이로 보내야 다음 랠리가 가능한지 기본 목표를 만든다. |
| low-apex recovery, lateral brake, followthrough | hand-coded primitive | 낮게 튀는 공, lateral out, 타격 후 회복을 보정한다. |
| 라켓 target position/tilt/velocity -> 7개 joint target | hand-coded Cartesian controller | MuJoCo Jacobian 기반으로 라켓 목표를 관절 목표로 바꾼다. |
| reward, success, failure 판단 | hand-coded 환경 로직 | useful bounce, stable cycle, low apex, out-of-bounds 등을 정의한다. |
| contact-frame residual action | PPO policy | 기본 타격 목표를 state-dependent하게 수정한다. |
| 최종 랠리 성능 | hybrid 결과 | hand-coded 구조와 PPO residual이 함께 만든 결과다. |

## PPO가 실제로 출력한 것

최종 v39의 action mode는 다음이다.

```text
position_contact_frame_velocity_tilt_lateral_apex_tracking_residual
```

이 action은 17차원 residual이다.

| action 구간 | 의미 | 해석 |
| --- | --- | --- |
| `0:3` | radial/tangent/z contact-frame position residual | 라켓 목표 위치를 contact-frame 기준으로 보정한다. |
| `3:5` | pitch/roll residual | 라켓 기울기를 보정한다. |
| `5:8` | desired velocity residual | outgoing velocity의 z scale, x/y 성분을 보정한다. |
| `8` | racket z velocity residual | 타격 순간 라켓의 상하 속도를 보정한다. |
| `9:11` | trajectory/centering tilt scale residual | hand-coded tilt 계산의 강도를 조정한다. |
| `11:13` | racket x/y velocity residual | 라켓의 lateral 추적 속도를 보정한다. |
| `13` | target apex z residual | 다음 공의 목표 apex 높이를 보정한다. |
| `14` | strike plane z residual | 타격 높이 기준면을 보정한다. |
| `15:17` | tracking x/y velocity residual | 공이 하강 중일 때 pre-contact XY tracking 속도를 보정한다. |

이것이 “RL이 담당한 부분”이다. 관절 7개를 직접 제어한 것이 아니라, 사람이 만든 타격 구조를 매 step마다 상태에 맞게 조정한 것이다.

## hand-coded 영향이 컸던 이유

탁구 keep-up은 random exploration으로 배우기 어렵다.

- 공을 라켓에 맞히는 contact 자체가 드물다.
- 맞혀도 useful bounce가 되려면 방향, 높이, lateral 안정성, 다음 intercept까지 맞아야 한다.
- 관절 7개를 직접 움직이면 action space와 credit assignment가 너무 커진다.
- 실패 대부분은 바닥 접촉, low apex, ball out, robot body contact처럼 빠르게 끝난다.

그래서 이 프로젝트는 “PPO가 모든 것을 발명하게 하기”보다 “탁구가 가능할 법한 구조를 사람이 만들고, PPO가 그 안에서 보정하도록 하기”로 진화했다.

이 방향은 최종적으로 꽤 명확하게 코드에 남아 있다. `RacketCartesianController`가 라켓 목표 위치/기울기/속도를 7개 관절 target으로 바꾸고, `KeepUpEnv.step()`은 PPO action을 contact-frame residual로 해석한 뒤 controller에 target을 넘긴다.

## 그럼 RL은 별로 한 게 없는가?

아니다. 다만 RL의 역할이 원래 상상했던 것보다 좁고 구조화되어 있다.

RL이 하지 않은 것:

- 7개 관절의 torque를 직접 출력하지 않았다.
- raw joint control을 처음부터 학습하지 않았다.
- 공을 어디서 맞히고 어떤 기본 반동을 만들지 완전히 새로 발견하지 않았다.
- low-level inverse kinematics/controller를 학습하지 않았다.

RL이 한 것:

- contact-frame 기준 타격 위치를 상태에 따라 조정했다.
- 라켓 pitch/roll을 보정했다.
- outgoing velocity, racket velocity, target apex, strike plane을 조정했다.
- pre-contact tracking 속도를 보정했다.
- hand-coded primitive만으로 부족한 장기 랠리 안정성을 PPO fine-tuning으로 끌어올렸다.

발표에서는 이렇게 말하는 것이 가장 정확하다.

> 본 프로젝트는 완전한 end-to-end joint-control RL이 아니라, hand-coded contact primitive와 Cartesian controller 위에 PPO residual policy를 얹은 구조입니다. PPO는 로봇팔의 모든 관절 움직임을 직접 생성한 것이 아니라, 공의 상태에 따라 타격 위치, 속도, 라켓 자세, apex/timing, tracking 보정값을 학습했습니다.

## “환경만 만들면 RL이 알아서 탁구를 배운다”와 다른 점

환경만 설계하고 PPO가 7개 관절을 직접 움직이게 하는 방식도 이론적으로는 가능하다. 하지만 이 프로젝트에서는 그 방식으로 끝까지 간 것이 아니다.

순수 end-to-end joint-control로 가려면 보통 다음이 필요하다.

- joint torque/position action space 재정의
- 훨씬 큰 학습량
- sparse reward를 버틸 수 있는 curriculum 또는 imitation data
- 더 강한 exploration 전략
- contact-rich task에 맞는 안정적인 low-level control
- sim-to-real까지 고려한다면 domain randomization과 safety constraint

이 프로젝트는 졸업 프로젝트 시간 안에서 안정적인 결과를 내기 위해, end-to-end joint-control보다 residual RL 구조를 선택한 것으로 해석하는 편이 맞다.

## heuristic과의 관계

heuristic은 두 층으로 봐야 한다.

1. `HeuristicKeepUpPolicy` 같은 heuristic bootstrap은 초기에 actor를 warm-start하는 임시 보조 장치였다.
2. contact-frame primitive와 Cartesian controller는 최종 v39에도 남아 있는 영구적인 구조다.

그래서 “거푸집” 비유는 heuristic bootstrap에는 잘 맞는다. 초기에 policy가 이상한 action 분포에서 시작하지 않게 형태를 잡아주고, 최종 v39 학습에서는 직접 사용되지 않았다.

반면 contact-frame primitive는 거푸집이라기보다 골조에 가깝다. 최종 모델에서도 PPO는 이 골조 위에서 residual을 학습한다.

## 발표 질의응답용 짧은 답변

질문: “RL이 로봇팔을 직접 제어한 건가요?”

답변:

> 직접 관절 7개를 제어한 것은 아닙니다. 라켓의 기본 타격 목표와 관절 제어는 hand-coded planner/controller가 담당하고, PPO는 그 위에서 17차원 residual action을 출력해 타격 위치, 라켓 자세, 속도, apex/timing, tracking을 보정했습니다.

질문: “그러면 hand-coded가 대부분이고 RL은 작은 역할 아닌가요?”

답변:

> 전체 제어 스택 기준으로는 hand-coded 비중이 큽니다. 하지만 장기 랠리 성능은 고정된 hand-coded controller만으로 나온 것이 아니라, contact-frame 구조 안에서 PPO가 상태별 residual을 학습하면서 나온 결과입니다. 이 프로젝트의 핵심은 순수 end-to-end RL이 아니라, 문제를 residual control 문제로 재정의한 데 있습니다.

질문: “환경만 만들면 PPO가 알아서 탁구를 배울 줄 알았는데 아닌가요?”

답변:

> 이론적으로는 가능하지만, 이 과제에서는 contact가 드물고 useful bounce 조건이 까다로워서 random exploration만으로 7개 관절 제어를 배우기 어렵습니다. 그래서 사람이 물리적으로 가능한 타격 구조를 만들고, PPO가 그 안에서 필요한 보정을 배우도록 설계했습니다.

## 근거 링크

- [action_modes.py](../src/pingpong_rl2/envs/action_modes.py:3): 최종 action mode가 residual 계열임을 보여준다.
- [keepup_env.py step](../src/pingpong_rl2/envs/keepup_env.py:1490): PPO action을 residual로 해석하고 controller target으로 넘기는 흐름.
- [keepup_env.py contact-frame residual helpers](../src/pingpong_rl2/envs/keepup_env.py:2109): velocity, tilt, apex, tracking residual 함수들.
- [keepup_env.py contact-frame planner](../src/pingpong_rl2/envs/keepup_env.py:2197): contact position, target apex, desired outgoing velocity를 hand-coded로 계산하는 부분.
- [ee_pose_controller.py](../src/pingpong_rl2/controllers/ee_pose_controller.py:7): 라켓 target pose/velocity를 7개 joint target으로 바꾸는 Cartesian controller.
- [report 49](report/49_racket_center_tracking_spin_and_two_ball_plan.md:3): v25/v26 기준 hand-coded 부분과 residual RL 역할을 명시한 보고서.
- [report 21](report/21_contact_frame_primitive_report.md:130): contact-frame primitive가 residual-RL action mode로 확장된 배경.
- [v39 training summary](../artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json:1): v39가 v36에서 resume된 최종 학습 결과.
- [v39 bootstrap null](../artifacts/ppo_runs/keep1_v39_17d_mid_curriculum_fixed/keep1_v39_17d_mid_curriculum_fixed_training_summary.json:453): 최종 v39 학습에 bootstrap이 직접 들어가지 않았다는 근거.
