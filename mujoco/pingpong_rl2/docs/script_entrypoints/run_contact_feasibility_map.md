# run_contact_feasibility_map.py

## 한 줄 역할

`scripts/run_contact_feasibility_map.py`는 heuristic controller의 contact 관련 하이퍼파라미터 조합을 grid sweep해서 어떤 설정이 keep-up에 유리한지 찾는 진단 entrypoint다.

## 대표 실행 형태

```bash
conda run -n mujoco_env env PYTHONPATH=src python scripts/run_contact_feasibility_map.py \
  --analysis-name contact_feasibility_map_v1 \
  --coarse-episodes 1 \
  --top-k 3 \
  --finalist-episodes 30
```

## 코드 흐름

1. sweep 범위를 CLI로 받는다.
   - pitch, roll, strike z boost, followup lift boost, contact offset ratio 후보를 받는다.

2. coarse grid를 만든다.
   - `itertools.product()`로 모든 후보 조합을 만든다.
   - 각 조합마다 `build_env_kwargs()`로 환경 설정을 만든다.

3. 각 후보를 heuristic으로 평가한다.
   - `evaluate_configuration()`이 `PingPongKeepUpGymEnv`와 `HeuristicKeepUpPolicy`를 만든다.
   - PPO는 사용하지 않는다.
   - contact마다 outgoing velocity error, racket velocity, normal alignment, next intercept 정보를 모은다.

4. coarse 결과를 정렬한다.
   - `summary_sort_key()`는 max useful bounce, 3회 이상 rate, 평균 useful bounce, useful contact rate, outgoing error를 기준으로 점수를 매긴다.

5. finalist를 재평가한다.
   - coarse 상위 `top_k`만 더 많은 episode로 다시 실행한다.

6. 산출물을 저장한다.
   - `<analysis_name>_summary.json`
   - `<analysis_name>_config_rows.csv`
   - `<analysis_name>_contacts.csv`

## 주요 호출 관계

```text
run_contact_feasibility_map.py
  -> run_heuristic_keepup_diagnostic.write_csv
  -> controllers/heuristic_keepup.py
  -> envs/gym_env.py
```

## 호출을 계속 파고 들어가면 보이는 구조

이 파일은 하나의 heuristic 설정을 평가하는 것이 아니라, 여러 contact-control 후보를 grid search로 훑는다. PPO 학습 전에 어떤 tilt, z boost, followup lift, contact offset 조합이 반복 타격에 유리한지 확인하는 사전 실험용이다.

```text
run_contact_feasibility_map.py
  -> parse_args()
  -> pitch/roll/strike_z_boost/followup_lift/contact_offset 후보 grid 생성
  -> for coarse config
       build_env_kwargs()
       evaluate_configuration(stage="coarse")
  -> coarse 결과를 summary_sort_key로 정렬
  -> top_k 후보만 finalist episode 수로 재평가
  -> summary.json / config_rows.csv / contacts.csv 저장
```

`evaluate_configuration()` 안에서는 매 후보마다 새 `PingPongKeepUpGymEnv`와 `HeuristicKeepUpPolicy`를 만든다. policy는 PPO가 아니라 hand-coded heuristic이다. 각 episode에서 heuristic action을 넣고, contact event가 생기면 outgoing velocity, racket velocity, contact normal, apex, next intercept 관련 값을 contact row로 남긴다.

## coarse와 finalist 단계

coarse 단계는 넓은 후보를 적은 episode로 빠르게 훑는 단계다. 기본값 기준으로 pitch 값, roll 값, strike z boost, followup lift boost, contact offset ratio의 모든 조합을 `itertools.product()`로 만든다.

각 후보의 결과는 `summary_sort_key()`로 정렬된다. 정렬 기준은 대략 다음 순서다.

```text
max_useful_bounces
three_or_more_useful_bounce_rate
two_or_more_useful_bounce_rate
mean_useful_bounces
useful_contact_rate
outgoing_velocity_error가 작을수록
```

finalist 단계는 coarse에서 살아남은 상위 `top_k` 후보만 더 많은 episode로 다시 실행한다. coarse는 운 좋은 episode 하나 때문에 과대평가될 수 있으므로, finalist에서 더 안정적인 표본으로 다시 확인하는 구조다.

## 산출물 해석

`config_rows.csv`는 설정별 요약이다. pitch, roll, strike z boost, followup lift boost, contact offset ratio와 함께 mean/max useful bounce, useful contact rate, outgoing velocity error 같은 값이 들어간다. 여러 후보를 표로 비교할 때 이 파일을 본다.

`contacts.csv`는 각 후보의 contact별 원자료다. 어떤 설정에서 라켓 z velocity가 높았는지, 실제 outgoing z가 목표와 얼마나 달랐는지, contact normal이 라켓 면과 얼마나 정렬됐는지 볼 수 있다.

`summary.json`에는 best coarse, best finalist, pass row가 들어간다. `shows_three_plus`와 `meets_three_plus_rate_target`은 "이 heuristic primitive가 최소한 반복 타격 가능성을 보였는가"를 빠르게 판단하는 flag다.

## PPO 학습과의 관계

이 파일은 PPO를 학습하지도 평가하지도 않는다. 하지만 contact primitive를 설계할 때 매우 중요하다. PPO에게 아무 구조 없이 7개 관절을 알아서 움직이라고 한 것이 아니라, 사람이 설계한 contact-frame/strike primitive 위에서 residual을 학습하게 했기 때문이다.

따라서 feasibility map은 "어떤 기본 타격 계획이 학습 전에 물리적으로 가능했는가"를 보여준다. 이후 PPO는 이 기본 구조를 observation과 reward에 맞춰 residual action으로 보정한다.

## 발표 때 설명 포인트

- PPO 학습 전 “어떤 접촉 자세/속도 설계가 가능성이 있었는가”를 보여줄 수 있다.
- 환경 구조도, 보상 설계, 실패 사례 분석 앞부분에 heuristic feasibility 결과를 baseline으로 둘 수 있다.
