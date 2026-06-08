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

## 발표 때 설명 포인트

- PPO 학습 전 “어떤 접촉 자세/속도 설계가 가능성이 있었는가”를 보여줄 수 있다.
- 환경 구조도, 보상 설계, 실패 사례 분석 앞부분에 heuristic feasibility 결과를 baseline으로 둘 수 있다.
