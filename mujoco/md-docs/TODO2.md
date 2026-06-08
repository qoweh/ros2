# 과거 car_rl 질문 메모

이 파일은 초기 `car_rl` 토이 프로젝트를 보며 남긴 질문을 보관한 것이다. 현재 구조에는 `car_rl/` 코드 디렉터리가 남아 있지 않으므로, 실행 파일 링크 대신 당시 질문의 의도와 남아 있는 비교 문서를 기준으로 본다.

## 남아 있는 질문

1. `test.py --headless`는 viewer 없이 평가하고, `--headless` 없이 실행하면 평가 rollout을 viewer로 볼 수 있는 구조였는가?
2. 학습 중 evaluation과 별도 test script evaluation은 어떤 차이가 있는가?
3. PPO에서 `epoch`, `n_steps`, `timesteps`, rollout, evaluation callback은 각각 어떤 역할인가?
4. "이 환경은 `mj_step()` 중심이 아니다"라는 설명은 MuJoCo가 물리를 계산하지 않는다는 뜻이 아니라, Gymnasium env가 action/reward/reset 계약의 중심이라는 뜻인가?
5. `car_rl` 토이 프로젝트와 `pingpong_rl` 프로젝트는 구조적으로 무엇이 달랐는가?

## 현재 참고할 문서

- `pingpong_rl/docs/report/08_car_rl_vs_pingpong_rl_structure_report.md`
- `pingpong_rl/README.md`
- `pingpong_rl2/README.md`

새로 PPO 개념을 정리할 때는 남아 있지 않은 `car_rl` 파일 경로를 기준으로 하지 말고, 현재 실행 가능한 `pingpong_rl2`의 `scripts/run_ppo_learning.py`, `src/pingpong_rl2/envs/gym_env.py`, `src/pingpong_rl2/training/vector_env.py`를 기준으로 설명한다.
