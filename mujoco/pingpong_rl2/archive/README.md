# Archive

이 디렉터리는 `pingpong_rl2`에서 현재 기본 경로로 쓰지 않는 실험 조각을 보관하는 곳이다. 핵심 학습/평가 흐름은 상위 `README.md`, `scripts/`, `configs/`, `src/`를 기준으로 본다.

현재 baseline에서는 아래 성격의 코드를 기본 경로에 섞지 않는다.

- env 내부 heuristic tracking assist
- hand-coded keep-up policy의 직접 주입
- 초기 tilt action baseline
- 과거 curriculum mutation 방식
- 해석이 어려운 reward term 누적본

필요하면 archived 아이디어를 하나씩 되살리되, 먼저 `tests/`와 짧은 smoke run으로 현재 v39/v40 계열 contract를 깨지 않는지 확인한다.
