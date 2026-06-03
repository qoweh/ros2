너는 senior product designer + frontend engineer + reinforcement learning technical writer 역할을 맡는다.

목표:
로봇팔 끝에 탁구채를 붙이고 MuJoCo 환경에서 탁구공을 계속 위로 띄우도록 강화학습한 프로젝트를 웹사이트로 소개하고 체험할 수 있게 만드는 제품/문서 기획안을 작성하라. 이 웹사이트는 단순 녹화 영상이 아니라, 브라우저에서 실제 MuJoCo WASM 시뮬레이션과 학습된 policy inference가 실행되는 형태를 목표로 한다.

참고 레퍼런스:
- RoboPianist 프로젝트 페이지: interactive demo, Overview, Simulation, MDP Formulation, Quantitative Results, Qualitative Results 구조를 참고한다.
- Zalo mujoco_wasm demo: 브라우저에서 MuJoCo 모델을 로드/렌더링하고 사용자가 상호작용하는 데모 스타일을 참고한다.
- Google DeepMind MuJoCo WASM README: @mujoco/mujoco, WebAssembly, JavaScript/TypeScript bindings, Three.js example, single-thread/multi-thread 주의사항을 참고한다.
- MuJoCo Playground: demo/paper/code/live demo/Colab 링크를 상단에 배치하고 결과를 카테고리별로 보여주는 프로젝트 페이지 구조를 참고한다.

핵심 사용자:
1. 강화학습을 잘 모르는 일반 개발자
2. 소프트웨어 전공자로 기본적인 수학/ML/RL 용어는 어느 정도 이해 가능한 사람
3. 포트폴리오 또는 연구 프로젝트를 평가하는 사람
4. 웹에서 직접 시뮬레이션을 만져보고 싶은 사용자

사이트 목표:
- 사용자가 “이건 녹화 영상이 아니라 실제 policy가 현재 상태를 보고 제어하는 것”이라고 느끼게 한다.
- 강화학습의 observation, action, reward, policy, episode, randomization 개념을 시각적으로 이해하게 한다.
- 프로젝트의 기술적 완성도와 구현 과정을 설득력 있게 보여준다.
- GitHub로 이동해 코드를 확인할 수 있게 한다.

필수 페이지:
1. Main / Live Demo page
2. Docs / Method page
3. Experiments or Playground section
4. Results section
5. About / Tech Stack / GitHub links

Main page 요구사항:
- Hero에 프로젝트 제목, 짧은 설명, Try Demo, Read Docs, GitHub 버튼을 둔다.
- 중앙에 MuJoCo WASM 기반 live simulation canvas를 둔다.
- 매 reset마다 탁구공의 초기 위치, 초기 속도, spin이 랜덤하게 달라져야 한다.
- episode number와 seed를 화면에 표시한다.
- “This is not a recorded video. The policy runs live at each control step.”라는 메시지를 명확히 보여준다.
- Play, Pause, Step, Reset Random Episode, Replay Same Seed 기능을 설계한다.
- 카메라 모드는 Free View, North, South, East, West, 4-Camera View를 제공한다.
- 사용자가 공 위치를 직접 드래그하거나 슬라이더로 조절할 수 있게 한다.
- 사용자가 공에 외력을 주는 버튼을 제공한다. 예: Push Ball, Wind Gust.
- Ball trajectory trail, contact marker, target height band를 시각화한다.
- policy 선택 기능을 둔다: Random, Early Checkpoint, Mid Checkpoint, Final Policy.
- 간단한 reward breakdown 패널을 제공한다.

Playground 요구사항:
- 사용자가 변수들을 바꿔보면서 policy robustness를 체험하게 한다.
- 조절 후보: gravity, ball mass, restitution, initial speed, disturbance strength, observation noise, action delay, torque limit.
- 너무 많은 옵션은 Advanced 패널로 숨긴다.
- 각 변수 옆에는 “이 값이 커지면 무엇이 어려워지는지”를 한 줄로 설명한다.
- preset을 제공한다: Easy, Normal, Hard, Chaos, Low Gravity, Noisy Sensor.
- preset/seed/config를 URL query로 공유할 수 있는 구조를 제안한다.

Docs page 요구사항:
다음 섹션을 포함해서 RL을 모르는 전공자도 이해할 수 있게 작성하라.

1. Overview
   - 어떤 문제를 풀었는지
   - 왜 scripted controller가 아니라 RL을 썼는지
   - 최종 데모가 무엇을 보여주는지

2. Simulation Environment
   - robot arm DOF
   - paddle geometry
   - ping-pong ball properties
   - MuJoCo model structure
   - control frequency
   - episode length
   - termination conditions

3. MDP Formulation
   - Observation vector를 표로 정리한다.
   - 각 observation component의 dimension과 의미를 설명한다.
   - Action space를 설명한다.
   - Reward function을 항별로 설명한다.
   - Episode initial state distribution과 randomization을 설명한다.

4. Policy and Training
   - 사용 알고리즘 예: PPO 또는 SAC
   - policy network 구조
   - value network 구조가 있다면 설명
   - total timesteps
   - parallel environments
   - hyperparameters
   - checkpoint 저장 방식
   - evaluation protocol

5. Results
   - training reward curve
   - keep-up duration
   - contacts per episode
   - success rate
   - failure cases
   - policy checkpoint comparison

6. Ablation Study
   - reward term 제거 실험
   - initial randomization 제거 실험
   - observation noise 실험
   - torque/action smoothing 실험
   - early/mid/final checkpoint 비교

7. Web Deployment
   - Python training pipeline
   - policy export format
   - TypeScript inference runner
   - MuJoCo WASM simulation
   - Three.js rendering
   - Spring backend를 쓴다면 API/로그/모델 파일 관리 역할 설명
   - single-thread WASM MVP와 multi-thread WASM 주의사항 설명

8. Limitations
   - simulation-only policy
   - sim-to-real gap
   - extreme initial conditions에서 실패 가능
   - browser performance limits
   - camera-based perception 미포함 여부

시각화 요구사항:
- RL loop diagram
- observation vector block diagram
- action-to-joint mapping diagram
- reward breakdown chart
- training curve
- checkpoint comparison strip
- ball trajectory visualization
- 4-camera layout mockup
- failure case gallery

산출물:
1. 전체 정보구조 IA
2. 페이지별 wireframe 설명
3. 주요 UI 컴포넌트 목록
4. MVP / v1 / v2 기능 우선순위
5. Docs 페이지 목차와 각 섹션 초안
6. 데모 상호작용 상세 스펙
7. 기술 스택 제안
8. GitHub README에 넣을 요약 섹션
9. 구현 task breakdown
10. acceptance criteria

기술 스택 가정:
- Frontend: React 또는 Next.js, TypeScript, Three.js, @mujoco/mujoco
- Simulation: MuJoCo WASM
- Policy inference: TypeScript MLP runner, ONNX Runtime Web, 또는 JSON weight loader 중 하나 제안
- Backend: Java/Spring은 유지 가능. 사용자 설정 저장, 학습 job 관리, 모델 파일 제공, 로그 API, WebSocket 진행률 전달에 사용
- Training: Python + MuJoCo + PPO/SAC 기반. 웹 학습은 MVP 범위에서 제외하고, preset/checkpoint 기반 체험을 우선한다.

중요한 설계 원칙:
- 메인 페이지는 복잡한 논문 페이지처럼 보이면 안 된다. 먼저 “와, 실제로 움직인다”를 보여준다.
- Docs는 너무 쉬운 Python 문법 설명이 아니라, RL 프로젝트의 핵심 설계인 observation/action/reward/policy/training/evaluation을 설명한다.
- 녹화 영상처럼 보이지 않게 seed, random reset, user perturbation, real-time reward/action overlay를 반드시 넣는다.
- GitHub 링크는 상단 nav와 hero 버튼, footer에 배치한다.
- 사용자가 조작 가능한 변수는 처음에는 5개 이하로 제한하고, 나머지는 Advanced 패널로 숨긴다.
- 모든 UI 문구는 영어 기준으로 작성하되, 한국어 버전이 필요하면 쉽게 번역 가능하게 짧고 명확하게 쓴다.

최종 답변은 한국어로 작성하라.
