# 로봇팔 강화학습 시뮬레이션 개발 방향 제안

## 현재 환경 요약

| 항목 | 상태 |
|------|------|
| HW | Apple M1 Pro, 32GB RAM |
| OS | macOS (arm64) |
| Python | 3.12 (conda ros_env) |
| 설치된 시뮬레이터 | Gazebo Harmonic (gz-sim8) |
| mujoco / gymnasium | **미설치** (pip으로 즉시 가능) |
| NVIDIA GPU | 없음 → Isaac Sim 불가 |

---

## 1. 시뮬레이터 선택

### 후보 비교

| 시뮬레이터 | Mac M1 지원 | RL 적합성 | 설치 난이도 | 특징 |
|-----------|-------------|----------|------------|------|
| **MuJoCo** | ✅ 완벽 | ⭐⭐⭐⭐⭐ | pip 1줄 | RL 연구 표준, 빠른 physics, Apple Silicon 공식 지원 |
| **Gazebo Harmonic** | ✅ 설치됨 | ⭐⭐ | 이미 있음 | ROS 2 통합 강점, RL에는 느림 |
| **PyBullet** | ✅ | ⭐⭐⭐ | pip 1줄 | 오래됨, MuJoCo보다 느림, 점점 사용↓ |
| **dm_control** | ✅ | ⭐⭐⭐⭐ | pip 1줄 | MuJoCo 래퍼, DeepMind tasks 포함 |
| **Genesis** | ✅ (Metal) | ⭐⭐⭐⭐ | pip + 빌드 | 2024년 신규, GPU 병렬화, 아직 불안정 |
| **Isaac Sim / Isaac Lab** | ❌ | ⭐⭐⭐⭐⭐ | 불가 | NVIDIA GPU 전용 |
| **ManiSkill (SAPIEN)** | ✅ | ⭐⭐⭐⭐ | pip 1줄 | GPU 병렬 환경 (CPU fallback 가능), 태스크 다양 |

### 권장: **MuJoCo**

- RL 논문의 99%가 사용하는 사실상 표준
- SO-101 공식 MJCF 파일(`so101_new_calib.xml`) 이미 존재
- `pip install mujoco` 한 줄로 설치, M1 완벽 지원
- `gymnasium-robotics` 같은 기성 환경과 바로 연동 가능
- Gazebo 대비 시뮬레이션 속도 5~50배 빠름 (RL 학습에 결정적)

> Gazebo는 실제 로봇 통합(ROS 2 제어) 목적에는 좋지만,  
> RL 수천만 스텝 학습에는 속도가 너무 느려서 적합하지 않음.

---

## 2. 로봇팔 선택

### 후보 비교

| 로봇팔 | MJCF/URDF 품질 | Gym 환경 지원 | 학습 레퍼런스 | 실물 가능성 | 특징 |
|--------|---------------|-------------|------------|------------|------|
| **Franka Panda** | ⭐⭐⭐⭐⭐ | ✅ (gymnasium-robotics 기본) | 매우 풍부 | 고가 | RL 연구 표준 로봇팔 |
| **SO-101** | ⭐⭐⭐ | ❌ (직접 구성 필요) | 거의 없음 | ✅ 저비용 실물 있음 | LeRobot 통합, 공식 MJCF 존재 |
| **UR5 / UR10** | ⭐⭐⭐⭐ | ✅ (robosuite 등) | 풍부 | 실물 고가 | 산업 표준 |
| **xArm6/7** | ⭐⭐⭐ | 제한적 | 중간 | 비교적 저가 | |
| **MyoArm / Aloha** | ⭐⭐⭐ | 특수 환경 | 특정 분야 | - | |

### 판단 기준별 추천

**빠른 RL 학습 시작 목적 → Franka Panda**
- `gymnasium-robotics`에 FetchReach, FetchPickAndPlace 등 기성 환경 포함
- 수천 개의 논문/코드 레퍼런스
- `mujoco_menagerie`에 고품질 MJCF 포함

**실물 연동 + 저비용 목적 → SO-101**
- 실물을 갖고 있거나 살 예정이면 SO-101이 유리
- LeRobot imitation learning과 직결
- 공식 MJCF로 MuJoCo에서 바로 사용 가능
- RL 기성 환경은 없으므로 환경 직접 작성 필요

> **결론:** 순수 RL 실험이 목적이면 Panda, 실물 연동/imitation learning이면 SO-101

---

## 3. 학습 작업 및 알고리즘

### 탁구공 튕기기 평가

**아이디어:** 로봇팔 끝에 탁구채를 달고 공이 안 떨어지게 위로 계속 튕기기

| 항목 | 평가 |
|------|------|
| 흥미도 | ⭐⭐⭐⭐⭐ |
| 학습 난이도 | 🔴 매우 어려움 |
| 시뮬 충실도 의존성 | 매우 높음 (공 바운스 물리) |
| Sim-to-real gap | 매우 큼 |
| 졸업 기간 내 수렴 가능성 | 낮음 |

**왜 어려운가:**
- Contact-rich task: 채와 공의 충돌 타이밍이 ms 단위로 중요
- 공의 3D 위치 추적 + 예측이 필요
- 하나의 "튕기기"를 성공해도 연속 N회는 exponential하게 어려워짐
- RL reward 설계 자체가 비자명 (희소 보상)
- 졸업 프로젝트 기간(수 개월)에 수렴 가능성이 낮음

**만약 시도한다면:**
- 알고리즘: Curriculum Learning + SAC (공 높이 목표를 단계적으로 높임)
- 또는 Model-based RL (Dreamer v3) — 샘플 효율성 중요
- MuJoCo 필수 (공 physics)

---

### 추천 학습 작업 목록

#### 🟢 난이도 낮음 (빠른 성과)

| 작업 | 설명 | 추천 알고리즘 | 비고 |
|------|------|--------------|------|
| **Reaching** | 로봇 end-effector를 목표 좌표에 가져가기 | PPO, SAC | 가장 기본 태스크. gymnasium-robotics에 FetchReach 있음 |
| **Push to target** | 테이블 위 물체를 목표 위치로 밀기 | SAC | 간단한 contact task |

#### 🟡 난이도 중간 (졸업 프로젝트 적합)

| 작업 | 설명 | 추천 알고리즘 | 비고 |
|------|------|--------------|------|
| **Pick & Place** | 물체를 집어서 목표 위치에 놓기 | SAC + HER | RL 연구 표준 태스크. 레퍼런스 매우 많음 |
| **Block Stacking** | 블록을 N단으로 쌓기 | SAC + HER + Curriculum | Pick & Place 확장 |
| **Drawer Opening** | 서랍 손잡이 잡고 열기 | SAC, TD3 | contact task |
| **Rope Manipulation** | 줄을 목표 형태로 배치 | SAC + 특수 보상 | MuJoCo 물리 필요 |

#### 🔴 난이도 높음 (도전적)

| 작업 | 설명 | 추천 알고리즘 | 비고 |
|------|------|--------------|------|
| **Peg-in-hole** | 좁은 구멍에 막대 삽입 | SAC + force sensing | 정밀 제어 필요 |
| **탁구공 튕기기** | 위에서 평가함 | Curriculum + SAC / MBRL | 매우 어려움 |
| **Dexterous grasping** | 다양한 형태의 물체 파지 | PPO + domain rand | 복잡한 손 필요 |

---

### 알고리즘 간략 설명

| 알고리즘 | 분류 | 특징 | 적합 태스크 |
|---------|------|------|------------|
| **SAC** (Soft Actor-Critic) | Off-policy RL | 연속 행동 공간 표준, 샘플 효율 좋음 | Pick&Place, Reaching |
| **TD3** | Off-policy RL | SAC와 유사, 분산 낮음 | 유사 태스크 |
| **PPO** | On-policy RL | 안정적, 구현 쉬움 | 단순 태스크 |
| **HER** (Hindsight Experience Replay) | 보조 기법 | Sparse reward 문제 해결 | goal-conditioned tasks |
| **Diffusion Policy** | Imitation Learning | 시연 데이터로 학습 | Pick&Place, 조작 |
| **ACT** | Imitation Learning | SO-101 + LeRobot 공식 지원 | 데이터 수집 후 모방 |
| **Dreamer v3** | Model-based RL | 샘플 효율 최고, 복잡한 task | 탁구공 등 어려운 태스크 |

---

## 4. 최종 방향 조합 추천

### 조합 A: RL 표준 접근 (빠른 결과)
```
시뮬레이터 : MuJoCo
로봇팔     : Franka Panda (mujoco_menagerie MJCF)
환경       : gymnasium-robotics (FetchPickAndPlace)
태스크     : Pick & Place
알고리즘   : SAC + HER (stable-baselines3)
설치       : pip install mujoco gymnasium gymnasium-robotics stable-baselines3
```
- 장점: 레퍼런스 넘침, 기성 환경 사용으로 개발 빠름, 결과 비교 쉬움
- 단점: 실물 SO-101 연동 불가

### 조합 B: SO-101 중심 imitation learning
```
시뮬레이터 : MuJoCo (공식 so101_new_calib.xml)
로봇팔     : SO-101
태스크     : Pick & Place 또는 Push
학습       : ACT / Diffusion Policy (LeRobot)
           → 실물로 시연 데이터 수집 → MuJoCo에서 sim 학습 → 실물 배포
```
- 장점: 실물 연동 직결, LeRobot 생태계 활용
- 단점: 시연 데이터 수집 필요, RL보다 데이터 의존적

### 조합 C: SO-101 + MuJoCo + SAC (RL)
```
시뮬레이터 : MuJoCo (so101_new_calib.xml + 커스텀 gymnasium env)
로봇팔     : SO-101
태스크     : Reaching → Pick & Place (단계적)
알고리즘   : SAC + HER
```
- 장점: SO-101 실물 연동 가능 + RL 방식
- 단점: Gymnasium 환경 직접 작성 필요 (1~2일 작업)

---

## 결정을 위한 질문

1. **실물 SO-101을 보유하거나 구매 예정인가?**
   - 있으면 → 조합 B 또는 C
   - 없으면 → 조합 A (Panda + gymnasium-robotics가 가장 빠름)

2. **졸업 프로젝트의 평가 기준은 무엇인가?**
   - "특정 태스크를 RL로 학습시켜 성공"이면 → SAC + Pick&Place
   - "실물 로봇을 움직이는 데모"이면 → Imitation Learning + SO-101

3. **탁구공 튕기기에 얼마나 집착하는가?**
   - 단순 데모 수준이면 scripted policy로 흉내 낼 수 있음
   - 진짜 RL로 학습하려면 최소 수개월 이상 소요 가능

4. **Gazebo를 꼭 써야 하는가?**
   - ROS 2 통합이 필요하면 Gazebo, 순수 RL 학습이면 MuJoCo 권장
