# pingpong_rl 현재 상태 보고서

## 1. 프로젝트 목표의 현재 정의

현재 `pingpong_rl`의 직접 목표는 `실제 탁구 경기`가 아니라, `탁구공을 떨어뜨리지 않고 라켓으로 반복적으로 위로 튕겨 올리는 keep-up`에 가깝다.

이 정의가 중요한 이유는, 현재 scene에는 table/net 기반 rally task보다 `라켓-공 접촉`, `EE 제어`, `반복 bounce 안정화`가 핵심이기 때문이다.

## 2. 최근 핵심 수정

### 2.1 기본 공 높이 상향

- 학습/렌더링 기본 공 높이를 `0.5`로 통일

### 2.2 접촉 물리 수정

- 로봇 body collision mesh를 다시 활성화
- 공이 arm body에 닿으면 `robot_body_contact`로 실패 기록

이제 공이 본체를 그냥 통과하는 비물리적 장면 대신, 실제 접촉 실패로 처리된다.

### 2.3 active-hit 보상 수정

- 접촉 순간 acceleration만 보고 active hit를 주지 않도록 수정
- 실제 upward racket velocity가 있어야 useful hit로 인정

이 변경은 `접촉 순간 떨기`를 줄이는 데 목적이 있다.

### 2.4 controller 응답 속도 상향

- EE controller 기본 gain / position step을 높여 실제 strike speed가 나오도록 조정

이전 probe에서는 최대 upward 명령에도 라켓 z속도가 약 `0.093 m/s` 수준이었지만, 수정 후에는 약 `0.467 m/s`까지 올라간다.

### 2.5 curriculum 도입

`run_ppo_baseline.py` 기본 학습은 `keepup_v1` curriculum을 사용한다.

단계는 아래와 같다.

- bootstrap: 기본 upward hit 패턴 형성
- stabilize: 반복 성공률 안정화
- refine: 자세/관절/부드러움까지 반영

## 3. 현재 해석

최근까지 관측된 이상 행동은 크게 세 가지였다.

- 위로 확실히 치지 않고 접촉 근처에서 잔진동
- 로봇팔 응답이 너무 느려 strike가 약함
- collision 설정 실수로 공이 arm body를 통과

이 중 `공 질량 과다`는 주원인으로 확인되지 않았다.

## 4. 다음 학습 권장 방식

기존 checkpoint를 그대로 이어서 학습하기보다, 수정된 동역학과 보상 기준으로 새 run name으로 다시 시작하는 것이 더 적절하다.

권장 방식:

1. `150k~200k` 단위로 짧게 학습
2. 매 구간마다 render로 upward strike 유무 확인
3. strike는 생겼지만 거칠면 smoothness/tilt penalty 미세조정
4. strike 자체가 약하면 reward보다 controller 응답과 contact velocity 분포를 먼저 재확인

## 5. 보고서용 핵심 문장

이번 단계의 핵심 메시지는 아래처럼 요약할 수 있다.

`문제는 단순히 공이 무겁거나 팔 힘이 약한 것이 아니라, active-hit 판정의 오설계, EE 응답 속도 부족, 그리고 collision mesh 비활성화로 인한 비물리적 pass-through가 복합적으로 겹친 것이었다. 따라서 최근 수정은 더 오래 학습시키는 방향보다, 물리/제어/보상 구조를 먼저 바로잡는 방향으로 진행됐다.`
