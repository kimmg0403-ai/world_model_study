# World Model과 World Action Model — 개념·논문·연구 흐름 학습 노트

> **마지막 조사·정리: 2026-10-13 (한국 시간)**
>
> **목표:** “세계를 예측하는 모델이 어떻게 행동하는 에이전트가 되는가?”를 기초부터 설명하고, 핵심 논문을 연결해서 읽는다.
>
> **범위:** 모델 기반 강화학습, 영상·잠재표현 기반 세계 모델, 로봇 World Action Model(WAM). 전체 문헌을 망라하는 목록이 아니라 핵심 계보와 최근 쟁점을 선별한 학습 지도다. 최신 항목은 2026-10-08 공개 논문까지 포함한다.
>
> **읽는 기준:** 논문에 근거한 설명에는 원문 링크를 붙였다. ‘해석’, ‘한계’, ‘읽을 질문’은 이 노트의 학습용 정리이며, 별도 실험으로 입증한 주장이 아니다. 수식은 공통 구조를 이해하기 위한 단순화이며 각 논문의 정확한 구현과 구분한다. 최근 프리프린트의 결과는 저자가 보고한 해당 실험 범위에서 읽는다.

## 이 노트의 읽기 규칙: 공통 예제를 끝까지 따라가기

모든 개념을 **“탁자 위의 컵을 로봇 팔이 집어 오른쪽 선반으로 옮기기”**에 적용해 본다. 중간에 공이 움직이는 보조 예시도 사용한다. **아래 가상 로봇 실험은 논문에서 실제로 수행한 실험이 아니다.** World Models(2018)의 실제 시연은 CarRacing·VizDoom이다. 숫자·행동 예제는 표기를 이해하기 위한 가정이다.

| 시각 | 실제 환경·관측 | 실행하는 행동 | 다음에 확인할 관측 |
|---|---|---|---|
| $t=0$ | $o_0$: 컵이 탁자에 있고 그리퍼는 멀리 있음 | $a_0$: 컵 위로 이동 | $o_1$: 컵 위에 도달했는가? |
| $t=1$ | $o_1$: 그리퍼가 컵 위에 있음 | $a_1$: 내려가서 닫기 | $o_2$: 컵이 잡혔나, 미끄러졌나? |
| $t=2$ | $o_2$: 컵이 잡혔거나 놓쳤음 | $a_2$: 컵 들어 올리기 | $o_3$: 컵이 함께 올라오는가? |
| $t=3$ | $o_3$: 컵을 들고 있음 | $a_3$: 선반으로 이동해 놓기 | $o_4$: 컵이 목표 위치에 놓였나? |

한 전이의 **기록 데이터**는 $(o_t,a_t,o_{t+1},r_t)$이다. 즉 **$a_t$는 $o_t$를 본 뒤 실행하며 그 결과를 $o_{t+1}$에서 관찰한다.** 보상 $r_t$는 해당 전이에 대응한다. 예를 들어 최종 성공 시 $+1$, 실패 시 $0$으로 가정할 수 있다.

**이 문서에서 고정할 기호:** $s_t$는 실제 환경 상태(물체 위치·속도·마찰·접촉 등), $o_t$는 실제 카메라·센서 관측, $x_t$는 모델 내부에서 **학습해 추론한 잠재 상태**다. 실제 데이터에 $x_t$의 정답 레이블이 주어지는 것은 아니다. $g$는 목표(“컵을 선반에 놓아라”)다. 5.2절 VAE–MDN-RNN의 $z_t,h_t$와 5.3절 RSSM의 $x_t=(u_t,z_t)$는 **각기 다른 모델의 변수**이므로 반드시 구분한다.

수식을 볼 때마다 **입력 → 예측할 target → 학습 loss → 실제 행동에 쓰이는 방식**을 확인한다. 3절은 기호, 4절은 학습 메커니즘, 5절은 초기 논문들의 서로 다른 행동 선택법, 6–7절은 영상·행동 생성으로의 확장을 설명한다.

<a id="contents"></a>
## 목차

1. [먼저 보는 핵심 지도와 연표](#map)
2. [기초 개념: 무엇을 모델링하는가?](#basics)
3. [수학적 언어: 상태·관측·행동·목표](#math)
4. [어떻게 학습하고 행동하는가?](#mechanisms)
5. [필독 논문 I: 모델 기반 RL과 상상 학습](#papers-rl)
6. [필독 논문 II: 영상·표현 기반 세계 모델](#papers-foundation)
7. [필독 논문 III: VLA에서 WAM으로](#papers-wam)
8. [2026년 최신 쟁점과 추가 읽기](#recent)
9. [대표 방법 비교와 논문 평가법](#evaluation)
10. [권장 공부 순서와 작은 실험](#study)
11. [자주 헷갈리는 질문과 이해도 점검](#faq)
12. [용어 사전](#glossary)
13. [후속 질문·수정 기록](#notebook)
14. [원문과 공식 자료 모음](#references)

---

<a id="map"></a>
## 1. 먼저 보는 핵심 지도와 연표

### 1.1 다섯 문장으로 잡는 큰 그림

1. **World Model(WM)**은 경험으로부터 환경의 구조와 변화 규칙을 배워, 현재 정보와 가능한 행동을 바탕으로 미래를 예측하는 모델이다. 다만 영상 생성 연구에서는 행동 입력 없이 미래 영상을 만드는 모델까지 넓게 부르므로, 논문마다 정의를 확인해야 한다.
2. **Policy(정책)**는 “무엇을 할지” 결정한다. WM은 “그렇게 하면 무엇이 일어날지” 예측한다. 두 기능은 분리할 수도, 하나의 네트워크 안에 결합할 수도 있다.
3. **모델 기반 RL**은 학습한 세계 모델로 계획하거나 가상 경험을 만들어 정책을 개선한다. PlaNet, Dreamer, MuZero가 대표적인 서로 다른 설계다.
4. **Foundation World Model**은 대규모 영상 등으로 사전학습한 세계 예측 기반 모델을 여러 환경과 작업에 적응시키려는 방향이다. 사전학습 데이터가 커도, 정밀한 행동 조건화와 물리적 정확성이 자동으로 보장되지는 않는다.
5. **World Action Model(WAM)**은 최근 로봇 연구에서 미래 관측·세계 표현과 행동을 함께 모델링하는 접근을 가리킨다. DreamZero는 중요한 사례지만, 영상에서 행동을 얻거나 영상과 행동을 공동 학습하는 아이디어는 그 이전부터 발전했다. [정의 논의](https://arxiv.org/abs/2607.06401), [DreamZero](https://arxiv.org/abs/2602.15922), [UniPi](https://arxiv.org/abs/2302.00111)

### 1.2 발전은 하나의 직선이 아니라 합류하는 여러 계보다

~~~text
학습한 환경 모델로 가상 경험 활용
  Dyna → World Models → PlaNet → Dreamer 1/2/3 → Dreamer 4
                         └ 온라인 계획           └ 큰 영상 모델 + 상상 속 RL

의사결정에 필요한 표현을 학습
  MuZero ────────────────────┐
  TD-MPC / TD-MPC2 ───────────┼→ “모든 픽셀을 복원할 필요가 있는가?”
  JEPA → V-JEPA → V-JEPA 2/2.1┘

영상 생성과 상호작용 환경
  영상 예측 → IRIS / DIAMOND / Diffusion Forcing
           → Genie 계열 / Cosmos 계열

언어·시각에서 로봇 행동으로
  RT-2 / π0 / π0.5                   : 관측·언어 → 행동
  UniPi                              : 미래 영상 → 역동역학 → 행동
  UVA / WorldVLA                     : 영상·행동 공동 모델링
  Cosmos Policy / DreamZero          : 사전학습 영상 모델을 제어로 연결
  2026 후속 연구                     : 행동에 맞는 표현·속도·계획 범위·실행 상태
~~~

위 화살표는 **학습용 개념 연결**이다. 모든 방법이 바로 앞 방법의 직접 후속작이거나 동일한 아키텍처를 쓴다는 의미는 아니다.

### 1.3 핵심 연표

연도는 원칙적으로 최초 공개 기준이다. 예컨대 PlaNet은 2018년 프리프린트 / ICML 2019, Dreamer는 2019년 프리프린트 / ICLR 2020이다.

| 최초 공개 | 논문·시스템 | 바뀐 질문 | 읽기 우선순위 |
|---|---|---|---|
| 1990–1991 | Dyna | 실제 경험과 모델이 만든 경험을 함께 쓰면? | 역사적 배경 |
| 2018 | World Models | 영상 압축·시간 예측·제어를 분리하면? | 필독 |
| 2018 | PlaNet | 픽셀에서 배운 잠재 동역학으로 온라인 계획하면? | 필독 |
| 2019 | Dreamer | 계획을 매번 탐색하는 대신 상상 속에서 정책을 학습하면? | 필독 |
| 2019 | MuZero | 화면 복원 대신 보상·정책·가치만 예측하면? | 필독 |
| 2020 | DreamerV2 | 이산 잠재변수로 상상 학습을 개선하면? | 연결 읽기 |
| 2022 | LeCun의 자율지능 구상 | 예측 불가능한 픽셀 대신 추상 표현을 예측하면? | 관점 읽기 |
| 2022 | IRIS | 이미지 토큰의 Transformer가 세계 모델이 되면? | 연결 읽기 |
| 2023 | DreamerV3 | 여러 도메인에서 하나의 설정으로 학습하면? | 필독 |
| 2023 | UniPi | 언어 목표에 맞는 미래 영상에서 행동을 얻으면? | WAM 계보 필독 |
| 2023 | Diffusion Policy / RT-2 | 확산 행동 정책 / 웹 지식 기반 VLA | 비교용 필독 |
| 2023 | TD-MPC2 | 제어에 맞춘 decoder-free 잠재 모델을 확장하면? | 필독 |
| 2024 | V-JEPA / Genie / DIAMOND | 표현 예측 / 잠재 행동 / 확산 세계 모델 | 분기별 필독 |
| 2024 | Diffusion Forcing / π0 | 시퀀스 확산의 조건화 / 연속 행동 생성 | 연결 읽기 |
| 2025-01 | Cosmos | 세계 모델 사전학습을 공통 플랫폼으로 만들면? | 연결 읽기 |
| 2025-02 | UVA | 영상과 행동을 함께 배우면서 추론은 빠르게 하면? | WAM 필독 |
| 2025-04 | π0.5 | 이질적인 데이터를 활용해 실제 환경 일반화를 높이면? | 비교용 |
| 2025-06 | V-JEPA 2 / WorldVLA | 표현 공간 계획 / 자기회귀 영상·행동 통합 | 필독 |
| 2025-08 | Genie 3 | 실시간 상호작용 세계를 생성하면? | 공식 발표 참고 |
| 2025-09 | Dreamer 4 | 큰 세계 모델 안에서 오프라인 데이터로 RL하면? | 필독 |
| 2026-01 | Cosmos Policy | 영상 모델에 행동·미래 관측·가치를 넣으면? | WAM 필독 |
| 2026-02 | DreamZero | 영상과 행동의 공동 생성으로 새 동작에 일반화하면? | WAM 필독 |
| 2026-03 | V-JEPA 2.1 | 공간·시간에 조밀한 표현이 제어에 도움이 되면? | 최신 연결 |
| 2026-06 | Cosmos 3 / AGRA | 모달리티 통합 / 예측 표현을 행동에 정렬 | 최신 쟁점 |
| 2026-09 | WAM 설계 연구 / ReWAM / Planning Limits | 무엇이 성능을 만들고, 계획은 어디까지 가능한가? | 비판적 읽기 |
| 2026-10 | HWAM | 명령한 행동과 실제 몸의 움직임 사이를 모델링하면? | 최신 프리프린트 |

원문은 [14절의 자료 목록](#references)에서 바로 열 수 있다.

---

<a id="basics"></a>
## 2. 기초 개념: 무엇을 모델링하는가?

### 2.1 컵을 집는 로봇으로 생각하기

로봇에게 “컵을 집어서 오른쪽 선반에 놓아라”라고 지시했다고 하자.

- **관측:** 카메라 영상, 관절 각도, 그리퍼 상태. 컵의 정확한 질량이나 마찰계수는 바로 보이지 않는다.
- **상태:** 컵과 로봇의 위치·속도·접촉 상태 등 환경의 실제 조건. 관측보다 넓다.
- **행동:** 관절 목표, 말단장치의 위치 변화, 그리퍼 개폐 등 로봇에 보내는 명령.
- **세계 모델:** “이쪽으로 이동해서 그리퍼를 닫으면 컵이 잡힐까, 밀려날까?”를 예측한다.
- **정책:** 실제로 보낼 행동을 결정한다.
- **목표·보상·가치:** 무엇을 성공으로 볼지, 지금 행동이 장기 성공에 얼마나 유리할지 나타낸다.

이 예에서 중요한 것은 멋진 집기 영상을 만드는 능력뿐 아니라 **행동을 조금 바꿨을 때 결과가 어떻게 달라지는지 구별하는 능력**이다.

### 2.2 Model-free와 model-based

| 구분 | 핵심 학습 대상 | 행동을 얻는 방식 | 주의할 점 |
|---|---|---|---|
| Model-free RL | 정책 또는 가치 | 실제 경험 등으로 직접 개선한 정책·가치 사용 | 내부에 기억이나 환경 지식이 없다는 뜻은 아님 |
| Model-based RL | 환경의 전이·보상 등에 대한 모델 | 모델을 이용한 계획 또는 가상 경험 학습 | 모델 오차를 이용하는 잘못된 행동이 나올 수 있음 |
| Imitation learning | 시연 행동 분포 | 관측·목표에서 시연과 유사한 행동 생성 | 보상 최대화 RL과 동일하지 않음 |
| WAM 기반 정책 학습 | 미래 관측·표현과 행동의 관계 | 공동 생성, 역동역학, 또는 추가 계획 | 반드시 RL로 학습하거나 탐색하는 것은 아님 |

**“모델이 크다”, “Transformer다”, “추론 중 생각한다”만으로 model-based라고 판정하지 않는다.** 학습한 환경 예측을 의사결정에 어떻게 사용하는지 봐야 한다.

### 2.3 세 가지 예측을 구별하기

| 이름 | 질문 | 단순화한 형태 |
|---|---|---|
| Forward dynamics, 정동역학 | 지금 이 행동을 하면 어떻게 되는가? | $p(o_{t+1}\mid h_t,a_t)$ |
| Inverse dynamics, 역동역학 | 지금 모습에서 저 모습으로 가려면 어떤 행동이 필요한가? | $p(a_t\mid o_t,o_{t+1})$ |
| Policy, 정책 | 목표를 위해 지금 무엇을 할 것인가? | $\pi(a_t\mid h_t,g)$ |

여기서 $h_t$는 지금까지의 관측·행동 이력이고, $g$는 목표다. 표 안의 조건부 확률에는 $\mid$를 사용하여 Markdown 표의 구분자와 충돌하지 않도록 했다.

역동역학은 정동역학의 단순한 행렬 역연산이 아니다. 동일한 관측 변화가 여러 행동으로 가능하거나, 가려진 접촉·힘 때문에 영상을 보고 행동을 유일하게 정할 수 없기 때문이다.

### 2.4 World model이라는 이름으로 묶이는 세 종류

1. **행동 조건부 시뮬레이터:** 후보 행동에 따라 미래가 어떻게 달라지는지 예측한다. 계획에 직접 연결하기 쉽다.
2. **행동 입력 없는 영상·표현 예측기:** 장면의 시간적 변화에 대한 사전지식을 배운다. 로봇 제어에 쓰려면 행동과 연결하는 추가 학습이 보통 필요하다.
3. **의사결정용 추상 모델:** 모든 관측을 복원하지 않고 보상·가치·목표 달성에 필요한 구조를 예측한다.

이 분류는 상호 배타적이지 않다. 예컨대 웹 영상으로 행동 없는 사전학습을 하고, 로봇 행동 데이터로 조건부 모델을 만들 수 있다. [V-JEPA 2](https://arxiv.org/abs/2506.09985)

### 2.5 무엇을 표현할 것인가?

| 표현 | 장점 | 어려움 | 대표 연결 |
|---|---|---|---|
| 원본 영상 또는 복원 가능한 영상 latent | 사람이 미래를 확인하기 쉽고 영상 데이터를 활용하기 좋음 | 배경·조명처럼 제어와 덜 관련된 세부에도 용량 사용 | DIAMOND, Cosmos, DreamZero |
| 예측용 embedding | 픽셀 복원 부담을 줄이고 의미·동작 정보에 집중 가능 | 거리의 작음이 목표 성공을 뜻하는지 검증 필요 | V-JEPA 2 |
| 보상·가치 중심 latent | 의사결정 목적에 맞게 학습 | 다른 목표에 필요한 정보를 버릴 수 있음 | MuZero, TD-MPC2 |
| 물체·입자·그래프·3D 상태 | 기하·상호작용 구조를 직접 표현 가능 | 상태 추출, 가려짐, 센서 연결이 어려움 | Graph Network-based Simulators |

**두 종류의 latent를 혼동하지 말자.** 영상 VAE latent는 압축 후 복원하기 위한 표현이고, JEPA embedding은 예측과 이해를 위한 표현이다. 둘 다 잠재공간이지만 학습 목적이 다르다. 또한 세계 모델 연구 전체가 영상 생성 연구로 환원되지는 않는다. [GNS](https://arxiv.org/abs/2002.09405)

---

<a id="math"></a>
## 3. 수학적 언어: 상태·관측·행동·목표

### 3.1 최소한의 선수지식

조건부 확률, 기댓값, 신경망의 손실함수와 역전파를 알면 개념 부분부터 읽을 수 있다. 심화 수식에는 다음이 필요하다.

| 개념 | 여기서 필요한 수준 |
|---|---|
| MDP / POMDP | 실제 상태와 관측의 차이, 전이확률, 부분 관측 |
| RL | 보상, 할인율, 정책, 가치함수, actor–critic |
| VAE / 변분추론 | 관측을 압축하고 잠재변수 분포를 학습하는 이유 |
| KL divergence | 두 분포의 차이를 손실로 제약하는 역할 |
| Transformer | 토큰, attention, causal mask, KV cache |
| Diffusion / flow matching | 잡음에서 데이터로 반복 생성하는 방식 |
| MPC | 미래를 계획하되 일부만 실행하고 다시 계획하는 방식 |

처음부터 모든 구현을 익히기보다, 각 개념이 등장하는 절에서 필요한 만큼 보충한다.

### 3.2 POMDP 관점

환경은 다음처럼 표현할 수 있다.

```math
s_{t+1}\sim P(s_{t+1}\mid s_t,a_t),\qquad
o_t\sim O(o_t\mid s_t),\qquad
r_t=R(s_t,a_t).
```

- $s_t$: 실제 환경 상태.
- $o_t$: 센서로 얻은 관측.
- $a_t$: 환경에 적용한 행동.
- $r_t$: $a_t$를 실행한 전이에 대응하는 보상.
- $h_t=(o_0,a_0,\ldots,a_{t-1},o_t)$: 관측과 행동의 이력.
- $b_t(s)=P(s_t=s\mid h_t)$: 가능한 실제 상태에 대한 믿음, 즉 belief.
- $x_t$: 관측과 행동 history로 **추론한 모델 내부의 잠재 상태**. 실제 환경의 $s_t$나 정확한 belief $b_t$와 같다고 가정하지 않는다.

정지 화면 한 장으로는 컵이 왼쪽으로 움직이는지 오른쪽으로 움직이는지 알기 어렵다. **부분 관측 때문에 기억과 이력 조건화가 필요하다.**

학습한 세계 모델의 한 가지 일반형은 다음이다.

```math
p_\theta(x_{t+1},r_t,c_t\mid x_t,a_t),
\qquad p_\theta(o_t\mid x_t).
```

$c_t$는 전이 이후 에피소드가 계속될지 나타내는 continuation 변수다. 관측 decoder, 보상 head, continuation head는 모든 WM에 필수인 것은 아니다.


#### 보충: 실제 상태 $s_t$, 관측 $o_t$, 모델 내부 상태 $x_t$는 어떻게 다른가?

이 문서에서는 아래의 표기를 일관되게 사용한다. **원논문마다 같은 기호를 다르게 쓸 수 있다는 점**은 별도로 표시한다.

| 기호 | 의미 | 얻는 방법 | 공이 움직이는 상황의 예 |
|---|---|---|---|
| $s_t$ | 실제 환경 상태 (true environment state) | 현실에 존재하지만 전부 직접 관측할 수는 없음 | 위치, 속도, 질량, 접촉 상태 |
| $o_t$ | 실제 관측 (observation) | 카메라·센서가 측정 | RGB frame, 로봇 관절 센서값 |
| $x_t$ | 모델의 학습된 잠재 상태 (learned model state) | 관측 및 과거 행동 이력으로부터 모델이 추론 | 현재 공의 운동 방향에 대한 내부 feature |
| $b_t(s)$ | Belief state | 실제 상태에 관한 조건부 확률분포 | 가능한 물리 상태에 대한 불확실성 |

실제 환경에서의 관측 생성은

```math
s_{t+1}\sim P(s_{t+1}\mid s_t,a_t),\qquad
o_t\sim O(o_t\mid s_t).
```

이다. 반면 모델이 내부 표현을 형성하는 *추론 방향*은 개념적으로

```math
x_t\sim q_\phi(x_t\mid o_{\le t},a_{<t})
\quad\text{또는}\quad
x_t=E_\phi(o_{\le t},a_{<t}).
```

이다. 첫 식은 확률적 추론, 둘째 식은 결정론적 인코딩의 추상적인 예시다. 모든 모델이 이 두 식 중 하나를 그대로 구현하는 것은 아니다.

따라서 $s_t\to o_t\to x_t$는 **현실 → 관측 → 모델의 추론**이라는 설명용 화살표다. $o_t$가 latent $x_t$를 직접 측정한 것이 아니다. 학습된 world model의 *generative/decoder 방향*은 내부 상태로부터 관측을 예측하는 반대 방향이기도 하다.

```math
p_\theta(o_t\mid x_t),\qquad
p_\theta(x_{t+1}\mid x_t,a_t).
```

**예시:** 현재 사진에 공이 같은 위치에 있는 두 경우를 상상하자. 하나는 오른쪽으로 이동 중이고 다른 하나는 왼쪽으로 이동 중이다. 한 장의 $o_t$만으로는 구분하기 어렵지만 과거 frame을 보면 모델의 $x_t$는 운동 방향을 다르게 추정할 수 있다. 반대로 영상으로부터 알기 어려운 공의 질량은 $x_t$에 담기지 않을 수도 있다.

**핵심:** $s_t$는 환경의 실제 상태, $o_t$는 측정값, $x_t$는 모델의 학습된 내부 표현이다. $x_t$는 $s_t$의 완전하고 일대일인 복제본이 아니다. 단일 frame을 압축한 VAE latent와 이력을 통합한 RSSM state도 서로 구별해야 한다.

**원문과 이 노트의 표기 차이:** PlaNet과 Dreamer는 **RSSM의 모델 내부 상태**를 $s_t$, recurrent memory를 $h_t$로 표기하기도 한다. 이 문서에서는 실제 환경 상태와 혼동하지 않도록 RSSM의 내부 상태를 $x_t=(u_t,z_t)$로 통일한다. 따라서 원문 수식을 읽을 때에는 이 표기를 대응시켜야 한다.


### 3.3 정책과 계획의 목적

여기서 $g$는 **goal(목표)** 을 뜻한다. 자연어 지시, 목표 이미지, 원하는 최종 상태 또는 이들의 임베딩 등으로 주어질 수 있다. $h_t$는 지금까지의 관측·행동 이력이다. 모든 policy가 반드시 명시적 goal을 입력받는 것은 아니다.

```math
\pi_\psi(a_t\mid h_t,g),\qquad
J(\pi)=\mathbb E_{\pi,P}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].
```

$\gamma$는 미래 보상의 할인율이다. 세계 모델로 계산한 예상 수익 $\hat{J}$가 커도 실제 환경의 $J$가 크다는 보장은 없다. 이것이 model bias 문제의 출발점이다.

보상이 없는 목표 조건부 계획도 가능하다.

```math
a^\star_{t:t+H-1}
=\arg\min_{a_{t:t+H-1}}
\mathbb E_{\hat P_\theta}\left[
d(\hat x_{t+H},x_g)+
\eta\sum_{k=0}^{H-1}C(a_{t+k})
\right].
```

$x_g$는 목표 표현, $d$는 목표까지의 거리, $C$는 행동 비용이다. 이 수식은 **실제 세계가 아니라 모델이 예측한 세계 안에서 최적화**한다.

### 3.4 WAM의 핵심 확률분포

미래 관측열을 $O^{+}=o_{t+1:t+H}$, 행동열을 $A=a_{t:t+H-1}$라고 하면, 학습용 일반형은 다음과 같다. 여기서 **$g$는 목표(goal) 조건**이고 $h_t$는 과거 관측·행동 이력이다. 예컨대 컵을 집는 로봇에서는 $g$를 "컵을 오른쪽 선반에 놓아라"라는 **언어 지시**로 주거나, 컵이 선반 위에 있는 **목표 이미지·상태**로 줄 수 있다.

```math
p_\theta(O^+,A\mid h_t,g).
```

이는 **현재까지 무엇이 일어났는가**($h_t$)와 **어떤 미래를 원하는가**($g$)가 주어졌을 때 미래 관측 및 행동의 결합분포를 모델링한다는 의미다. $g$는 reward $r_t$와 구별된다. Goal은 원하는 결과를 지정하는 조건이고, reward는 행동 결과의 좋고 나쁨을 평가하는 신호다. 또한 이 식은 목표 조건부 WAM을 설명하는 일반형이지 **모든 WAM에 $g$가 반드시 입력되어야 한다는 뜻은 아니다**.

두 가지 분해가 가능하다.

```math
p_\theta(O^+,A\mid h_t,g)
=
p_\theta(O^+\mid h_t,g)\,
p_\theta(A\mid O^+,h_t,g)
```

```math
=
p_\theta(A\mid h_t,g)\,
p_\theta(O^+\mid A,h_t,g).
```

- 첫 번째는 **미래를 그린 다음 그 미래를 실현할 행동을 구한다**는 관점이다.
- 두 번째는 **행동을 제안하고 그 결과를 예측한다**는 관점이다.
- 실제 공동 생성 모델이 이 순서대로 두 개의 독립 모델을 실행한다는 뜻은 아니다.
- 두 분해가 수학적으로 가능하다고 해서 아키텍처·attention mask·학습 절차가 동등한 것은 아니다.
- 확률적 공동 모델링만으로 $P(O^{+}\mid \mathrm{do}(A))$라는 인과적 개입 효과를 정확히 배웠다고 결론 낼 수 없다.

**핵심 질문:** 이 모델은 그럴듯한 미래와 행동의 상관관계를 배우는가, 아니면 내가 선택한 새로운 행동의 결과까지 믿을 만하게 예측하는가? [DreamZero §3.1](https://arxiv.org/html/2602.15922v1#S3.SS1), [WAM 설계 비교 연구](https://arxiv.org/abs/2609.24048)

---

<a id="mechanisms"></a>
## 4. 어떻게 학습하고 행동하는가?

**이 절의 핵심 질문은 관측·행동 데이터로 세계의 변화 규칙을 어떻게 학습하고, 그 예측을 행동 결정에 어떻게 활용하는가?** 이다. 예를 들어 로봇이 공을 밀면 현재 영상 $o_t$, 행동 명령 $a_t$, 다음 영상 $o_{t+1}$, 보상 $r_t$가 기록된다. World model은 이러한 trajectory에서 **다음 관측과 보상을 예측**하도록 학습한다. 내부 상태 $x_t$는 정답이 주어진 물리 상태가 아니라, 예측과 관측 재구성 과정에서 함께 학습하는 잠재 표현이다.

먼저 **RSSM(4.1)**은 이력에서 잠재 상태를 추론하고 확률적 dynamics를 학습하는 방법을 다룬다. 이어 **MPC(4.2)**는 학습한 모델에서 후보 행동의 미래를 비교하고, **Dreamer(4.3)**는 가상 trajectory로 policy를 학습한다. 마지막 **4.4–4.6절**에서는 장기 예측 오차, diffusion/flow 기반 생성, 실시간 closed-loop 제어 문제를 살펴본다.

### 4.1 RSSM: 기억과 불확실성을 함께 담기

PlaNet과 초기 Dreamer 계열을 읽는 열쇠는 **Recurrent State-Space Model(RSSM)**이다. 여기서는 실제 환경 상태 $s_t$와 혼동하지 않도록 RSSM 내부 상태를 $x_t=(u_t,z_t)$로 표기한다. 단순화하면:

```math
u_t=f_\theta(u_{t-1},z_{t-1},a_{t-1}),
```

```math
p_\theta(z_t\mid u_t)
\quad\text{와}\quad
q_\phi(z_t\mid u_t,o_t),
\qquad x_t=(u_t,z_t).
```

- $u_t$: 과거를 누적하는 deterministic recurrent state.
- $z_t$: 여러 가능한 상황을 나타내는 stochastic state.
- **Prior** $p$: 현재 실제 관측 없이 미래를 예측할 때 사용.
- **Posterior** $q$: 실제 관측을 보고 상태를 추정할 때 사용.

#### 보충: Prior $p_\theta$, posterior $q_\phi$와 deterministic/stochastic의 정확한 의미

**우선 두 분포의 역할을 구분하자.**

- **Dynamics prior $p_\theta(z_t\mid u_t)$:** 새로운 센서 관측 $o_t$를 보기 전, 과거 history가 요약된 $u_t$만으로 *현재 latent가 어떤 값을 가질지 예측하는* 분포다.
- **Inference / approximate posterior $q_\phi(z_t\mid u_t,o_t)$:** $u_t$와 새로 들어온 실제 관측 $o_t$를 함께 사용해 *현재 latent의 상태를 추론하는* 분포다.
- 아래의 $\theta$와 $\phi$는 각각 네트워크 파라미터 집합의 표기다. 구현에서는 encoder, transition, decoder 등에서 파라미터 일부를 공유할 수 있고 이름의 분리가 학습 단계의 완전한 독립성을 뜻하지는 않는다.

Generative model에 관측 분포 $p_\theta(o_t\mid u_t,z_t)$가 있으면, 이상적인 Bayes posterior는

```math
p_\theta(z_t\mid u_t,o_t)
\propto p_\theta(o_t\mid u_t,z_t)\,
p_\theta(z_t\mid u_t)
```

이다. 실제로 이 posterior를 계산하기 어려우므로 학습된 inference network $q_\phi$로 **근사**한다. 따라서 $q_\phi$를 '실제 상태를 100% 정확히 아는 모델'로 해석해서는 안 된다.

**중요: deterministic은 '확실한 정보', stochastic은 '무작위 잡음만'이라는 구분이 아니다.**

1. $u_t=f_\theta(u_{t-1},z_{t-1},a_{t-1})$가 deterministic이라는 뜻은 **함수의 입력을 고정했을 때 출력이 고정된다**는 것이다. 하지만 그 입력 중 $z_{t-1}$이 이전에 sampling한 값이면, 미래 rollout마다 $u_t$도 서로 달라질 수 있다. $u_t$가 실제 세계에 대해 항상 정확하다는 의미는 아니다.
2. $z_t$가 stochastic이라는 뜻은 **조건부 분포에서 sampling할 수 있다**는 뜻이다. $u_t$와 새 관측 $o_t$가 정보를 제공하므로, $z_t$는 전혀 예측 불가능한 '순수 난수'가 아니다. 분포가 매우 집중되어 있으면 조건이 주어졌을 때 거의 결정론적인 값을 가질 수도 있다.
3. 따라서 $u_t$에는 오차나 불확실성을 반영한 표현이 들어갈 수 있고, $z_t$에는 규칙적으로 예측되는 정보도 들어갈 수 있다. **두 부분이 의미상 '정확한 물리량 vs 랜덤 물리량'으로 자동 분해된다는 보장은 없다.**

Gaussian stochastic latent의 직관적인 예시는

```math
z_t=\mu_\theta(u_t)+\sigma_\theta(u_t)\odot\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I)
```

이다. $\epsilon_t$는 난수지만 $\mu_\theta$와 $\sigma_\theta$는 $u_t$에 의해 결정된다. 관측이 주어진 posterior에서는 이 평균과 표준편차가 $u_t,o_t$의 함수가 된다. 실제 Dreamer 버전에 따라 Gaussian 대신 categorical latent를 사용하기도 하므로, 이 식은 *모든 버전의 구현 수식이 아닌 개념적 예시*다.

**실제 관측을 사용하는 한 timestep의 순서:**

1. 이전 단계의 $u_{t-1},z_{t-1}$와 행동 $a_{t-1}$로 $u_t$를 **결정론적으로 계산**한다.
2. $p_\theta(z_t\mid u_t)$에서 새 관측 없이 가능한 상태들을 **예측**한다.
3. 카메라에서 $o_t$를 관측하면 $q_\phi(z_t\mid u_t,o_t)$로 latent를 **보정해 추론**한다.
4. 새 관측에 의해 추론한 $z_t$를 사용해 현재의 $x_t=(u_t,z_t)$를 구성하고, 이후 단계의 recurrent memory를 갱신한다.
5. 반대로 **imagination rollout**에서는 새로운 $o_t$가 없으므로 posterior를 호출하지 않고 prior에서 $z_t$를 생성한다.

공이 일정 속도로 움직이는 예에서 $u_t$는 지금까지의 운동 이력을 전달하고, prior는 다음 위치·속도에 대한 예측을 표현할 수 있다. 충돌로 운동이 바뀐 장면을 새로 관측하면 posterior는 이 변화에 맞게 표현을 갱신한다. 이는 이해를 위한 예시이지 내부 차원이 특정 물리량과 일대일 대응한다는 뜻은 아니다.

**왜 prior와 posterior를 가깝게 학습할까?** 관측을 보고 계산한 posterior로만 reconstruction을 잘하더라도, 미래에는 실제 관측이 없어 prior로 여러 단계를 rollout해야 한다. 따라서 RSSM의 KL 항

```math
D_{\mathrm{KL}}\!\left(
q_\phi(z_t\mid u_t,o_t)\,\Vert\,
p_\theta(z_t\mid u_t)
\right)
```

으로 학습 중 두 분포를 연결한다. 단, reconstruction과 지나치게 강한 KL의 균형이 필요하며, '둘을 무조건 동일하게 만든다'는 뜻은 아니다. [PlaNet](https://arxiv.org/abs/1811.04551) · [Dreamer](https://arxiv.org/abs/1912.01603)

훈련 때 관측을 본 posterior만 잘 맞고 prior가 나쁘면, 관측 없는 상상 rollout은 금방 무너진다. 따라서 양쪽을 연결해야 한다.

다음은 관측·보상 예측과 latent regularization을 묶은 **개념적 손실**이다.

```math
\mathcal L_{\rm WM}
=
\sum_t\mathbb E_q[
-\log p_\theta(o_t\mid x_t)
-\log p_\theta(r_t\mid x_t,a_t)]
+\beta D_{\rm KL}
\left(q_\phi(z_t\mid u_t,o_t)\,\|\,p_\theta(z_t\mid u_t)\right).
```

continuation 예측 손실 등을 추가할 수 있다. 실제 논문은 보상 head의 입력·인덱스, KL balancing, free bits/free nats, stop-gradient, 이산 latent 등의 세부가 다르다. 위 수식을 그대로 모든 Dreamer 버전의 구현이라고 읽지 않는다. [PlaNet](https://arxiv.org/abs/1811.04551), [Dreamer](https://arxiv.org/abs/1912.01603)

### 4.2 MPC: 매번 예상하고, 조금 실행하고, 다시 관측하기

~~~text
현재 관측 → 현재 latent 추정
          → 후보 행동열 N개 생성
          → 세계 모델에서 H단계 rollout
          → 예상 보상·목표 거리·가치로 평가
          → 가장 좋은 행동열의 첫 행동 또는 짧은 구간 실행
          → 새 관측으로 다시 시작
~~~

CEM(Cross-Entropy Method)은 좋은 후보들만 남겨 행동열 샘플링 분포의 평균·분산을 갱신하는 방법이다. 이름의 cross-entropy가 분류 손실을 쓴다는 뜻은 아니다.

**장점:** 실제 관측으로 오차를 자주 교정하고 목표가 달라져도 계획을 바꿀 수 있다.  
**비용:** 후보 수 × rollout 길이 × 최적화 반복 횟수만큼 추론 비용이 증가한다.  
**한계:** 짧은 예측이 먼 목표를 향한 좋은 행동을 구별할 충분한 신호를 제공하지 못할 수 있다. [PlaNet](https://arxiv.org/abs/1811.04551), [V-JEPA 2 계획 절](https://arxiv.org/html/2506.09985v1#S4), [Planning Limits](https://arxiv.org/abs/2609.39235)

### 4.3 Dreamer: 상상 속에서 정책을 미리 학습하기

MPC는 행동할 때 후보를 검색한다. Dreamer의 핵심은 **훈련 때 상상 rollout으로 actor와 critic을 개선해서, 실행할 때는 정책을 빠르게 호출하는 것**이다.

~~~text
실제 데이터 → 세계 모델 업데이트
           → 실제 데이터에서 출발 상태 추정
           → 정책으로 행동을 뽑아 latent 미래 상상
           → 예상 보상·가치로 critic 학습
           → 상상 속 성과가 좋아지도록 actor 학습
           → 개선된 정책으로 실제 경험 수집
~~~

짧은 상상을 장기 가치와 연결하는 예가 λ-return이다.

```math
G^\lambda_t=\hat r_t+
\gamma\hat c_t\left[(1-\lambda)V(\hat x_{t+1})
+\lambda G^\lambda_{t+1}\right],
\qquad G^\lambda_{t+H}=V(\hat x_{t+H}).
```

이는 단순화한 재귀식이다. 상상 구간 끝의 critic이 그 뒤의 보상을 추정한다. 하지만 부정확한 critic을 붙이면 먼 미래가 정확해지는 것은 아니다. Dreamer 버전별 actor gradient 구성도 서로 다르므로, “모든 버전이 순수하게 동역학 미분만 쓴다”고 일반화하지 않는다. [Dreamer](https://arxiv.org/abs/1912.01603), [DreamerV3](https://arxiv.org/abs/2301.04104)

### 4.4 예측 오차가 누적되는 이유

한 단계 후 예측 $\hat{x}_{t+1}$를 다시 입력해 $\hat{x}_{t+2}$를 만들면, 모델은 점점 실제 데이터에서 벗어난 상태를 보게 된다.

세 가지 다른 문제가 있다.

1. **동역학 오차:** 행동의 결과를 잘못 예측한다.
2. **목표 평가 오차:** 예측은 맞아도 embedding 거리나 value가 성공을 잘 측정하지 못한다.
3. **탐색·행동 오차:** 좋은 후보를 못 만들거나 계산 지연 때문에 예상과 다른 시점에 실행한다.

“모델을 키운다”는 해결책은 이 세 문제 모두를 동시에 해결하지 않는다. 짧은 rollout, 재계획, 가치함수, 서브골, 데이터 확장, 불확실성 추정은 서로 다른 문제에 대응한다.

### 4.5 Diffusion과 flow matching을 읽는 최소 수식

행동이나 영상은 가능한 미래가 여러 개이므로 하나의 평균만 회귀하면 부자연스러울 수 있다. 확률적 생성 모델은 다양한 가능한 결과를 샘플링한다.

DDPM의 대표적인 학습 형태:

```math
x_\tau=\sqrt{\bar\alpha_\tau}x_0+
\sqrt{1-\bar\alpha_\tau}\epsilon,\qquad
\mathcal L_{\rm noise}
=\mathbb E\|\epsilon-\epsilon_\theta(x_\tau,\tau,\mathrm{cond})\|^2.
```

여기서 $\tau$는 **환경 시간 $t$와 다른 잡음 단계**다.

Flow matching의 단순한 직선 경로 예:

```math
x_\tau=(1-\tau)\epsilon+\tau x_{\rm data},
\qquad
\mathcal L_{\rm FM}
=\mathbb E\left\|
v_\theta(x_\tau,\tau,\mathrm{cond})-
(x_{\rm data}-\epsilon)
\right\|^2.
```

추론은 잡음에서 출발해 $\frac{dx}{d\tau}=v_\theta$를 적분하는 과정이다. 실제 방법은 경로, 가중치, parameterization, solver가 달라진다.

WAM에서는 $x$가 영상 latent와 행동을 함께 담을 수 있다. **둘을 같은 모델로 생성한다는 것과 실제 로봇 실행 결과가 정확히 일치한다는 것은 별도로 검증할 문제**다. [DDPM](https://arxiv.org/abs/2006.11239), [Flow Matching](https://arxiv.org/abs/2210.02747)

### 4.6 행동 chunk, 실시간성, 폐루프

- **Action chunk:** 매번 하나의 행동 대신 여러 미래 제어 명령을 생성한다.
- **Open-loop:** 생성한 구간을 관측 갱신 없이 실행한다.
- **Closed-loop:** 새 센서 관측을 반영해 행동을 갱신한다.
- **Asynchronous execution:** 추론과 로봇 실행을 겹쳐 계산 지연을 줄인다.

영상은 낮은 fps, 관절 제어는 높은 Hz로 동작할 수 있다. 따라서 “영상 16프레임”과 “행동 16개”가 같은 시간 길이를 뜻한다고 가정하면 안 된다. 논문에서 timestamp, chunk 길이, 실행 비율, latency를 확인한다.

또한 **정책이 chunk를 갱신하는 주파수**와 **하위 제어기의 servo 주파수**는 다르다. DreamZero의 약 7 Hz 보고를 로봇의 모든 제어 루프가 7 Hz라는 뜻으로 읽지 않는다. [Diffusion Policy](https://arxiv.org/abs/2303.04137), [DreamZero](https://arxiv.org/html/2602.15922v1)


---

<a id="papers-rl"></a>
## 5. 필독 논문 I: 모델 기반 RL과 상상 학습

이 절은 **작은 환경에서 직접 상호작용하며 배운 모델을 어떻게 제어에 쓰는가**에 초점을 둔다. 각 논문은 먼저 구조 그림을 보고, 모델 학습 손실과 행동 선택 절차를 따로 읽는다.

### 5.1 Dyna — 가상 경험도 학습 데이터가 된다

**원문:** Sutton, *Dyna, an Integrated Architecture for Learning, Planning, and Reacting* (1991). [논문 DOI](https://doi.org/10.1145/122344.122377), [관련 초기 Dyna 장, MIT Press](https://direct.mit.edu/books/edited-volume/3977/chapter/165869/First-Results-with-Dyna-an-Integrated-Architecture)

- **문제:** 실제 환경에서 얻는 경험은 비싸고 제한적이다.
- **아이디어:** 실제 전이로 환경 모델을 학습하고, 그 모델이 만든 가상 전이로 가치·정책 학습을 추가한다.
- **왜 중요한가:** 신경망·영상 생성 이전부터 “학습한 세계 안에서 연습하기”라는 구조가 존재했다.
- **읽을 질문:** 모델로 가상 경험을 많이 만들수록 항상 좋은가? 틀린 모델이 만드는 경험은 어떻게 학습을 왜곡할까?

### 5.2 World Models (Ha & Schmidhuber, 2018) — 시각, 기억, 제어의 분리

**원문:** *World Models* (2018). [논문](https://arxiv.org/abs/1803.10122) · [공식 프로젝트](https://worldmodels.github.io/)

#### 문제의식: 컵을 이해하는 일과 행동을 잘하는 일을 분리하자

로봇 팔이 컵을 집으려면 **(1) 영상에서 컵과 그리퍼를 표현하고 (2) 과거 움직임을 기억해 다음 상태를 예측하며 (3) 보상을 얻는 행동을 결정**해야 한다. 이 모든 능력을 reward만으로 학습시키는 대신 시각 인코더 **V**, 동역학 메모리 **M**, 행동 선택기 **C**로 역할을 나누자는 아이디어다.

> 아래 컵 예시는 **설명을 위해 구성한 가상 실험**이다. World Models 논문의 실제 제어 실험은 CarRacing과 VizDoom이다. 로봇 팔에서 실험했다고 이해해서는 안 된다.

#### 먼저 기호와 실제 입력을 대응하자

| 변수 | 컵 옮기기 예시에서 뜻 | 실제 데이터인지, 학습한 내부 표현인지 |
|---|---|---|
| $o_t$ | $t$에 찍힌 컵·그리퍼 RGB 영상 | 실제 센서 **입력** |
| $a_t$ | 그리퍼 이동/닫기 명령 | Controller **출력**, 실행된 행동 |
| $z_t$ | VAE가 $o_t$를 압축해 만든 벡터 | **학습한 시각 latent**; 좌표 라벨이 아님 |
| $h_t$ | MDN-RNN의 hidden state | **학습한 과거 정보의 요약**; 물리 속도 라벨이 아님 |
| $\hat z_{t+1}$ | M이 예측하는 다음 latent sample | 다음 관측 $o_{t+1}$의 VAE latent와 대응 |
| $W_c,b_c$ | Controller 파라미터 | 누적 reward를 높이도록 최적화 |

**표기 주의:** World Models의 $z_t$는 **VAE가 단일 이미지에서 만든 latent**다. PlaNet의 $z_t$는 RSSM 내부의 **stochastic state**다. 둘을 동일한 모델로 읽어서는 안 된다. 또한 일부 논문은 실제 이미지를 $x_t$라 표기하지만, 이 노트에서는 이미지에 $o_t$를 예약하고 내부 전체 상태에 $x_t$를 사용한다.

#### V: 이미지의 무엇을 학습하는가?

카메라 영상 $o_1$에는 컵·그리퍼·배경이 보인다. VAE Encoder는 이를 $z_1$로 압축하고 Decoder로 같은 영상 $o_1$을 복원한다.

```math
\mathcal L_V(o)
=-\mathbb E_{q_\phi(z\mid o)}\log p_\theta(o\mid z)
+D_{\mathrm{KL}}(q_\phi(z\mid o)\Vert p(z)).
```

- **입력과 target:** 이미지 $o_t$가 주어지고, **복원할 target도 $o_t$ 자체**다.
- **학습되는 것:** Encoder/decoder와 압축 representation $z_t$.
- **주의:** $z_t$의 첫 차원이 컵 좌표이고 다음 차원이 속도가 되도록 직접 감독하는 것은 아니다.

#### M: 다음 순간에 무엇이 일어날까?

$z_1$을 본 뒤 $a_1=$ “그리퍼를 닫기”를 실행한다고 하자. MDN-RNN은 현재 $z_1$, 동작 $a_1$, 기존 memory $h_1$으로 **다음 영상의 latent $z_2$에 대한 조건부 분포**를 예측한다.

```math
p_\theta(z_{t+1}\mid z_t,a_t,h_t)
=\sum_{k=1}^K\pi_{t,k}
\mathcal N(z_{t+1};\mu_{t,k},\Sigma_{t,k}),
\qquad\sum_k\pi_{t,k}=1.
```

컵이 잘 잡힐 수도 있고 미끄러질 수도 있어서 여러 미래가 가능하다. Mixture는 이런 다중 가능성을 표현할 수 있으나, **각 component가 '잡기 성공'과 '실패'라는 이름으로 감독되는 것은 아니다.**

- **입력:** 현재 VAE latent $z_t$, 이미 실행한 action $a_t$, 과거 history $h_t$.
- **Training target:** 실제 행동 뒤 얻은 **다음 센서 영상 $o_{t+1}$을 VAE로 encoding한 latent $z_{t+1}$**.
- **Loss:** 해당 target latent의 MDN 예측분포에 대한 negative log-likelihood.
- **$h_t$의 뜻:** 과거 정보에 관한 RNN hidden vector로, 정해진 물리 상태 좌표가 아니다.

#### C: 그러면 어떤 행동을 할까?

```math
a_t=W_c[z_t;h_t]+b_c.
```

컵 예제에서는 그리퍼 이동·닫기 명령을 출력한다고 생각하면 된다. **실제 CarRacing에서는 조향·가속·브레이크 행동**에 대응한다. 여기서 Controller는 “정답 action” 레이블에 대한 imitation loss를 학습하지 않는다. **환경에서 얻는 누적 reward를 높이도록 CMA-ES로 controller 파라미터를 최적화한다.**

#### 같은 한 전이를 V → M → C로 읽기

1. **$o_0\to z_0$:** 컵이 보이는 실제 영상을 VAE가 압축한다.
2. **$[z_0;h_0]\to a_0$:** Controller가 컵 쪽으로 이동하는 명령을 낸다.
3. **$(z_0,a_0,h_0)\to p(z_1\mid\cdot)$:** M이 *이 행동의 다음 latent*를 예측한다.
4. **실제 $o_1\to z_1$:** 실제로 행동한 뒤 촬영한 영상에서 얻은 latent가 M을 학습시키는 target이 된다. RNN memory도 시간에 따라 갱신된다.
5. **반복:** 같은 방식으로 컵을 잡고 들어 올리는 action을 선택한다. 최종 task reward가 Controller의 optimization signal이다.

즉 **V의 target은 이미지, M의 target은 다음 이미지의 VAE latent, C의 목표는 누적 reward**다. 같은 $o_t$를 다루지만 세 모듈이 *서로 다른 문제*를 학습한다.

#### 'Training Inside of the Dream'

학습한 M에서 $z_{t+1}$을 sample하면 실제 환경을 호출하지 않고도 가상의 미래를 이어갈 수 있다.

```math
(z_t,h_t)\xrightarrow{\mathrm{Controller}}a_t
\xrightarrow{\mathrm{MDN\text{-}RNN}}
(\hat z_{t+1},\hat h_{t+1})
\longrightarrow a_{t+1}\longrightarrow\cdots
```

컵 예제로는 “실제로 컵을 잡지 않고도, 잡는 행동이 만든 미래 latent”를 시뮬레이션하는 셈이다. **논문에서는 VizDoom에서 dream training을 시도**했다. 하지만 model이 컵을 항상 잡힌 것으로 잘못 상상한다면 controller는 그 오류를 악용할 수 있다(model exploitation). 저자들은 sampling **temperature**가 꿈속 학습의 견고성에 미치는 영향을 탐구했다.

#### 무엇을 해결했고 무엇이 남았는가?

- **해결:** VAE 영상 표현, MDN-RNN 기억·예측, 작은 controller를 역할별로 분리해 학습할 수 있음을 보여줬다. CarRacing에서는 $z_t$만 사용하는 경우보다 $h_t$를 함께 쓰는 경우가 유리했다.
- **한계:** learned dynamics가 틀리면 가상 환경에서 높은 reward를 받는 행동이 현실에서 실패할 수 있다. 작고 단순한 controller와 CMA-ES가 복잡한 실제 로봇 전체에 곧바로 적합한 것은 아니다.
- **다음 질문:** Controller를 한 번 훈련시키는 대신, **현재 컵 위치에서 여러 행동열을 latent로 시뮬레이션하고 직접 비교**할 수 있을까? → **PlaNet**.

**이 절을 읽은 뒤 설명할 것:** V의 입력/target, M의 입력/target, C의 학습 목표를 컵 예제로 각각 한 문장씩 말해 보자.

---

### 5.3 PlaNet (Hafner et al., ICML 2019) — RSSM과 latent planning

**원문:** *Learning Latent Dynamics for Planning from Pixels* ([2018 프리프린트](https://arxiv.org/abs/1811.04551), [ICML 2019](https://proceedings.mlr.press/v97/hafner19a.html)) · [프로젝트](https://planetrl.github.io/)

#### 문제의식

World Models가 '학습된 환경에서 controller를 훈련할 수 있다'를 보였다면 PlaNet은 더 직접적으로 묻는다.

> 이미 미래 dynamics를 모델링할 수 있다면, **현재 상황에서 여러 action sequence를 시뮬레이션하고 최선의 행동을 곧바로 선택**할 수 있지 않을까?

원본 픽셀을 생성하며 수천 개 후보를 평가하기에는 비용이 너무 높다. 따라서 **미래 예측도 후보 행동 평가도 latent space에서만 수행**한다.

#### 방법론 ①: Recurrent State-Space Model (RSSM)

이 노트에서는 RSSM 전체 상태를 $x_t=(u_t,z_t)$라고 정의한다. $u_t$는 결정론적 기억, $z_t$는 확률적 변수다. **PlaNet 원문은 이 내부 상태를 $s_t$, recurrent memory를 $h_t$로 쓰지만, 여기에서는 실제 환경 상태 $s_t$와의 혼동을 피하려고 기호를 변경했다.**

```math
\begin{aligned}
u_t&=f_\theta(u_{t-1},z_{t-1},a_{t-1}),\\
p_\theta(z_t\mid u_t)&:\text{ dynamics prior},\\
q_\phi(z_t\mid u_t,o_t)&:\text{ observation posterior},\\
p_\theta(o_t\mid u_t,z_t)&:\text{ observation decoder},\\
p_\theta(r_t\mid u_t,z_t)&:\text{ reward predictor}.
\end{aligned}
```

**왜 deterministic + stochastic인가?**

- 오직 deterministic RNN이면 같은 상태·행동에서 다중 가능한 미래를 확률분포로 표현하기 어렵다.
- 오직 stochastic state만 쓸 경우 긴 이력의 안정적인 정보 전달이 어려울 수 있다.
- 따라서 $u_t$가 과거의 정보를 전달하고, $z_t$가 미래의 확률적 불확실성을 표현한다. 단, 개별 차원의 의미가 해석 가능하게 감독되는 것은 아니다.

여기서 **posterior는 실제 관측을 본 후의 추론**, **prior는 아직 미래 관측을 볼 수 없을 때의 예측**이라는 차이를 반드시 기억하자.

#### 방법론 ②: World model의 변분 학습

Observation/reward의 negative log-likelihood와 one-step prior/posterior KL로 모델을 학습한다. RSSM을 압축하여 나타낸 개념식은

```math
\mathcal L_{\rm RSSM}
=
\sum_t \mathbb E_q
\left[
-\log p_\theta(o_t\mid x_t)
-\log p_\theta(r_t\mid x_t)
\right]
+\beta\sum_t
\mathbb E_q
D_{\rm KL}\!\left(
q_\phi(z_t\mid u_t,o_t)\Vert
p_\theta(z_t\mid u_t)\right).
```

학습에서는 관측으로 posterior 상태를 추정하지만, 미래 rollout 동안에는 새로운 관측이 없으므로 prior로 샘플링한다.

#### 방법론 ③: Latent Overshooting

One-step loss를 작게 만들어도 여러 번 rollout하면 예측 오차가 누적될 수 있다. 그래서 **$d$-step latent prior를 실제 데이터를 반영한 posterior와 맞추는** auxiliary objective를 제안했다.

```math
\mathcal L_{\rm over}
\;\sim\;
\sum_{t,d}
\mathbb E\,
D_{\rm KL}\!\left(
q_\phi(z_t\mid u_t,o_t)
\Vert
p_\theta^{(d)}(z_t\mid x_{t-d},a_{t-d:t-1})
\right).
```

위 식은 목적을 설명하는 **개념적 표기**다. 정확한 논문 수식에는 중간 확률적 상태에 대한 적분·기댓값과 conditioning이 포함된다.

**중요한 실험상의 주의:** Latent overshooting은 중요한 연구 아이디어지만, 최종 RSSM의 성능 향상을 모두 설명하는 필수 요소는 아니다. PlaNet의 ablation에서는 모델 종류에 따라 효과가 달랐다.

#### 방법론 ④: Cross-Entropy Method (CEM)와 MPC

길이 $H$의 후보 행동열 $A=(a_t,\ldots,a_{t+H-1})$을 만들고 학습된 모델을 rollout하여 예상 누적 reward를 계산한다.

```math
A^\star\approx\arg\max_A
\mathbb E_{\hat p_\theta}
\left[
\sum_{k=0}^{H-1}\gamma^k\hat r_{t+k}
\right].
```

1. Action-sequence Gaussian proposal을 초기화한다.
2. 여러 행동열을 sampling하고 각각 RSSM에서 rollout한다.
3. 높은 reward를 얻은 elite sample을 선정한다.
4. Elite로 proposal의 평균·분산을 갱신한다. 이 과정을 반복한다.
5. 가장 좋은 행동열의 **첫 번째 행동만** 실행한다.
6. 다음 실제 observation을 받아 다시 상태를 추론하고 다시 계획한다(receding-horizon MPC).

World model의 미래는 **영상 decoder 없이** latent와 reward head만으로 평가한다. PlaNet은 Actor/Critic을 학습해 policy로 즉시 행동하는 방식이 아니다.

#### 해결한 문제와 한계

- **해결:** 원본 영상에서 학습한 stochastic dynamics를 실제 online control에 활용했다. Latent planning으로 image generation 비용 없이 후보 행동의 미래를 비교한다. 픽셀 기반 model-free RL보다 높은 sample efficiency를 보고했다.
- **한계:** 환경 step마다 CEM을 반복하므로 inference 비용이 크다. 짧은 planning horizon 밖의 장기 reward를 놓칠 수 있다. Horizon을 무작정 늘리면 model error와 비용이 증가한다.
- **Dreamer로 이어지는 핵심 질문:** 이러한 반복적인 online optimization의 결과를 **상상한 trajectory로부터 미리 policy에 학습시킬 수 없을까?**

**읽을 질문**
- RSSM posterior가 필요한 순간과 prior가 필요한 순간은 정확히 언제인가?
- Latent overshooting은 왜 decoded image loss를 추가하는 방법보다 비용 면에서 유리한가?
- CEM에서 후보 action의 수·horizon·iteration 수를 키우면 무엇이 늘고 무엇이 악화되는가?
- MPC가 첫 번째 action만 실행하는 이유는 무엇인가?

---

### 5.4 Dreamer 계보 — 잠재 상상에서 Actor-Critic을 학습하기

#### 5.4.1 Dreamer (Hafner et al., ICLR 2020)

**원문:** *Dream to Control: Learning Behaviors by Latent Imagination* ([2019 프리프린트](https://arxiv.org/abs/1912.01603), [ICLR 2020](https://openreview.net/forum?id=S1lOTC4tDS))

**문제의식:** PlaNet은 매 단계마다 후보 action sequence를 대량으로 최적화해야 한다. 이미 differentiable한 world model이 있다면 **그 안에서 미래를 상상해 actor와 critic을 학습**하고, 실제 행동 시에는 policy만 한 번 호출하면 되지 않을까?

**World model:** PlaNet의 RSSM을 사용한다. 실제 데이터를 담은 replay buffer에서 posterior state를 얻고, 그 지점 이후에는 learned dynamics prior로 imagined trajectory를 생성한다. 여기의 $x_t$는 실제 환경의 $s_t$가 아니라 모델 내부의 RSSM 상태다.

```math
a_\tau\sim\pi_\psi(a_\tau\mid x_\tau),\quad
x_{\tau+1}\sim p_\theta(x_{\tau+1}\mid x_\tau,a_\tau),\quad
\hat r_\tau\sim p_\theta(r_\tau\mid x_\tau).
```

**Actor와 Critic:**

```math
\pi_\psi(a_\tau\mid x_\tau),\qquad
V_\xi(x_\tau)\approx
\mathbb E\left[\sum_{k=0}^{\infty}\gamma^k r_{\tau+k}\mid x_\tau\right].
```

짧게만 imagine하면 먼 미래의 보상을 놓칠 수 있으므로, 마지막 state의 value로 bootstrap한다. 길이 $n$의 return은

```math
G_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^k\hat r_{t+k}
+\gamma^n V_\xi(x_{t+n})
```

이고 길이 $H$의 imagination에서 $\lambda$-return은

```math
G_t^\lambda
=(1-\lambda)\sum_{n=1}^{H-1}
\lambda^{n-1}G_t^{(n)}
+\lambda^{H-1}G_t^{(H)}.
```

이 식은 PPO에서 공부하는 GAE/TD($\lambda$)와 밀접하다. 여러 bootstrap horizon의 bias–variance trade-off를 조절한다.

**가장 큰 기여: differentiable dynamics를 통한 actor gradient.**
간단히 deterministic reparameterized dynamics를 쓰면

```math
x_{t+1}=f_\theta(x_t,a_t),\qquad
a_t=g_\psi(x_t,\epsilon_t)
```

이므로

```math
\frac{\partial x_{t+1}}{\partial\psi}
=
\frac{\partial f_\theta}{\partial x_t}
\frac{\partial x_t}{\partial\psi}
+
\frac{\partial f_\theta}{\partial a_t}
\frac{\partial g_\psi}{\partial\psi}.
```

즉 단지 $\nabla_\psi\log\pi_\psi$에 return을 곱하는 score-function estimator만 쓰는 것이 아니라, **액션이 미래 상태·보상에 미치는 경로를 미분**할 수 있다. 확률적 RSSM과 policy에서는 reparameterization을 활용한다.

Critic은 stop-gradient된 $\lambda$-return을 regression target으로 학습한다.

```math
\mathcal L_V=
\mathbb E\left[(V_\xi(x_t)-\operatorname{sg}(G_t^\lambda))^2\right].
```

**해결:** 온라인 CEM을 매번 반복하는 대신 상상 속에서 정책과 가치함수를 훈련하고 실제 실행 시 actor를 호출할 수 있게 했다. 다수의 visual continuous-control task에서 sample efficiency와 수행 성능의 개선을 보고했다.

**남은 문제:** 보상의 규모, action space, 환경의 관측 특성 등에 따라 hyperparameter와 optimization stability가 크게 달라진다. 학습된 모델의 bias는 상상에서 최적화한 policy로 전달될 수 있다.

**읽을 질문**
- Latent imagination의 첫 state는 posterior인데 다음 상태는 왜 prior로 생성하는가?
- Original Dreamer의 actor gradient와 PPO의 likelihood-ratio gradient는 어떻게 다른가?
- Critic이 있다면 짧은 imagination horizon으로도 장기 return을 다룰 수 있는 이유는?
- World model의 loss와 actor의 목표함수는 왜 분리되는가?

#### 5.4.2 DreamerV2 — V3를 이해하기 위한 연결 고리

**원문:** *Mastering Atari with Discrete World Models* ([2020 프리프린트](https://arxiv.org/abs/2010.02193)).

DreamerV2는 discrete/categorical stochastic latent, straight-through gradient, KL balancing 등을 도입·발전시켜 Atari에서도 강력한 imagination learning을 구현했다. **따라서 categorical latent와 KL balancing을 모두 DreamerV3의 최초 발명으로 귀속하면 안 된다.** 이 노트에서는 V2를 V3의 안정화 설계를 이해하기 위한 브리지로 읽는다.

#### 5.4.3 DreamerV3 (Hafner et al., 2023) — 다양한 환경에서 공통 설정으로 학습

**원문:** *Mastering Diverse Domains through World Models* ([2023 프리프린트](https://arxiv.org/abs/2301.04104), [공식 구현](https://github.com/danijar/dreamerv3)).

**문제의식:** Dreamer의 핵심 구조가 효과적이라도 게임·로봇·Minecraft 등에서는 observation/reward scale, sparse reward, continuous/discrete action, gradient stability가 다르다. 환경마다 손실 가중치 등을 다시 조절하면 범용성을 얻기 어렵다.

**주요 방법 ①: RSSM의 안정화.**
Prediction loss와 dynamics/representation KL loss를 나누고, 서로 다른 stop-gradient 방향으로 prior와 posterior를 업데이트한다.

```math
\mathcal L_{\rm WM}
=
\mathcal L_{\rm pred}
+\beta_{\rm dyn}\mathcal L_{\rm dyn}
+\beta_{\rm rep}\mathcal L_{\rm rep}.
```

개념적으로

```math
\begin{aligned}
\mathcal L_{\rm dyn}
&=\max\!\left(1,\,
D_{\rm KL}(\operatorname{sg}(q_\phi)\Vert p_\theta)\right),\\
\mathcal L_{\rm rep}
&=\max\!\left(1,\,
D_{\rm KL}(q_\phi\Vert\operatorname{sg}(p_\theta))\right).
\end{aligned}
```

각각은 **dynamics prior가 posterior를 따라가도록**, 그리고 **representation posterior가 prior와 지나치게 멀어지지 않도록** 작용한다. $\max(1,\cdot)$는 KL이 충분히 작을 때 더 줄이는 gradient를 중단하는 free-bits/free-nats 설계다. 논문에서 사용한 비중은 $\beta_{\rm dyn}=1$, $\beta_{\rm rep}=0.1$이다.

**주요 방법 ②: Symlog 및 distributional prediction.**

```math
\operatorname{symlog}(x)
=\operatorname{sign}(x)\log(1+|x|),\qquad
\operatorname{symexp}(y)
=\operatorname{sign}(y)(e^{|y|}-1).
```

큰 크기의 값을 압축해 환경마다 다른 수치 규모에 덜 민감하게 한다. Reward와 value에는 연속 target을 인접한 두 bin에 분배한 **two-hot categorical prediction**을 사용한다.

예: target $2.3$을 bin $2$, $3$ 사이에서 0.7과 0.3으로 나누는 방식. 실제 구현에서는 symlog 좌표의 bin을 사용한다.

**주요 방법 ③: Return-scale normalization과 Actor objective.**

DreamerV3에서 중요한 차이는 **original Dreamer와 달리 actor 학습에 REINFORCE 형태의 score-function estimator를 사용한다는 점**이다. 개념적으로

```math
\mathcal L_\pi
=-
\mathbb E\left[
\frac{\operatorname{sg}(G_t^\lambda-V_\xi(x_t))}
{\max(1,S)}
\log\pi_\psi(a_t\mid x_t)
+\eta\mathcal H(\pi_\psi(\cdot\mid x_t))
\right].
```

여기서 $S$는 $\lambda$-return의 5–95 percentile span을 기반으로 하는 EMA scale이다. $\max(1,S)$로 매우 작은 scale에서 advantage가 과도하게 증폭되는 것을 막는다. 원 논문 구현의 세부 entropy/normalization 항과 구별해 이해할 것.

**해결:** Atari, ProcGen, DMLab, DeepMind Control, Minecraft 등 서로 다른 유형의 환경을 고정 hyperparameter로 다루며 강력한 성능을 보고했다. 특히 Minecraft diamond 획득은 장기 계획과 sparse reward의 사례로 중요하다.

**한계:** 모델이 부정확할 때 정책이 그 오류를 활용할 수 있다는 근본적인 model-bias 문제는 남는다. 여러 환경에서 공통 설정이 작동했다는 경험적 결과가 모든 실제 환경에서의 물리적 정확성을 보장하지는 않는다.

**읽을 질문**
- Dynamics KL과 representation KL은 서로 왜 다른 stop-gradient를 적용하는가?
- Symlog의 도함수는 큰 $|x|$에 대해 얼마나 감소하는가?
- Two-hot과 scalar MSE는 loss scale과 gradient에서 어떻게 다른가?
- V3가 original Dreamer의 analytic gradient 대신 REINFORCE를 쓰는 장단점은?

#### 5.4.4 네 논문의 핵심 차이를 한 번에 비교

| 논문 | World representation | 행동 선택 | 가장 중요한 변화 |
|---|---|---|---|
| World Models | VAE + MDN-RNN | 작은 controller, CMA-ES | 시각·동역학·제어를 분리하고 꿈 안에서 제어 학습 |
| PlaNet | RSSM | 매 순간 CEM + MPC | latent로 미래를 예측하고 온라인 행동 계획 |
| Dreamer | RSSM | imagined actor–critic | 상상한 경험으로 정책을 학습해 반복 계획 비용 감소 |
| DreamerV3 | 안정화한 categorical RSSM | REINFORCE 기반 actor–critic | 서로 다른 reward/관측 환경에서 공통 설정 |

**요약:** 분리된 representation과 dynamics → latent planning → imagination-trained policy → 범용적인 안정화. 이 화살표는 각 논문이 이전 논문을 전부 대체한다는 뜻이 아니다. 무엇을 예측하는지와 그것을 어떤 행동 선택 알고리즘에 연결하는지가 다른 설계 선택이다.

---

### 5.5 MuZero — 세계의 모든 세부를 예측할 필요는 없다

**원문:** Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (2019). [논문](https://arxiv.org/abs/1911.08265)

핵심 구성은 $h$(representation), $g$(dynamics), $f$(prediction)이다. 여기의 $h$는 논문의 함수 이름이며 이 노트의 이력 $h_t$와 다르다.

~~~text
과거 관측 → representation → latent
latent + 행동 → dynamics → 다음 latent + 보상
latent → prediction → 정책 prior + 가치
이 구조로 MCTS 탐색
~~~

- **핵심:** 다음 화면 복원 없이 계획에 필요한 보상·정책·가치를 학습한다.
- **의미:** 세계 모델을 평가하는 목적이 의사결정이라면, photorealism은 필수 조건이 아닐 수 있다.
- **한계·해석:** latent가 사람이 해석하는 물리 상태와 일치한다고 보장하지 않으며, 실험 밖의 임의 목표와 개입에 대한 정확성도 별도 문제다.
- **읽을 부분:** 모델의 세 함수, unroll 학습, MCTS, 실제 관측으로 보정하는 위치.

### 5.6 TD-MPC2 — decoder-free 제어 모델의 확장

**원문:** Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control* (2023). [논문](https://arxiv.org/abs/2310.16828), [공식 프로젝트](https://www.tdmpc2.com/)

- **아이디어:** 관측 reconstruction 대신 latent dynamics·보상·가치를 제어 목적에 맞게 학습하고, 짧은 trajectory optimization과 terminal value를 결합한다.
- **중요성:** Dreamer 외에도 강력한 모델 기반 제어 계보가 있으며, decoder 없이도 세계 모델을 쓸 수 있음을 보여준다.
- **확장:** 여러 작업·로봇·행동 공간을 다루는 큰 모델과 데이터의 효과를 연구한다.
- **한계·해석:** 학습한 가치와 작업 조건에 기대는 만큼, 일반 영상 이해나 임의 목표에 대한 범용 시뮬레이터와 동일하게 보지 않는다.

**비교 질문:** MuZero, TD-MPC2, JEPA는 모두 픽셀 복원 없이 동작할 수 있는데, 각각 무엇을 정답으로 삼아 latent를 학습하는가?

### 5.7 IRIS와 DIAMOND — 세계 모델의 표현 방식이 행동을 바꾼다

**IRIS:** Micheli et al., *Transformers are Sample-Efficient World Models* (2022). [논문](https://arxiv.org/abs/2209.00588)

이미지를 discrete autoencoder의 토큰으로 표현하고, 자기회귀 Transformer로 미래를 모델링한 뒤 상상 속에서 정책을 학습한다. 중요한 연결은 **언어모델식 시퀀스 예측이 이미지·행동 전이에도 쓰일 수 있다**는 것이다. 토큰화가 작은 물체의 위치나 움직임 정보를 잃으면 제어에도 영향을 줄 수 있다.

**DIAMOND:** Alonso et al., *Diffusion for World Modeling: Visual Details Matter in Atari* (2024). [논문](https://arxiv.org/abs/2405.12399)

확산 세계 모델에서 생성한 미래를 사용해 RL을 수행한다. 작은 시각적 차이가 게임 결과를 바꾸는 환경에서는 영상 세부의 보존이 의사결정에도 중요하다는 관점을 제공한다.

**같이 읽는 이유:** “픽셀은 낭비”와 “픽셀은 중요”를 보편적인 양자택일로 보지 않는다. 작업에 필요한 정보를 어떤 표현이 보존하는지 실험으로 판단해야 한다.

### 5.8 Dreamer 4 — 큰 세계 모델 안에서 오프라인 RL

**원문:** Hafner, Yan & Lillicrap, *Training Agents Inside of Scalable World Models* (2025-09). [논문](https://arxiv.org/abs/2509.24527), [구조와 학습 절차](https://arxiv.org/html/2509.24527v1)

- **문제:** 복잡한 물체 상호작용을 장기간 예측하면서 상상 학습을 빠르게 수행하기 어렵다.
- **구조:** causal tokenizer, Transformer dynamics, 적은 생성 단계로 추론하는 shortcut forcing, 정책·보상·가치 학습과 imagination RL.
- **데이터:** 행동이 있는 자료와 행동이 없는 영상을 활용하는 방향을 다룬다.
- **저자 보고:** 오프라인 데이터만 사용하는 Minecraft 설정에서 상상 속 RL을 통해 다이아몬드를 얻었다.
- **해석:** 단순히 “V3보다 큰 RSSM”으로 읽으면 중요한 구조·학습 설정의 변화를 놓친다.
- **한계:** 오프라인 데이터의 행동 범위를 벗어나는 rollout을 어떻게 신뢰할지는 여전히 중요한 문제다.

**WAM과 연결:** 영상에서 배운 세계 지식을 행동에 연결한다는 관심은 공유하지만, **상상 속 보상 최적화로 정책을 개선하는 것**과 **시연에서 영상·행동을 공동 생성하도록 학습하는 것**은 서로 다른 절차다.

---

<a id="papers-foundation"></a>
## 6. 필독 논문 II: 영상·표현 기반 세계 모델

이 절에서는 사전학습 데이터가 환경별 상호작용에서 대규모 영상으로 확장된다. 항상 **사전학습에 행동이 있는지**, **제어 단계에서 행동과 어떻게 연결하는지**를 표시하면서 읽는다.

### 6.1 LeCun의 구상과 V-JEPA — 예측할 표현을 학습하기

**원문:** LeCun, *A Path Towards Autonomous Machine Intelligence* (2022). [position paper](https://openreview.net/forum?id=BZ5a1r-kVsf)  
**구체적 모델:** Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (2024). [V-JEPA 논문](https://arxiv.org/abs/2404.08471)

- **출발점:** 미래 영상에는 빛 반사, 나뭇잎, 배경 등 예측하기 어렵거나 작업과 무관한 정보가 많다.
- **JEPA 관점:** 영상 전체를 복원하기보다, 보이지 않는 부분의 **표현**을 예측하도록 학습한다.
- **V-JEPA:** 영상의 가려진 부분에 대한 feature prediction으로 시각 표현을 학습한다.
- **핵심 구별:** action-free representation pretraining만으로 후보 로봇 행동의 결과를 예측하는 제어 모델이 완성된 것은 아니다.

다음은 학습 구조를 설명하는 개념식이다.

```math
\mathcal L_{\rm pred}
=
\left\|
\mathrm{Predictor}(E_\theta(\mathrm{context}),\mathrm{mask})
-\mathrm{sg}(E_{\bar\theta}(\mathrm{target}))
\right\|.
```

$\operatorname{sg}$는 stop-gradient이고, target encoder는 EMA 등으로 갱신할 수 있다. 표현 붕괴를 막는 세부 설계가 중요하므로, 이 한 줄만 구현하면 V-JEPA가 재현된다고 생각하지 않는다. LeCun의 2022 문서는 성능 비교 논문이 아닌 연구 방향 제안이다.

### 6.2 V-JEPA 2 — 행동 없는 사전학습에서 행동 조건부 계획으로

**원문:** Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* (2025-06). [논문](https://arxiv.org/abs/2506.09985), [학습·계획 설명](https://arxiv.org/html/2506.09985v1)

~~~text
대규모 영상·이미지 → action-free V-JEPA 2 표현 사전학습
로봇 관측 + 실제 행동 → action-conditioned V-JEPA 2-AC 후학습
현재 표현 + 후보 행동 → 미래 표현 예측
목표 이미지 표현과 거리 비교 → CEM / MPC → 실행
~~~

- **핵심:** 보기만 해서 배운 시각 지식을 비교적 적은 로봇 상호작용 자료로 제어와 연결한다.
- **데이터 주의:** 약 62시간의 DROID 자료를 사용한 제어 단계는 작업별 성공 라벨·보상 없이 학습한다는 의미다. **행동 기록까지 없다는 뜻은 아니다.**
- **실험 범위:** 새로운 환경의 Franka 로봇에서 이미지 목표에 따른 reaching·grasping·pick-and-place 계획을 평가한다.
- **한계·해석:** 모든 언어 지시·장기 작업·로봇 형태에 대한 무제한 zero-shot 제어로 일반화하면 안 된다.

**DreamZero와 가장 중요한 차이:** V-JEPA 2-AC는 주로 **후보 행동 → 미래 표현 → 목표 거리**로 계획하고, DreamZero는 **미래 영상과 행동을 공동 생성하는 정책**을 구성한다.

### 6.3 V-JEPA 2.1 — 조밀한 공간·시간 표현

**원문:** Mur-Labadia et al., *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning* (2026-03). [논문](https://arxiv.org/abs/2603.14482)

visible·masked token 모두를 활용하는 dense predictive loss, 여러 중간 층의 self-supervision, 이미지·영상 통합 학습과 확장을 통해 공간적으로 조밀하고 시간적으로 일관된 표현을 만든다.

**읽는 이유:** 행동은 “컵이 있다”는 전체 장면 의미만으로 결정되지 않는다. 손가락과 컵 가장자리의 상대 위치 같은 국소 정보가 필요하다. 이 논문은 전역 의미와 조밀한 표현을 함께 보는 연결 자료다.

**주의:** encoder 표현 개선, action-conditioned predictor, 최종 제어 성공률을 각각 구분해서 평가한다.

### 6.4 Genie — 행동 라벨 없이 상호작용 가능한 세계

**원문:** Bruce et al., *Genie: Generative Interactive Environments* (2024). [논문](https://arxiv.org/abs/2402.15391)

영상 tokenizer, latent action model, dynamics model을 구성해 행동 라벨이 없는 영상에서 상호작용 가능한 환경을 학습한다. 인접 프레임의 변화를 설명하는 latent action을 배워 생성 과정을 조절한다.

**핵심 한계:** latent action의 “왼쪽으로 이동하는 변화”가 실제 로봇의 토크·관절 명령과 같지는 않다. 실제 로봇으로 옮기려면 latent action과 실행 가능한 제어를 연결해야 한다.

**Genie 3:** 2025-08 공식 발표는 실시간 탐색 가능한 세계 생성의 발전을 보여준다. [공식 발표](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)

여기서는 Genie 3를 논문과 동일한 증거 수준으로 취급하지 않는다. 상호작용 시연의 인상과, 장기 상태 보존·정밀 물리·에이전트 학습 효과의 정량 검증을 구별한다.

### 6.5 Diffusion Forcing — 자기회귀와 확산의 결합

**원문:** Chen et al., *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* (2024). [논문](https://arxiv.org/abs/2407.01392)

시퀀스 토큰마다 독립적인 noise level을 적용해 학습한다. 과거를 조건으로 미래 토큰을 만들면서도, 여러 미래 시점에 대한 생성·guidance를 유연하게 조절한다.

**왜 필요한가:** 세계 모델에서는 “과거는 확정된 관측, 가까운 미래는 덜 불확실, 먼 미래는 더 불확실” 같은 구조가 자연스럽다. 전체 시퀀스에 같은 잡음을 넣는 방식과 다른 조건화를 제공한다.

**연결:** Dreamer 4의 shortcut forcing을 읽기 전 배경이다. **환경 시간의 causal order**와 **생성 시 noise를 제거하는 순서**는 서로 다른 축이라는 점을 기억한다.

### 6.6 Cosmos와 Cosmos 3 — 세계 모델을 범용 기반으로 확장

**Cosmos 원문:** *Cosmos World Foundation Model Platform for Physical AI* (2025-01). [논문](https://arxiv.org/abs/2501.03575)

영상 데이터 정제, tokenizer, 사전학습 세계 모델과 후학습 사례를 포함하는 플랫폼을 제안한다. “하나의 로봇 작업을 풀기 위한 모델”에서 “다양한 물리 AI 응용을 위한 기반 모델”로 관심이 확장된다.

**Cosmos 3 원문:** *Cosmos 3: Omnimodal World Models for Physical AI* (2026-06). [논문](https://arxiv.org/abs/2606.02800), [공식 프로젝트](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)

언어·이미지·영상·음성·행동을 mixture-of-transformers 구조로 함께 처리·생성하는 방향을 다룬다. 세계 이해, 생성, 시뮬레이션, 정책을 공통 기반으로 연결하려는 접근이다.

**해석:** 연구 흐름은 “VLA냐 WAM이냐”라는 이름 경쟁을 넘어, **무엇을 조건으로 어떤 모달리티를 생성하며, 그 표현이 제어에 얼마나 유용한가**로 이동하고 있다. 영상 품질·로봇 성공률·동역학 정확도는 각각의 실험과 데이터 조건에서 확인해야 한다.

---

<a id="papers-wam"></a>
## 7. 필독 논문 III: VLA에서 WAM으로

### 7.1 먼저 VLA를 공정하게 이해하기

**RT-2:** *Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* (2023). [논문](https://arxiv.org/abs/2307.15818)

웹에서 배운 시각·언어 지식을 로봇 제어에 연결하고, 행동을 토큰으로 표현해 시각·언어 작업과 함께 학습한다.

**π0:** *A Vision-Language-Action Flow Model for General Robot Control* (2024). [논문](https://arxiv.org/abs/2410.24164)

시각·언어 기반과 flow matching 행동 생성을 연결한다. 행동은 반드시 이산 언어 토큰으로만 출력하는 것이 아니다.

**π0.5:** *A Vision-Language-Action Model with Open-World Generalization* (2025). [논문](https://arxiv.org/abs/2504.16054)

여러 로봇, 웹 데이터, 의미적 subtask, 저수준 행동 등 이질적인 자료를 공동 학습해 새로운 실제 환경에 일반화하는 방향을 다룬다.

**흔한 과장 피하기:** VLA를 “물리를 전혀 모르고 의미만 아는 모델”로 단정할 수 없다. 행동 데이터에서도 동역학에 관한 정보를 간접적으로 학습할 수 있고, 미래 예측을 결합한 VLA도 존재한다. 비교할 것은 **명시적인 미래 학습 목표, 기반 모델, 데이터, 추론 절차**다.

또한 **Diffusion Policy**는 행동 분포를 확산으로 학습하지만, 그것만으로 미래 세계를 예측하는 WM이 되지는 않는다. [논문](https://arxiv.org/abs/2303.04137)

### 7.2 UniPi — 미래 영상 계획에서 행동 추출

**원문:** Du et al., *Learning Universal Policies via Text-Guided Video Generation* (2023). [논문](https://arxiv.org/abs/2302.00111)

~~~text
현재 영상 + 언어 목표 → 미래 영상 계획
현재·미래 프레임 → inverse dynamics → 실제 행동
~~~

- **핵심:** 서로 다른 환경의 상태·행동 공간을 영상이라는 공통 표현으로 연결한다.
- **WAM 계보에서의 의미:** “정책을 영상 생성으로 표현한다”는 아이디어는 2026년 이름이 널리 쓰이기 이전부터 존재했다.
- **한계·해석:** 생성한 계획이 물리적으로 가능해야 하고, 역동역학이 그것을 실행할 수 있어야 한다.
- **읽을 질문:** 미래 영상이 그럴듯해도 그립 접촉이나 힘이 틀리면 어떻게 되는가?

### 7.3 UVA — 공동 학습과 빠른 실행을 양립시키기

**원문:** Li et al., *Unified Video Action Model* (2025-02). [논문](https://arxiv.org/abs/2503.00200), [공식 프로젝트](https://unified-video-action-model.github.io/)

영상과 행동의 joint latent representation을 학습하고, 두 개의 가벼운 diffusion head로 decoding을 분리한다. 추론 때 영상 생성을 건너뛰고 행동을 빠르게 생성할 수 있다. 입력의 일부를 가리는 학습으로 policy, 정동역학, 역동역학 등의 조건화를 지원한다.

**중요한 교훈:** 미래 예측이 정책 학습에 유용하더라도 **실행 때 매번 모든 미래 영상을 렌더링해야 하는 것은 아니다.**

**읽을 질문:** 행동 추론에서 생략할 수 있는 모듈은 무엇이며, 학습 때 영상 손실은 어떤 표현을 개선하는가?

### 7.4 WorldVLA — 자기회귀 행동·세계 모델 통합

**원문:** Cen et al., *WorldVLA: Towards Autoregressive Action World Model* (2025-06). [논문](https://arxiv.org/abs/2506.21539)

하나의 자기회귀 프레임워크에서 이미지와 행동의 이해·생성을 통합한다. 세계 예측은 행동 생성을 돕고, 행동 정보는 미래 영상 예측을 돕는 공동 학습을 다룬다. 행동 chunk를 자기회귀적으로 생성할 때 앞선 행동 오차가 전파되는 문제에 attention mask 전략을 제안한다.

**의미:** WAM을 특정한 video diffusion backbone만으로 정의하면 이 계보를 놓친다. 통합 학습은 자기회귀, diffusion, representation prediction 등 여러 방식으로 가능하다.

### 7.5 Cosmos Policy — 행동·미래 관측·가치를 영상 latent 형식에 넣기

**원문:** Kim et al., *Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning* (2026-01). [논문](https://arxiv.org/abs/2601.16163), [공식 프로젝트](https://research.nvidia.com/labs/dir/cosmos-policy/)

**설계의 핵심은 latent frame injection이다.** 사전학습 Cosmos-Predict2의 latent sequence에 proprioception, action chunk, 미래 관측, value를 호환되는 latent frame 형식으로 삽입해 후학습한다.

~~~text
현재 로봇 상태·여러 카메라·언어
  → 사전학습 영상 모델의 latent sequence
  → 행동 chunk / 미래 로봇 상태·이미지 / 미래 value 생성
  → 직접 정책으로 실행하거나, 후보 결과·가치를 비교해 계획
~~~

**특히 읽을 구별:** 기본 정책은 action chunk 끝 시점의 미래 관측을 다루며, 모든 중간 프레임을 촘촘히 생성하는 미래 영상 rollout과는 다르다. policy rollout 자료를 추가로 활용한 계획 개선도, 초기 시연 후학습 단계와 나누어 본다. [방법 절](https://arxiv.org/html/2601.16163v1#S4)

**비교 포인트:** 모델 크기만이 아니라, 어떤 modality를 latent에 넣었는지와 추론 때 policy·forward model·value를 어떤 조건화로 호출하는지 확인한다.

### 7.6 DreamZero — 영상·행동 공동 생성과 실제 폐루프 제어

**원문:** Ye et al., *World Action Models are Zero-shot Policies* (2026-02). [논문](https://arxiv.org/abs/2602.15922), [공식 프로젝트](https://dreamzero0.github.io/), [공식 코드](https://github.com/dreamzero0/dreamzero)

**학습 구조:** 사전학습 Wan 계열의 14B 영상 diffusion backbone에 현재·과거 시각 정보, 언어, proprioception을 조건으로 넣고 미래 영상과 행동을 공동 예측한다. autoregressive chunk 구조와 flow matching을 사용한다. 개념적 분해는 “미래 영상 예측 × 역동역학”이지만 별도 두 모델을 순차 실행하는 구현과 구별한다. [모델 구조](https://arxiv.org/html/2602.15922v1#S3.SS1)

**실행 구조:** 관측 피드백과 비동기 실행을 사용하고, 실제 관측으로 cache의 시각 문맥을 갱신한다. 생성된 미래만 계속 재입력하는 오픈루프와 다르다. 영상·행동 생성 schedule 조정, caching, 병렬화 등 속도 설계를 함께 읽는다. [공식 프로젝트](https://dreamzero0.github.io/)

**성과를 읽는 법:** 저자들은 새 작업·환경 일반화와 영상만으로 얻는 cross-embodiment transfer를 보고한다. 그러나 “zero-shot”은 모든 선행학습·로봇 데이터가 없다는 뜻이 아니다. 새 task, 새 environment, 새 embodiment를 따로 확인하고, 새로운 몸체로의 few-shot adaptation 결과도 구별한다. 실제 실행의 하드웨어·지연 조건은 [공식 코드의 배포 설명](https://github.com/dreamzero0/dreamzero)과 함께 읽는다.

**논문을 읽으며 답할 질문:**

1. 어떤 video prior를 가져오고, 어떤 robot data로 행동을 연결했는가?
2. 시각 문맥과 행동 chunk의 시간 해상도는 어떻게 정렬하는가?
3. 미래 영상 생성이 좋은 행동에 기여했다는 것을 어떤 ablation으로 확인하는가?
4. 익숙하지 않은 동작과 새로운 물체·배경에 대한 일반화를 분리했는가?
5. 실제 관측을 언제 반영하며 계산이 늦으면 어떤 행동을 실행하는가?

### 7.7 네 접근을 한 화면에서 비교

| 접근 | 핵심 흐름 | 미래의 표현 | 행동 선택 때 중요한 요소 |
|---|---|---|---|
| UniPi | 영상 계획 → 역동역학 | 미래 영상 | 영상 실현 가능성, IDM |
| V-JEPA 2-AC | 후보 행동 → 미래 표현 → 목표 거리 | embedding | CEM/MPC, 목표 이미지 |
| Cosmos Policy | 정책·전이·가치 공동 학습 | latent frame으로 표현한 미래 관측·상태 | 직접 정책 또는 후보와 가치 비교 |
| DreamZero | 미래 영상·행동 공동 생성 | 영상 latent와 행동 | autoregressive 문맥, 관측 피드백, 빠른 생성 |

표는 대표적인 사용법을 요약한다. 지원 가능한 모든 조건화나 파생 구현을 나열한 것은 아니다.

---

<a id="recent"></a>
## 8. 2026년 최신 쟁점과 추가 읽기

초기의 latent world model과 달리 최근 연구의 관심은 **대규모 영상 사전학습 지식이 실행 가능한 action과 어떻게 연결되는가**로 확장되었다. 아래는 학습할 때 대비할 대표적인 접근이다.

| 논문·모델 | 출발점 | 관찰할 설계 선택 | 읽을 자료 |
|---|---|---|---|
| Unified World Models (2025) | Video + action diffusion | Modality별 독립 노이즈·조건부 생성 | [논문](https://arxiv.org/abs/2504.02792) |
| Cosmos Policy (2026) | Pretrained video model | Action, future observation, value의 latent 표현 | [논문](https://arxiv.org/abs/2601.16163) |
| DreamZero (2026) | Video diffusion backbone | 미래 영상·행동의 공동 생성과 로봇 제어 | [논문](https://arxiv.org/abs/2602.15922) |
| DUST (ICML 2026) | Dual-stream video-action diffusion | 서로 다른 modality의 상호작용과 sampling 비용 | [논문](https://proceedings.mlr.press/v306/won26a.html) |
| Motus (CVPR 2026) | Latent action + world model | Video, action, understanding을 함께 모델링 | [논문](https://openaccess.thecvf.com/content/CVPR2026/html/Bi_Motus_A_Unified_Latent_Action_World_Model_CVPR_2026_paper.html) |
| OpenWAM (2026-10) | Wan 기반 composable framework | Video-first, action-first, joint 등 구조 비교 | [논문](https://arxiv.org/abs/2610.07922) |

**주의:** 여기서 world-action model은 공식적으로 고정된 단 하나의 아키텍처 이름이 아니다. 논문마다 forward dynamics, inverse dynamics, policy, joint generation 가운데 어떤 기능을 함께 학습·실행하는지 확인한다.

### 8.1 열려 있는 다섯 가지 연구 문제

1. **Prediction과 control의 불일치:** 영상 prediction loss가 낮아도 로봇의 task success가 높지 않을 수 있다.
2. **Action 없는 영상에서 행동 얻기:** latent motion과 실제 actuator command 사이의 alignment가 필요하다.
3. **Counterfactual consistency:** 같은 상황에서 다른 행동을 적용했을 때 올바르게 다른 미래를 예측해야 한다.
4. **Long-horizon memory:** Sliding-window와 무제한 KV 유지 사이에서 기억 비용·오차의 균형이 필요하다.
5. **Closed-loop latency:** World prediction, action generation, planning, observation refresh의 전체 시간을 평가해야 한다.

<a id="evaluation"></a>
## 9. 대표 방법 비교와 논문 평가법

### 9.1 논문마다 체크할 축

| 평가 축 | 반드시 확인할 질문 |
|---|---|
| 데이터 | 행동 라벨이 있는가? Offline demonstration인가, 실제 online interaction인가? |
| Observation/state | RGB, VAE latent, representation embedding, object state 중 무엇을 쓰는가? |
| Dynamics | Deterministic, stochastic, autoregressive, diffusion/flow 중 무엇인가? |
| Policy | MPC/CEM, learned actor, inverse dynamics, joint action decoding 중 무엇인가? |
| 학습 | Reconstruction, future prediction, return optimization, imitation 중 주목적은? |
| 평가 | Pixel fidelity, prediction error, planning return, task success, latency 중 무엇을 측정하는가? |
| 전이 | 새로운 task, 환경, embodiment 중 어떤 일반화를 보고하는가? |
| Memory | History 길이 증가에 따라 KV memory와 rollout error가 어떻게 바뀌는가? |

### 9.2 예상 성능과 실제 성능을 분리하자

실제 환경에서의 return

```math
J_P(\pi)=\mathbb E_{P,\pi}\sum_{t\ge0}\gamma^t r_t
```

과 학습 모델에서의 return

```math
J_{\hat P}(\pi)=\mathbb E_{\hat P,\pi}\sum_{t\ge0}\gamma^t r_t
```

은 서로 다른 양이다. $J_{\hat P}$가 높아도 $J_P$가 높다고 결론 내릴 수 없다. 가상 rollout 수를 늘릴수록 policy가 모델 오류를 찾아 이용할 가능성도 있다.

<a id="study"></a>
## 10. 권장 공부 순서와 작은 실험

처음 한 바퀴는 **World Models → PlaNet → Dreamer → DreamerV3**에 집중한다. 각 논문에서 'world model이 무엇을 예측하는가', 'policy가 어떻게 action을 결정하는가', '어떤 데이터를 실제 환경에서 얻는가'를 3열 표로 기록한다.

| 단계 | 읽기 | 구현 또는 검증 |
|---|---|---|
| 1 | World Models, 2018 | VAE-only controller vs recurrent-memory controller 구조 비교 |
| 2 | PlaNet, 2019 | 짧은 latent rollout과 CEM 후보 수·horizon 변화 실험 |
| 3 | Dreamer, 2020 | $n$-step / $\lambda$-return 계산과 imagination actor–critic 확인 |
| 4 | DreamerV2/V3 | KL balancing, free bits, symlog, two-hot의 코드 위치 추적 |
| 5 | Genie, Diffusion Forcing | Action conditioning과 causal mask를 영상 예측으로 확장 |
| 6 | UWM, DreamZero, OpenWAM | Video/action denoising와 inference latency 및 KV history 비교 |

**작은 실험 아이디어:** 실제 환경 호출 수를 동일하게 고정하고 (i) MPC, (ii) imagination-trained actor의 decision latency와 environment return을 비교하자. 단, 구현·모델 용량·계산 예산까지 공정하게 맞추지 않으면 알고리즘 성능의 직접 비교가 아니다.

<a id="faq"></a>
## 11. 자주 헷갈리는 질문과 이해도 점검

**Q0. $s_t$, $o_t$, $x_t$는 어떻게 구분하는가?**  
$s_t$는 실제 환경 상태, $o_t$는 실제 센서 관측, $x_t$는 관측과 행동 이력으로부터 추론한 모델 내부의 잠재 상태다. $x_t$가 $s_t$와 같다는 보장은 없다. 상세한 표·수식·예시는 [3.2절](#math)에 정리했다.

**Q0-1. 3.4절의 $g$는 무엇인가?**  
$g$는 goal을 나타내는 조건으로, 자연어 지시·목표 이미지·목표 상태 등으로 표현할 수 있다. 관측·행동 이력 $h_t$는 현재 세계에 대한 정보를 제공하고 $g$는 원하는 미래를 지정한다. $g$는 reward $r_t$와 다른 개념이며, 모든 WAM에 필수인 입력은 아니다.

**Q1. 미래 영상을 잘 만들면 바로 world model인가?**  
넓은 영상 생성 문맥에서는 그렇게 부를 수도 있다. 그러나 의사결정용 world model은 대개 **행동에 대한 반응**을 다루므로 action controllability를 따로 검증해야 한다.

**Q2. RSSM의 prior와 posterior는 왜 둘 다 필요한가?**  
Prior $p_\theta(z_t\mid u_t)$는 아직 $o_t$를 보지 않고 latent를 예측한다. Approximate posterior $q_\phi(z_t\mid u_t,o_t)$는 실제 관측을 반영한 현재 latent의 추론 분포다. 미래 imagination에는 새 관측이 없으므로 prior를 사용하며, KL 학습으로 양쪽을 연결한다. [4.1절](#mechanisms) 참고.

**Q2-1. $u_t$가 확정적 정보이고 $z_t$가 100% 무작위 정보인가?**  
아니다. Deterministic은 **입력이 주어졌을 때 함수 출력이 고정된다**는 뜻이고, stochastic은 **조건부 분포에서 값을 sampling한다**는 뜻이다. $u_t$에도 과거 sampling에서 비롯된 불확실성이 전파될 수 있고, $z_t$도 주어진 $u_t,o_t$로 상당 부분 예측 가능하다. 둘이 정확한 물리량과 무작위 잡음으로 의미상 분리된다는 보장은 없다.

**Q3. Dreamer에서는 왜 매번 CEM을 수행하지 않는가?**  
World model의 imagined trajectory에서 actor를 미리 학습했기 때문이다. 실행 시에는 policy network가 행동을 제안한다.

**Q4. Dreamer의 actor gradient와 DreamerV3의 actor gradient는 같은가?**  
아니다. Original Dreamer는 differentiable dynamics의 경로 미분을 강조했고, DreamerV3는 REINFORCE 형태의 actor update를 사용한다.

**Q5. Video diffusion과 WAM의 차이는?**  
WAM은 세계 예측과 실행 가능한 행동 선택의 연결을 핵심으로 본다. Diffusion은 이를 구현하는 여러 수단 중 하나다.

**Q6. WAM이면 반드시 로봇 reward와 RL이 필요한가?**  
아니다. Imitation learning과 video-action joint generation으로도 policy를 학습할 수 있다. Reward 기반 planning은 추가 가능한 선택지다.

<a id="glossary"></a>
## 12. 용어 사전

| 용어 | 이 노트에서의 의미 |
|---|---|
| True state $s_t$ | 이 문서에서 실제 환경의 상태에 예약한 표기; 논문에 따라 내부 상태에 쓰기도 함 |
| Observation $o_t$ | 실제 카메라/센서에서 측정한 관측 |
| Model state $x_t$ | 관측·행동 이력으로부터 모델이 추론한 학습된 잠재 상태 |
| Belief $b_t$ | 실제 상태에 대한 조건부 확률분포 |
| Goal $g$ | 원하는 미래를 지정하는 조건(자연어 지시·목표 영상/이미지·목표 상태 등). Reward $r_t$와 구별 |
| Stochastic latent $z_t$ | RSSM 내부 상태 $x_t=(u_t,z_t)$에서 확률적인 변수 |
| Recurrent state $u_t$ | RSSM 내부의 결정론적 memory (원문에서 $h_t$를 사용하기도 함) |
| RSSM | Recurrent State-Space Model: 결정론적 기억 + 확률적 latent |
| Prior $p_\theta(z_t\mid u_t)$ / approximate posterior $q_\phi(z_t\mid u_t,o_t)$ | 각각 현재 관측을 보기 전의 latent 예측 분포 / 현재 관측을 반영한 latent 추론 분포 |
| Deterministic / stochastic | 조건부 입력을 고정하면 출력이 정해지는 계산 / 조건부 분포로 표현하고 sampling 가능한 변수; 확실성 vs 순수 난수의 구분이 아님 |
| World model | 세계 전이·관측·보상 등의 예측 모델 |
| MPC | 일정 horizon 계획 → 일부 실행 → 새 관측으로 재계획 |
| CEM | 높은 점수의 후보들로 sampling distribution을 반복 갱신하는 최적화 |
| Latent imagination | 학습한 dynamics에서 가상 미래 상태·보상을 rollout |
| Actor / Critic | 행동 분포를 학습하는 모델 / 기대 return을 예측하는 모델 |
| Dreamer $\lambda$-return | 여러 bootstrap 길이를 가중 결합한 return |
| Model exploitation | 잘못된 world model에서 높아 보이는 보상을 policy가 악용 |
| Action-conditioned video | 과거 관측과 action을 조건으로 미래 영상을 생성 |
| Inverse dynamics | 상태 변화에서 이를 유도한 행동을 추론 |
| WAM | 세계 예측과 행동 생성을 결합하는 연구 패러다임 |
| Closed-loop | 새 관측을 계속 반영해 행동을 수정하는 실행 |
| KV cache | 이전 token의 key/value를 유지해 attention 재계산 절약 |

<a id="notebook"></a>
## 13. 후속 질문·수정 기록

이 문서는 **학습 중 계속 확장하는 노트**다. 앞으로 **용어·표기·정의·수식의 의미를 분명히 하기 위한 사용자의 질문**은 대화에서 설명한 다음 문서의 해당 본문·FAQ·용어집에 반영하고, 변경 기록에도 남기는 것을 기본 원칙으로 한다.

### 13.1 설명 보강 요청 템플릿

~~~md
#### [날짜] 질문: (예: RSSM prior와 posterior의 KL)

- **위치:** 5.3의 RSSM 학습 목적
- **막힌 지점:** KL을 왜 이 방향으로 쓰는가?
- **현재 이해:** ...
- **보충 설명:** (대화 이후 여기에 수식/예제 추가)
- **원문 근거:** 논문 section / figure / equation
~~~

### 13.2 앞으로 깊게 보완할 후보

- [ ] PlaNet의 ELBO와 latent overshooting을 원문 조건부 분포로 처음부터 유도
- [ ] Original Dreamer의 $\lambda$-return과 analytic actor-gradient 전체 유도
- [ ] DreamerV2 vs DreamerV3의 actor update와 KL balancing 코드 비교
- [ ] VAE latent와 JEPA embedding을 동일한 planning task에서 비교
- [ ] World-model rollout error와 KV/history compression의 영향 정량화

### 13.3 변경 기록

| 날짜 | 변경 내용 |
|---|---|
| 2026-10-13 | 4절 도입부에 학습 데이터와 예측 대상, 공 밀기 예시, RSSM → MPC → Dreamer → 생성·실시간 제어 흐름을 간결하게 추가 |
| 2026-10-13 | 4.1절의 prior $p_\theta$ / approximate posterior $q_\phi$의 의미와 deterministic $u_t$ / stochastic $z_t$의 차이를 수식, 예시, 학습·rollout 순서로 보충; FAQ·용어집 업데이트 |
| 2026-10-13 | 3.3~3.4절의 목표 $g$를 구체적으로 정의하고 reward와의 차이, 가능한 목표 조건의 형태, 선택적 입력이라는 점을 명시; FAQ 및 용어집 보강 |
| 2026-10-13 | 실제 상태 $s_t$, 관측 $o_t$, 모델 내부 상태 $x_t$의 차이 명확화. RSSM 관련 PlaNet·Dreamer 수식의 notation 통일 및 FAQ·용어집·질문 반영 원칙 추가 |
| 2026-10-13 | 개요에 World Models / PlaNet / Dreamer / DreamerV3 상세 해설 통합; 목차 후반부 학습·질문 섹션 추가 |

<a id="references"></a>
## 14. 원문과 공식 자료 모음

### 초기 네 편 (우선 읽기)
1. Ha & Schmidhuber (2018), [World Models](https://arxiv.org/abs/1803.10122) · [프로젝트](https://worldmodels.github.io/).
2. Hafner et al. (2019), [Learning Latent Dynamics for Planning from Pixels (PlaNet)](https://arxiv.org/abs/1811.04551) · [ICML](https://proceedings.mlr.press/v97/hafner19a.html).
3. Hafner et al. (2020), [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603).
4. Hafner et al. (2023), [Mastering Diverse Domains through World Models (DreamerV3)](https://arxiv.org/abs/2301.04104) · [구현](https://github.com/danijar/dreamerv3).

### 연결 논문
- Hafner et al., [DreamerV2: Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193).
- Schrittwieser et al., [MuZero](https://arxiv.org/abs/1911.08265).
- Hansen et al., [TD-MPC2](https://arxiv.org/abs/2310.16828).
- Bruce et al., [Genie](https://arxiv.org/abs/2402.15391).
- Chen et al., [Diffusion Forcing](https://arxiv.org/abs/2407.01392).
- Chi et al., [Diffusion Policy](https://arxiv.org/abs/2303.04137).
- Black et al., [$\pi_0$: Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164).
- Ye et al., [DreamZero](https://arxiv.org/abs/2602.15922).
- [OpenWAM](https://arxiv.org/abs/2610.07922).

> **문서 작성 원칙:** 논문이 입증한 결과, 이를 이해하기 위한 단순화한 수식, 그리고 향후 연구를 위한 해석을 구분해 둔다. 특히 최신 프리프린트의 실험적 일반화 주장은 보고된 환경·데이터 범위 안에서 해석한다.
