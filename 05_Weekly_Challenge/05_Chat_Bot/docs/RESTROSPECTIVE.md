# 세부 과제 5 - 한국어 챗봇 만들기

## 트러블 슈팅 1 - 모델 아키텍처 선택 (Encoder-Decoder vs Decoder-Only vs Encoder-Only)

### 문제 상황

- 기존에 배운 Transformer 구조는 Encoder + Decoder가 결합된 형태
- 그러나 GPT, LLaMA 등 생성형 모델은 대부분 Decoder-Only 구조를 사용함
- 챗봇 과제의 특성(입력 문장을 받아 이어서 생성하는 자동회귀 생성 모델)에 맞는 구조를 선택할 필요

### 고려한 옵션

| 구조              | 주요 용도                                   | 우리 과제 적합도 | 선택 여부 |
| ----------------- | ------------------------------------------- | ---------------- | --------- |
| Encoder + Decoder | 번역, 요약, Sequence-to-Sequence            | 보통             | 비선택    |
| Encoder-only      | 문장 이해, 분류, 검색 (BERT 스타일)         | 낮음             | 비선택    |
| **Decoder-only**  | 텍스트 생성, 챗봇, 언어 모델링 (GPT 스타일) | **높음**         | **선택**  |

### 결정 및 이유

- 최종 결정: Decoder-Only Transformer
- 선택 이유:
  - Decoder-Only 구조는 Causal Mask를 통해 자동회귀 생성을 자연스럽게 지원함.
  - 과제의 목적이 "문장을 입력받아 자연스럽게 이어서 생성"하는 것이기 때문에 Decoder-Only 구조가 적합

## 트러블 슈팅 2 - 구현의 깊이 수준 선택

### 문제 상황

- "가급적 PyTorch를 사용하지 않고 직접 만들어보는 것"이 해당 과제의 핵심
- 그러나 "직접 만들기"의 범위가 모호함
  - `nn.Transformer`, `nn.MultiheadAttention` 같은 "고수준 모듈"까지 사용하지 말아야 하는가?
  - `nn.Linear`, `nn.LayerNorm` 같은 기본 모듈까지 직접 구현해야 하는가?
  - Attention 메커니즘만 직접 구현하면 되는가?

### 고려한 옵션

| 옵션 | 내용                                                            | 장점                                | 단점                                                    | 평가                 |
| ---- | --------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------- | -------------------- |
| A    | `nn.Transformer`, `nn.MultiheadAttention` 모두 사용             | 가장 빠르게 구현 가능               | 학습 효과 낮음, 강사님 의도와 거리 있음                 | 비추천               |
| B ✅ | 기본 모듈(`nn.Linear` 등)은 사용하고, **Attention만 직접 구현** | 학습 효과와 시간 효율의 균형이 좋음 | `nn.MultiheadAttention`을 사용하지 않으므로 구현량 증가 | **선택**             |
| C    | `nn.Linear`까지 직접 구현                                       | 학습 효과 가장 높음                 | 시간 소모가 매우 큼                                     | 시간 부족으로 비추천 |
| D    | LSTM으로 먼저 전체 파이프라인을 만들고 Transformer로 교체       | 파이프라인 완성 속도가 빠름         | Transformer 직접 구현 경험이 약해짐                     | 비추천               |

### 결정 및 이유

- 최종 결정: 옵션 B
  - `nn.Linear`, `nn.LayerNorm`, `nn.Dropout` 등 기본 모듈은 사용하고,
  - **Attention 메커니즘(Scaled Dot-Product Attention, Multi-Head Attention)은 직접 구현**하는 방향으로 진행.

- 선택 이유
  - 알렉스가 강조한 "직접 만들기"의 핵심은 **고수준 완성형 모듈(`nn.MultiheadAttention`, `nn.Transformer`)을 블랙박스로 사용하지 않는 것**이라고 판단.
  - `nn.Linear`와 같은 단순한 기본 모듈까지 직접 구현하는 것은 학습 효과 대비 시간 비용이 크다고 판단.
  - 제한된 시간 내에 **동작하는 결과물**을 만들면서도, Attention 메커니즘을 직접 구현하는 학습 효과를 어느 정도 확보할 수 있는 가장 현실적인 선택이라고 판단.

- 하위 결정: `nn.Linear` 직접 구현 여부
  - `nn.Linear`와 같은 기본 모듈까지 직접 구현할지 여부를 별도로 고민
  - 다만, 학습 효과 대비 시간 비용이 크다고 판단
  - 최종적으로 `nn.Linear`는 사용하고, Attention 구현에 집중하기로 결정

### 인사이트

- “직접 만들기”의 범위를 무조건 극단적으로 해석하기보다는, **학습 목적과 시간 제약을 함께 고려**하여 현실적인 수준을 정하는 것이 실무에서도 중요할 것으로 예상

## 트러블 슈팅 3 - 개념 이해와 코드 구현 사이의 간극

### 문제 상황

- Transformer의 주요 개념은 어느 정도 이해했으나, 실제 코드로 구현할 시 막막함을 느낌
- 특히 Scaled Dot-Product Attention의 동작 원리를 코드 수준까지 연결하기 위해 노력

### 결정

- 최종 결정:
  - FastAPI로 바로 배포할 수 있도록 전체 구조를 설계한 후, Transformer를 적용하기 전 Dummy 데이터 우선 구현
  - 본격적으로 Transformer 구현하기 에 앞서 바로 전체 코드를 작성하기 보다 Scaled Dot-Product Attention부터 단계적으로 구현하기로 함
  - 개념과 코드를 최대한 일대일 대응으로 명확히 하면서 점진적으로 구현 범위를 넓혀나가기로 결정

### 진행 계획

1. `ScaledDotProductAttention` 클래스 구현
2. `MultiHeadAttention` 클래스 구현
3. `DecoderLayer` 구현
4. `TransformerDecoder` 구현
5. 전체 모델 조립 및 학습/생성 파이프라인 연결

### 인사이트

- 개념 이해와 코드 구현 사이에는 상당한 간극이 존재
- 이를 메우기 위해서는 **작은 단위로 점진적으로 구현하는 것이 효과적**임을 배움

## 트러블 슈팅 4 - Causal Mask의 차원에 따른 코드 수정 위치 정립 (Single Head vs Multi-Head)

### 문제 상황

- `ScaledDotProductAttention` 테스트 중 Causal Mask를 적용했을 때, mask 차원에 따라 결과 shape이 깨지는 현상 발생
- 처음 AI가 제공한 테스트 코드에서는 테스트용 mask가 4차원으로 생성했으나, `scores`가 3차원이라 broadcasting이 실패하면서 shape이 비정상적으로 변형됨
- 이 과정에서 **"테스트 코드를 수정해야 하는지, 아니면 `ScaledDotProductAttention` 내부 코드를 수정해야 하는지"** 코드 수정 위치에 따른 "책임 분리 결정"에 대한 고민

### 원인 분석

- `ScaledDotProductAttention`의 `scores`는 `(batch, seq, seq)` 형태의 **3차원** 텐서
- Multi-Head Attention으로 확장하면 `scores`가 `(batch, heads, seq, seq)` 형태의 **4차원**이 됨
- 따라서 mask 차원도 단계에 따라 다르게 구성해야 함
- 차원 개수가 맞지 않으면 `masked_fill`에서 broadcasting이 깨져 shape이 비정상적으로 변형

### 결정 및 대응

- 최종 결정:
  - 테스트 코드를 수정하여 mask를 3차원으로 맞추는 방향으로 진행
  - `ScaledDotProductAttention` 내부에서 mask 차원을 자동으로 맞추는 로직은 추가하지 않음

### 결정 이유

- 현재 `ScaledDotProductAttention` 단계는 Mask의 결과가 **3차원** `(batch, seq, seq)` 또는 `(1, seq, seq)`으로 구성되는 것은 정상
- 이는 **현재 단계에서 테스트의 범주**이며 추후 Multi-Head 까지 확장하여 고려하면, 테스트 코드는 3차원, n차원 등 어떤 차원이든 동일한 결과가 나와야 하는 것도 설득력있는 설계 방법으로 고려할 수 있음.(예시: `shape: (batch_size, n_den, ...)`)
- 다만 지금은 Single Head이므로 테스트의 범주를 고려했을 때, `ScaledDotProductAttention` 내부에서 mask 차원을 자동으로 맞추는 로직은 과도한 일반화로 판단하여 현재는 추가하지 않음
- 대신 해당 설계 방법은 추후 확장 과정에서 의미있는 설계 아이디어이므로, `test_attention.py`에 **TODO 주석을 추가**하여 Multi-Head 구현 시 mask 차원 변경이 필요함을 명시

### 인사이트

- Attention 구현 시 mask 차원 관리가 매우 중요함
- mask 차원이 맞지 않을 때, **어디를 수정할 것인가**에 대한 판단 기준을 세우는 것을 고민
- Single Head와 Multi-Head 단계에서 mask 차원이 달라질 수 있으므로, 단계별로 mask 구성을 명확히 구분해야 함
- 차원 불일치로 인한 에러는 디버깅이 어렵기 때문에, 추후 작업으로 표시하고, 현재는 테스트 코드를 수정하는 단계로 변경

## 트러블 슈팅 5 - nn.Linear의 역할 이해 (Q, K, V Projection)

### 문제 상황

- `MultiHeadAttention` 구현 과정에서 `self.q_linear = nn.Linear(d_model, d_model)`과 같이 `nn.Linear`를 사용하는 코드를 작성함
- `nn.Linear`가 내부적으로 어떤 동작을 하는지, 왜 `in_features`와 `out_features`를 지정해야 하는지, 그리고 `forward` 시점에 실제로 어떤 일이 일어나는지에 대한 이해가 부족했음
- 특히 "입력 텐서의 마지막 차원 크기와 `in_features`가 일치해야 하는 이유"와 "`d_model, d_model`처럼 동일한 값을 넣는 의미"에 대해 혼란을 겪음

### 원인 분석

- `nn.Linear(in_features, out_features)`는 레이어를 **생성하는 시점**에 내부적으로 `(out_features, in_features)` 크기의 가중치 행렬 W와 bias를 생성함
- 이 가중치 행렬의 크기를 결정하기 위해 `in_features`와 `out_features`를 미리 지정해야 함
- 실제 선형 변환(`input @ W.T + bias`)은 `forward` 단계에서 이루어지며, 이때 입력 텐서의 마지막 차원 크기가 `in_features`와 정확히 일치해야 행렬 곱셈이 정상적으로 수행됨
- `d_model, d_model`처럼 입력과 출력 차원이 같을 경우에도, 단순히 차원을 유지하는 것이 아니라 **입력 정보를 특정 관점(Q, K, V)으로 재해석**하는 의미가 있음

### 결정 및 대응

- `nn.Linear`의 동작 원리를 다음과 같이 정리함:
  - 레이어 생성 시(`__init__`): 가중치 행렬 W 생성 및 변환 준비
  - 실제 사용 시(`forward`): 입력 텐서의 마지막 차원을 기준으로 선형 변환 수행
- `in_features`와 입력 텐서의 마지막 차원 크기가 일치해야 한다는 점을 명확히 인지
- `d_model, d_model`처럼 동일한 값을 사용하는 경우에도 **정보의 재해석**이라는 의미가 있음을 이해

### 인사이트

- `nn.Linear`는 단순히 차원을 늘리거나 줄이는 도구가 아니라, **학습 가능한 선형 변환**을 수행하는 레이어임
- 레이어를 생성하는 시점과 실제 데이터를 넣는 시점을 명확히 구분해야 함
- `in_features`는 입력 텐서의 마지막 차원 크기와 반드시 일치해야 하며, 이는 행렬 곱셈의 차원 조건 때문임
- `d_model, d_model`처럼 동일한 값을 사용하는 것은 차원을 유지하면서도 입력 정보를 특정 목적(Q, K, V)에 맞게 재해석하기 위한 설계임을 이해함

## 트러블 슈팅 6 - 폴더 구조에 따른 Import 에러 해결 (sys.path 수동 추가 방식)

### 문제 상황

- 프로젝트를 `src` 레이아웃으로 재구성한 후, `tests/` 폴더에서 모델 코드를 import하려고 했을 때 `ModuleNotFoundError: No module named 'src'` 에러가 발생함
- 예시 에러

  ```bash
  from src.model.decoder_layer import DecoderLayer
  ModuleNotFoundError: No module named 'src'
  ```

### 원인 분석

- Python은 기본적으로 sys.path에 등록된 경로만 모듈을 검색함
- `src` 폴더는 Python이 자동으로 인식하는 경로가 아니기 때문에, `from src.xxx import` 형태의 import가 실패함
- `src`를 패키지 루트로 인식할 필요

### 결정 및 대응

- 가장 기본적인 방법으로 테스트 파일 상단에 프로젝트 루트 경로를 동적으로 추가하는 방식 적용
- 코드 예시

  ```python
  import sys
  import os

  # 프로젝트 루트 경로 추가하여 임포트 에러 해결
  __TEST_FOLDER_PATH__ = os.path.dirname(__file__)
  joined_root_path = os.path.join(__TEST_FOLDER_PATH__, '..')
  absoluted_joined_root_path = os.path.abspath(joined_root_path)
  sys.path.append(absoluted_joined_root_path)
  ```

  - 이 코드를 통해 현재 테스트 파일의 위치를 기준으로 프로젝트 루트 경로를 계산한 후, sys.path에 추가하여 import가 가능하도록 함

### 인사이트

- sys.path.append 방식은 빠르게 문제를 해결할 수 있지만, 테스트 파일이 많아질수록 모든 파일에 동일한 코드를 중복해서 작성해야 한다는 단점이 있음
- 장기적으로는 유지보수성과 확장성이 떨어지는 방식이라는 점을 인지함
