# Transformer

## 단계별 개념 이해

**표 1. 근본 단위 -> Transformer 확장 경로**
| 단계 | 구현 대상 | 이 단계에서 다루는 근본 개념 | 다음 단계로 넘어가는 이유(한계) |
| --- | --- | --- | --- |
| 1 | 단일 뉴런 (single neuron) — `y = wx + b` | weight (가중치), bias (편향), forward pass (순전파) | 비선형 문제를 풀 수 없음 (예: XOR) |
| 2 | activation function (활성화 함수) 추가 — sigmoid, ReLU | non-linearity (비선형성) | 뉴런 하나로는 표현력(representational capacity) 부족 |
| 3 | MLP (Multi-Layer Perceptron, 다층 퍼셉트론) | layer stacking (층 쌓기), backpropagation (역전파)의 chain rule (연쇄 법칙) | 파라미터가 늘어날수록 chain rule을 손으로 유도하는 게 비효율적 |
| 4 | 연산자 오버로딩(operator overloading) 기반 자동 미분 — 자체 Value 엔진, PyTorch 비교 검증 | automatic differentiation (자동 미분), computational graph (계산 그래프) | 스칼라 단위 연산만 가능, 벡터·행렬 단위 배치 연산(batch operation) 불가 |
| 5 | 스칼라 → 벡터·행렬 단위 배치 연산 확장 (순수 Python + NumPy 두 경로) | vectorization (벡터화), batch operation (배치 연산) | 벡터 연산만으로는 아직 순서(sequence)가 있는 데이터를 다루는 구조가 없음. GPU 병렬화는 CUDA 드라이버·커널 연동이 필요해 이 단계에서도 미해결 |
| 6 | RNN cell 최소 구현 (hidden state 1개 유지) | sequential processing (순차 처리), hidden state (은닉 상태) | 장거리 의존성(long-range dependency) 약화, 병렬화 불가 |
| 7 | attention score 계산만 (Q, K만 사용, V·마스킹 없이 유사도 행렬만) | dot-product similarity (내적 기반 유사도) | 유사도 점수만으로는 실제 정보 전달이 안 됨 |
| 8 | scaled dot-product attention (Q, K, V 전부, `scaled_dot_product_attention.py` 수준) | scaling (스케일링), softmax weighted sum (가중합) | head가 하나뿐이라 한 가지 관점의 관계만 포착 |
| 9 | multi-head attention (`multi_head_attention.py` 수준) | head 분할·병합 | attention만으로는 순서 정보 없음 + 깊은 층에서 학습 불안정 |
| 10 | positional encoding + residual + LayerNorm 추가 (`decoder_layer.py` 수준) | 순서 주입, 학습 안정화 | 단일 층으로는 표현력 한계 |
| 11 | 층 쌓기 + 전체 Transformer (`transformer_model.py` 수준, 지금 가진 코드) | 지금까지 요소 전부 조합 | (지금 이 시점) |
