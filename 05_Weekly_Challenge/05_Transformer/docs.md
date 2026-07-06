# Transformer

## 단계별 개념 이해

**표 1. 근본 단위 -> Transformer 확장 경로**
| 단계 | 구현 대상 | 이 단계에서 다루는 근본 개념 | 다음 단계로 넘어가는 이유(한계) |
| --- | --- | --- | --- |
| 1 | 단일 뉴런 (single neuron) — `y = wx + b` | weight (가중치), bias (편향), forward pass (순전파) | 비선형 문제를 풀 수 없음 (예: XOR) |
| 2 | activation function (활성화 함수) 추가 — sigmoid, ReLU | non-linearity (비선형성) | 뉴런 하나로는 표현력(representational capacity) 부족 |
| 3 | MLP (Multi-Layer Perceptron, 다층 퍼셉트론) — `nn.Linear` 여러 층 | layer stacking (층 쌓기), backpropagation (역전파)의 chain rule (연쇄 법칙) | 입력 순서·문맥을 반영하지 못함 (각 입력을 독립적으로 처리) |
| 4 | 수동 loss/backward 학습 루프 (PyTorch autograd 없이 직접 gradient 계산 1회) | gradient descent (경사 하강법)의 실제 계산 원리 | 매번 손으로 미분하는 건 비효율적 → autograd로 전환 |
| 5 | RNN cell 최소 구현 (hidden state 1개 유지) | sequential processing (순차 처리), hidden state (은닉 상태) | 장거리 의존성(long-range dependency) 약화, 병렬화 불가 — 지난 턴에서 다룬 한계 |
| 6 | attention score 계산만 (Q, K만 사용, V·마스킹 없이 유사도 행렬만) | dot-product similarity (내적 기반 유사도) | 유사도 점수만으로는 실제 정보 전달이 안 됨 |
| 7 | scaled dot-product attention (Q, K, V 전부, `scaled_dot_product_attention.py` 수준) | scaling (스케일링), softmax weighted sum (가중합) | head가 하나뿐이라 한 가지 관점의 관계만 포착 |
| 8 | multi-head attention (`multi_head_attention.py` 수준) | head 분할·병합 | attention만으로는 순서 정보 없음 + 깊은 층에서 학습 불안정 |
| 9 | positional encoding + residual + LayerNorm 추가 (`decoder_layer.py` 수준) | 순서 주입, 학습 안정화 | 단일 층으로는 표현력 한계 |
| 10 | 층 쌓기 + 전체 Transformer (`transformer_model.py` 수준, 지금 가진 코드) | 지금까지 요소 전부 조합 | (지금 이 시점) |
