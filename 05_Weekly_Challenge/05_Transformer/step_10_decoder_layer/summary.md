# decoder layer (positional encoding + residual + LayerNorm)

## 배경

- 9단계 "결론 및 한계"에서 확정한 대로, attention은 "관련도"는 계산하지만 "순서"는 모릅니다. `Q = X @ Wq` 같은 계산은 `X`의 각 행이 시퀀스의 몇 번째 위치인지 전혀 참조하지 않아서, 입력 순서를 통째로 섞어도 결과가 (행 순서만 바뀔 뿐) 동일하게 나옵니다.
- 여기에 더해, 11단계에서 `DecoderLayer`를 여러 층 쌓게 되면(로드맵상 다음 단계), 층을 거칠 때마다 값의 크기(scale)가 들쭉날쭉해지거나 gradient가 점점 작아지는(vanishing) 문제가 생길 수 있습니다. 이건 층을 하나만 쓸 때는 드러나지 않다가, 층이 깊어지면서 누적되어 나타나는 문제입니다.
- 10단계는 이 두 문제를 각각 별개의 장치로 해결합니다. positional encoding(위치 정보 주입)으로 순서 문제를, residual connection + LayerNorm(학습 안정화)으로 깊은 층 문제를 대비합니다. 세 장치는 서로 독립적인 문제를 풀며, 이번 단계에서 함께 구현하되 개념은 따로 이해할 수 있습니다.

## 동작 원리

### 관련 개념 요약

**positional encoding (위치 인코딩)**

attention은 순서를 모릅니다. 입력을 뒤섞어도 attention score 자체는 (행 순서만 바뀔 뿐) 똑같이 계산됩니다. positional encoding은 위치마다 고유한 패턴(sin/cos 파형)을 만들어서, 입력에 "몇 번째 위치인지"라는 정보를 명시적으로 실어 보냅니다. 짝수 차원은 sin, 홀수 차원은 cos를 써서, 서로 다른 주기의 파형을 섞어 위치마다 겹치지 않는 고유한 패턴을 만듭니다.

**residual connection (잔차 연결)**

층이 깊어지면, backward 시 gradient가 여러 층을 거치며 점점 작아지거나(vanishing) 커지는(exploding) 문제가 생길 수 있습니다. residual connection은 "이 층이 계산한 것"에 "원래 입력"을 그대로 더해서, 입력에서 출력으로 가는 지름길(shortcut)을 하나 더 만들어줍니다. sublayer(attention)가 아무것도 유용한 걸 학습하지 못해도, 최소한 원래 입력은 그대로 보존되어 다음 층으로 전달된다는 게 핵심입니다.

**LayerNorm (층 정규화)**

층이 깊어질수록 각 층을 통과할 때마다 값의 크기가 들쭉날쭉해져서 학습이 불안정해질 수 있습니다. LayerNorm은 각 위치(행)마다, feature 축(열) 방향으로 평균 0, 분산 1이 되도록 다시 맞춰서 이 문제를 완화합니다. 정규화 후에는 `gamma`(스케일), `beta`(이동)라는 학습 가능한 파라미터로 다시 적절한 크기로 조정할 여지를 모델에게 남겨줍니다.

**forward**

```python
# Before — 9단계(multi_head_attention.py)
def forward(self, X: np.ndarray) -> tuple:
    head_outputs = [head.forward(X) for head in self.heads]
    head_O_values = [O for O, A in head_outputs]
    O_concat = NumpyValue.concat(head_O_values, axis=-1)
    O = O_concat @ self.Wo
    return O, head_outputs

# After — 10단계(decoder_layer.py)
def forward(self, X: np.ndarray) -> NumpyValue:
    seq_len = X.shape[0]
    pe = positional_encoding(seq_len, self.d_model)

    X_with_pos_data = X + pe
    X_with_pos = NumpyValue(X_with_pos_data)

    attention_output, _ = self.attention.forward(X_with_pos_data)
    residual_output = residual_connection(X_with_pos, attention_output)
    output = self.norm.forward(residual_output)

    return output
```

- 9단계는 `X`를 그대로 `MultiHeadAttention`에 넣었습니다.
- 10단계는 `X + pe`(위치 정보 주입)를 먼저 거친 뒤에야 attention을 계산합니다. `pe`는 학습되는 파라미터가 아니라, 위치마다 고정된 sin/cos 패턴입니다.
- `residual_connection(X_with_pos, attention_output)`이 9단계에는 없던 신규 단계입니다. attention의 출력에 "위치 정보가 더해진 원본 입력"을 다시 더합니다.
- `self.norm.forward(residual_output)`이 마지막 단계입니다. `LayerNorm`이 각 행마다 평균 0, 분산 1로 맞춘 뒤, `gamma`, `beta`로 다시 스케일을 조정합니다.
- forward 실행 시, NumpyValue 동작 과정
  1. `pe`(positional encoding)는 순수 NumPy 배열 계산이라, `NumpyValue` 연산자 오버로딩을 거치지 않습니다. `X + pe`도 아직 `NumpyValue`로 감싸기 전, 순수 배열 덧셈입니다.
  2. `X_with_pos = NumpyValue(X_with_pos_data)`에서 비로소 `NumpyValue`가 생성되고, 이 시점부터 계산 그래프가 시작됩니다(잎 노드, leaf node).
  3. `self.attention.forward(...)`가 9단계에서 검증한 계산 그래프를 그대로 만듭니다.
  4. (`residual_connection`, 즉 `x + sublayer_output`) `__add__`가 호출되어, `X_with_pos`와 `attention_output`을 자식으로 하는 새 노드가 생성됩니다. `X_with_pos`가 이 시점에 두 번째로 그래프에 등장합니다(한 번은 attention의 입력으로, 한 번은 residual의 피연산자로) — 4단계에서 다룬 "다중 경로 gradient 합산"과 같은 구조입니다.
  5. (`LayerNorm.forward`) `mean`, `centered = x + (-1*mean)`, `variance = (centered*centered).mean(...)`, `std = (variance+eps).sqrt()`, `normalized = centered / std`, `normalized * gamma + beta` 순서로 신규 연산(`mean`, `sqrt`, `__truediv__`)이 연쇄적으로 호출되며 최종 출력이 만들어집니다.

## 구현

- [NumpyValue 클래스 (mean, sqrt, truediv 추가, NumPy)](./value_numpy.py)
- [positional_encoding, residual_connection, LayerNorm](./positional_residual_layernorm.py)
- [DecoderLayer (세 요소 + 9단계 MultiHeadAttention 조합)](./decoder_layer.py)
- [검증 테스트](./test_decoder_layer.py)

## 결과 정리

**표 1. 10단계 검증 결과**
| 검증 항목 | 결과 |
| --- | --- |
| positional encoding이 위치마다 다른 패턴을 만드는가 | 6개 위치 전부 서로 다름 |
| LayerNorm 출력의 행별 평균/분산 | 평균 ≈ 0, 분산 ≈ 1 (정확히 일치) |
| LayerNorm backward vs 수치 미분 | `gamma` 4개 원소 전부 일치 |
| `DecoderLayer` 전체 forward-backward | 출력 모양 `(5,8)`, 정규화 유지, 전체 파라미터에 gradient 전파 확인 |

## 결론 및 한계

- 9단계까지의 두 가지 한계가 이번 단계에서 함께 해결됐습니다. positional encoding으로 "순서를 모른다"는 문제가, 위치마다 서로 다른 고유 패턴(표 1)이 실제로 생성된다는 게 확인됐습니다. residual + LayerNorm으로는 "층이 깊어질 때의 불안정성"에 대비했고, 최종 출력이 항상 평균 0·분산 1로 유지된다는 게 검증됐습니다.
- 다만 지금은 `DecoderLayer` 하나만 만들었을 뿐, 실제로 여러 층을 쌓지는 않았습니다. residual + LayerNorm이 "여러 층을 쌓을 때"의 문제에 대비한 장치인데, 층을 하나만 써서는 이 장치가 실제로 안정성을 개선하는지 확증하기 어렵습니다.
- 또한 FFN(Feed-Forward Network, 각 위치를 독립적으로 한 번 더 비선형 변환하는 층)이 아직 없습니다. 지금의 `DecoderLayer`는 attention만 담고 있는데, 실제 Transformer의 decoder layer는 attention 다음에 FFN을 거치고, FFN 뒤에도 residual + LayerNorm이 한 번 더 있는 구조입니다.
- 이 한계가 11단계(층 쌓기 + 전체 Transformer)로 넘어가는 이유입니다. 11단계는 `DecoderLayer`를 여러 개 쌓아 `TransformerDecoder`를 완성하고, 지금까지 만든 모든 구성 요소(1단계의 forward-loss-backward-update부터 10단계의 attention까지)를 하나로 조립합니다.

## 참고

- positional encoding, residual connection, LayerNorm의 원 논문 근거는 Vaswani et al., "Attention Is All You Need"(2017) 3.1절(Encoder and Decoder Stacks), 3.5절(Positional Encoding)입니다. (https://arxiv.org/abs/1706.03762)
- LayerNorm 자체의 원 논문은 Ba, Kiros, Hinton, "Layer Normalization"(2016)입니다. (https://arxiv.org/abs/1607.06450)
