# 층 쌓기 + 전체 Transformer (FFN, TransformerDecoder)

## 배경

- 8단계에서 검증한 `O = A @ V`를 다시 보면, 이건 `V`들의 가중 평균(weighted sum)입니다. 가중치(`A`)를 정하는 softmax에는 비선형성이 있지만, 정해진 가중치로 `V`를 실제로 섞는 연산 자체(`A @ V`)와 `V`를 만드는 연산(`X @ Wv`) 모두 선형(linear) 연산입니다. 즉 지금까지 attention이 하는 일은 "어떤 위치를 얼마나 참조할지"를 정할 뿐, 그렇게 모은 정보를 위치별로 비선형적으로 재가공하는 단계가 전혀 없습니다.
- 이건 1~2단계에서 확인했던 문제의 재현입니다. 1단계(`SingleNeuron`)가 선형식만 표현 가능해서 XOR을 못 풀었던 것과 같은 이유로, attention만 있는 층은 위치별로 정보를 "재분배"할 수는 있어도 "비선형적으로 재가공"하지는 못합니다. 실제로 attention을 거치지 않고 순수 선형 변환(`W`, `b`)만으로 위치별 비선형 함수(`y=x^2`)를 학습시켜본 결과, MSE `5.73`으로 수렴하지 못했습니다 — 목표값(`1,4,9,1,4`)과 전혀 다른 값(`4.16, 5.07, 5.98, ...`)에 머물렀습니다.
- 11단계는 이 문제를, 3단계에서 만든 MLP(비선형 activation을 포함한 2층 신경망)를 시퀀스의 각 위치에 독립적으로 적용해서 해결합니다. 이걸 FFN(Feed-Forward Network, 위치별 순방향 신경망)이라고 부릅니다.

## 동작 원리

### 관련 개념 요약

**Feed-Forward Network (위치별 순방향 신경망)**

FFN은 3단계에서 만든 MLP와 구조가 완전히 같습니다 — 입력을 은닉층으로 선형 변환한 뒤 비선형 activation(ReLU)을 거치고, 다시 출력으로 선형 변환합니다. 차이는 이 계산이 시퀀스의 각 위치(토큰)마다 독립적으로, 그러나 모든 위치에 동일한 가중치(`W1`, `W2`)로 적용된다는 점입니다. attention이 "위치들 사이"의 관계를 다뤘다면, FFN은 "위치 하나 안"의 정보를 재가공합니다 — 이 둘이 번갈아 적용되는 것이 decoder layer의 골격입니다. (선형 함수는 합성에 닫혀 있음)

```
FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
```

**forward**

```python
# Before — 10단계(decoder_layer.py)
def forward(self, X: np.ndarray) -> NumpyValue:
    seq_len = X.shape[0]
    pe = positional_encoding(seq_len, self.d_model)
    X_with_pos_data = X + pe
    X_with_pos = NumpyValue(X_with_pos_data)
    attention_output, _ = self.attention.forward(X_with_pos_data)
    residual_output = residual_connection(X_with_pos, attention_output)
    output = self.norm.forward(residual_output)
    return output

# After — 11단계(decoder_layer.py)
def forward(self, X) -> NumpyValue:
    X_val = X if isinstance(X, NumpyValue) else NumpyValue(X)
    seq_len = X_val.data.shape[0]
    pe = positional_encoding(seq_len, self.d_model)
    X_with_pos = X_val + pe

    attention_output, _ = self.attention.forward(X_with_pos)
    residual1 = residual_connection(X_with_pos, attention_output)
    normed1 = self.norm1.forward(residual1)

    ffn_output = self.ffn.forward(normed1)
    residual2 = residual_connection(normed1, ffn_output)
    output = self.norm2.forward(residual2)

    return output
```

- 10단계는 attention → residual → LayerNorm으로 끝났습니다.
- 11단계는 그 뒤에 FFN → residual → LayerNorm이 한 번 더 이어집니다. `self.ffn.forward(normed1)`이 신규 추가된 부분이고, 그 결과에 다시 `residual_connection`(원본과 더함)과 `self.norm2.forward`(정규화)가 적용됩니다.
- `X_val = X if isinstance(X, NumpyValue) else NumpyValue(X)`도 신규 변경점입니다. `X`가 이번 층의 원본 입력(`np.ndarray`)일 수도 있고, 여러 층을 쌓았을 때 이전 층의 출력(`NumpyValue`, 계산 그래프 포함)일 수도 있기 때문입니다. 이미 `NumpyValue`면 그대로 쓰고, 아니면 새로 감싸서, 층 사이에서 계산 그래프가 끊기지 않게 보존합니다.
- forward 실행 시, NumpyValue 동작 과정
  1. (`normed1 @ self.ffn.W1 + self.ffn.b1`) 곱셈·덧셈 연산이 실행되어 새 노드가 생성됩니다. `_prev`에 `normed1`, `ffn.W1`(그리고 덧셈 시 `ffn.b1`)이 자식으로 기록됩니다.
  2. (`.relu()`) 신규 연산입니다. 입력이 0보다 큰 위치는 그 값을 그대로, 0 이하인 위치는 0으로 바꾼 배열을 담은 새 노드가 생성됩니다.`_backward`는 "0보다 큰 지점만 기울기 1을 그대로 통과시키고, 나머지는 0을 곱해서 막는다"는 마스크(mask) 방식으로 정의만 해둡니다.
  3. (`hidden @ self.ffn.W2 + self.ffn.b2`) 1과 동일한 방식으로 최종 `ffn_output`이 만들어집니다.
  4. (`residual_connection(normed1, ffn_output)`) `__add__`가 호출되어, `normed1`과 `ffn_output`을 자식으로 하는 새 노드가 생성됩니다. `normed1`이 이 시점에 두 번째로 그래프에 등장합니다(한 번은 FFN의 입력으로, 한 번은 residual의 피연산자로) — 4단계에서 다룬 "다중 경로 gradient 합산"과 같은 구조입니다.
  5. (`self.norm2.forward(...)`) 10단계에서 검증한 LayerNorm 계산이 한 번 더 반복되어 최종 `output`이 만들어집니다.

## 구현

- [NumpyValue 클래스 (relu 추가, NumPy)](./value_numpy.py)
- [FeedForward (FFN)](./feed_forward.py)
- [DecoderLayer (attention + FFN + residual + LayerNorm, 여러 층 연결 가능하도록 수정)](./decoder_layer.py)
- [TransformerDecoder (DecoderLayer 여러 개를 쌓은 전체 구조)](./transformer_decoder.py)
- [검증 테스트](./test_transformer.py)

## 결과 정리

**표 1. FFN 필요성 및 정확성 검증 결과**
| 검증 항목 | 결과 |
| --- | --- |
| ReLU backward vs 수치 미분 | 5개 지점 전부 일치 |
| FFN 없이 위치별 비선형 함수(`y=x^2`) 학습 | MSE 5.7302 (실패) |
| FFN 사용 시 같은 과제 학습 | MSE 0.1333, 목표(`1,4,9,1,4`)에 근접(`0.67,4.67,8.67,1.0,4.0`) |
| `DecoderLayer`(FFN 포함) 전체 backward vs 수치 미분 | `ffn.W1` 24개 원소 전부 일치 |
| 3층을 쌓았을 때 첫 번째 층까지 gradient 도달 | 3개 층 전부 0이 아닌 gradient 확인 |

## 결론 및 한계

- 8~9단계에서 확인한 "attention은 선형 연산만으로 이루어져 있다"는 한계가 FFN으로 해결됐습니다. 같은 위치별 비선형 함수(`y=x^2`)에 대해, FFN 없이는 MSE `5.73`으로 실패했지만 FFN을 쓰면 `0.13`까지 줄어들어 목표값에 근접했습니다(표 1).
- 여러 층을 쌓는 과정에서 실제로 계산 그래프가 끊기는 문제를 발견하고 수정했습니다. 각 컴포넌트가 입력을 무조건 `NumpyValue(X)`로 새로 감싸던 방식은, 이전 층의 출력(이미 계산 그래프를 가진 `NumpyValue`)을 다음 층에 전달할 때 그 그래프를 끊어버렸습니다. `isinstance` 체크로 "이미 NumpyValue면 그대로, 아니면 새로 감싼다"는 교체해서, 3층을 쌓은 상태에서도 gradient가 첫 번째 층까지 정확히 도달한다는 걸 검증했습니다(표 1의 마지막 행).
- 이로써 1단계(forward-loss-backward-update)부터 11단계(FFN)까지 만든 모든 요소가 `TransformerDecoder` 하나로 조립됐습니다. 처음 로드맵(`docs.md`)에서 "지금까지 요소 전부 조합"이라고 예고했던 최종 단계가 완성된 것입니다.
- 다만 지금 구현은 여전히 loss·update가 실제 언어 과제(next-token prediction 등)로 완결되어 있지 않습니다. `test_with_ffn_can_fit_positionwise_nonlinear_function`처럼 부분적인 학습 능력만 검증했을 뿐, `TransformerDecoder` 전체를 실제 텍스트 데이터로 학습시키는 것은 이번 로드맵의 범위를 넘어서는 별도의 작업입니다.
- 또한 지금까지 만든 `NumpyValue` 자동 미분 엔진은 교육 목적의 최소 구현입니다. 실제 프로덕션 환경에서는 4단계에서 비교 검증했던 PyTorch의 `torch.Tensor`(GPU 병렬화, 최적화된 커널)를 사용하는 것이 맞고, 이번 로드맵 전체가 그 이유(왜 각 구성 요소가 필요한지, 무엇을 계산하는지)를 근본부터 이해하기 위한 과정이었습니다.

## 참고

- FFN(position-wise feed-forward network)의 원 논문 근거는 Vaswani et al., "Attention Is All You Need"(2017) 3.3절입니다. (https://arxiv.org/abs/1706.03762)
