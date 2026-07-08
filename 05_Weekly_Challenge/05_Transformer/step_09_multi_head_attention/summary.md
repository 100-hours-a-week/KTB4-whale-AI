# multi-head attention

## 배경

- 8단계 "결론 및 한계"에서 확정한 대로, `Wq`, `Wk`, `Wv` 한 세트(head 1개)로는 한 가지 관점의 관련성만 포착할 수 있었습니다. 문장 안에서 문법적 관계(주어-동사)와 의미적 관계(대명사-선행사)는 서로 다른 종류의 관련성인데, 하나의 세트만으로는 이 다양성을 동시에 표현할 방법이 없었습니다.
- 근본적으로, "한 세트의 가중치가 여러 관점을 동시에 대변해야 한다"는 요구 자체가 무리한 것입니다. `Wq`, `Wk` 하나는 학습을 통해 어떤 하나의 유사도 기준(예: "품사가 같은가")으로 수렴할 수는 있어도, 동시에 "품사가 같은가"와 "의미가 통하는가"라는 서로 다른 기준을 함께 만족시키도록 수렴하기는 어렵습니다. 이건 8단계 구조의 표현력(representational capacity) 자체가 갖는 한계입니다.
- 9단계는 이 문제를, `Wq`, `Wk`, `Wv` 세트를 여러 개(head) 병렬로 두어서 해결합니다. 각 head가 독립적인 파라미터로 각자 다른 기준을 학습하게 하고, 그 결과를 나중에 합칩니다.

## 동작 원리

**forward**

```python
# Before — 8단계(scaled_dot_product_attention.py)
def forward(self, X: np.ndarray) -> tuple:
    X_val = NumpyValue(X)
    Q = X_val @ self.Wq
    K = X_val @ self.Wk
    V = X_val @ self.Wv
    S = Q @ K.transpose()
    S_scaled = S * (1.0 / np.sqrt(self.d_k))
    A = S_scaled.softmax(axis=-1)
    O = A @ V
    return O, A

# After — 9단계(multi_head_attention.py)
def forward(self, X: np.ndarray) -> tuple:
    head_outputs = [head.forward(X) for head in self.heads]
    head_O_values = [O for O, A in head_outputs]

    O_concat = NumpyValue.concat(head_O_values, axis=-1)
    O = O_concat @ self.Wo

    return O, head_outputs
```

- 8단계는 `Wq`, `Wk`, `Wv` 한 세트로 직접 `Q`, `K`, `V`, `S`, `A`, `O`를 전부 계산했습니다.
- 9단계는 이 계산 전체(`ScaledDotProductAttention.forward()`)를 `self.heads`(head 개수만큼 만들어둔 리스트) 각각에 대해 반복 호출합니다. 각 head는 8단계 코드를 그대로 재사용하되, `self.__init__`에서 서로 다른 `seed`로 초기화되어 있어(2단계에서 은닉층 뉴런들을 서로 다르게 초기화했던 것과 같은 이유) 각자 독립적인 `Wq`, `Wk`, `Wv`를 갖습니다.
- `NumpyValue.concat(head_O_values, axis=-1)`이 8단계에는 없던 신규 연산입니다. 각 head의 출력(`(seq_len, d_k)`)을 열 방향으로 이어 붙여서 `(seq_len, d_model)`(head 개수 × `d_k` = `d_model`)로 되돌립니다.
- `O_concat @ self.Wo`가 최종적으로 추가된 선형 변환입니다. 각 head가 독립적으로 계산한 결과를 단순히 나열한 것(concat)만으로는 head들의 정보가 서로 섞이지 않으므로, `Wo`(output projection, 출력 투영)가 이 여러 head의 결과를 다시 하나로 혼합합니다.
- forward 실행 시, NumpyValue 동작 과정
  1. 각 `head.forward(X)`가 호출될 때마다, 8단계에서 검증한 계산 그래프(`Q@Wq`, `K@Wk`, `V@Wv`, `S=Q@K.T`, scaling, softmax, `A@V`)가 독립적으로 만들어집니다. `head` 개수만큼 이 그래프가 병렬로(서로 겹치지 않고) 존재합니다.
  2. (`NumpyValue.concat`) 새로운 노드가 생성되고, `_prev`에 모든 head의 출력(`O_1, ..., O_num_heads`)이 자식으로 기록됩니다. 이 노드가 여러 head의 그래프를 하나로 합류시키는 첫 지점입니다.
  3. (`O_concat @ self.Wo`) 곱셈 연산이 실행되어 최종 `O`가 만들어집니다. `_prev`에 `O_concat`, `Wo`가 자식으로 기록됩니다.
  4. `(O, head_outputs)`를 반환합니다. `head_outputs`를 함께 반환하는 이유는, 각 head가 실제로 서로 다른 attention 패턴을 보이는지 직접 확인하기 위함입니다.

**backward** (신규 연산 `concat`만 다룸, 나머지는 8단계와 동일)

```python
# After — 9단계(value_numpy.py)
def _backward():
    offset = 0
    for v, size in zip(values, sizes):
        slicer = [slice(None)] * out.grad.ndim
        slicer[axis] = slice(offset, offset + size)
        v.grad += out.grad[tuple(slicer)]
        offset += size
```

- `concat`이 "여러 조각을 하나로 합치는" 연산이었으므로, 그 backward는 정반대로 "하나로 합쳐진 gradient를 원래 조각의 크기만큼 잘라서 각자에게 돌려주는" 연산입니다. `offset`을 누적하면서 `out.grad`를 슬라이싱(slicing)해 각 head의 출력(`v`)에 정확히 그 몫만큼만 전달합니다.
- 이 분배가 정확해야, 각 head의 `Wq`, `Wk`, `Wv`가 오직 자신이 담당한 구간의 gradient만 받고, 다른 head의 gradient가 섞여 들어오지 않습니다.

## 구현

- [NumpyValue 클래스 (concat 추가, NumPy)](./value_numpy.py)
- [ScaledDotProductAttention (8단계, head 1개 단위로 재사용)](./scaled_dot_product_attention.py)
- [MultiHeadAttention (head 여러 개 + concat + Wo)](./multi_head_attention.py)
- [backward 정확성 및 head별 관점 차이 검증 테스트](./test_multi_head_attention.py)

## 결과 정리

**표 1. multi-head attention 검증 결과**
| 검증 항목 | 결과 |
|---|---|
| 출력 모양이 입력과 같은 `d_model` 차원을 유지하는가 | `O.shape=(5, 8)`, `d_model=8`과 일치, head 개수=4 |
| 전체 forward-backward vs 수치 미분 | head 0의 `Wq` 18개 원소 전부 일치 |
| 서로 다른 head가 다른 attention weight를 학습하는가 | 4개 head 전부 서로 다른 패턴 |

**표 2. head별 attention weight (시퀀스 첫 번째 위치 기준, 4개 위치에 대한 비중)**
| head | 위치 0 | 위치 1 | 위치 2 | 위치 3 | 가장 크게 반영한 위치 |
|---|---|---|---|---|---|
| 0 | 0.2511 | 0.2597 | 0.2800 | 0.2091 | 위치 2 |
| 1 | 0.2635 | 0.2748 | 0.2141 | 0.2475 | 위치 1 |
| 2 | 0.2626 | 0.2274 | 0.2382 | 0.2717 | 위치 3 |
| 3 | 0.2035 | 0.2367 | 0.2771 | 0.2827 | 위치 3 |

- 4개 head가 같은 입력(첫 번째 위치)에 대해 서로 다른 위치를 가장 크게 반영하고 있습니다(head 0은 위치 2, head 1은 위치 1, head 2와 head 3은 위치 3). `Wq`, `Wk`, `Wv`를 head마다 독립적으로 초기화한 결과, 각 head가 실제로 서로 다른 관점의 관련도를 계산하고 있다는 근거입니다.

## 결론 및 한계

- 8단계에서 "head 1개로는 한 가지 관점만 포착 가능하다"고 짚었던 한계가 구조적으로 해결됐습니다. 4개 head를 만들어 실제로 실행해보니, 각 head의 attention weight(표 1)가 서로 다르게 나왔습니다 — head마다 시퀀스 안에서 서로 다른 위치를 중요하게 여기고 있다는 뜻입니다. `Wq`, `Wk`, `Wv`를 head마다 독립적으로 초기화한 게 이 차이를 만드는 근거입니다.
- 다만 지금은 무작위 초기화 상태라, "각 head가 정확히 문법적 관계, 의미적 관계처럼 해석 가능한 역할을 나눠 맡는다"는 보장은 없습니다. 지난 턴에서 다룬 Clark et al.(2019)의 BERT 분석처럼, 학습이 진행된 후에야 head별 역할 분담이 실제로 의미 있게 나타나는지 확인할 수 있습니다.
- 더 근본적인 한계는, 지금까지의 attention(7~9단계)이 전부 "위치 정보(positional information)" 자체를 담고 있지 않다는 점입니다. `Q = X @ Wq` 같은 계산은 `X`의 각 행이 시퀀스의 몇 번째 위치인지 전혀 참조하지 않습니다 — 즉 입력 순서를 통째로 섞어도 `S`, `A`, `O`의 값 자체는 (행 순서만 바뀔 뿐) 동일하게 나옵니다. attention은 "관련도"는 계산하지만 "순서"는 모릅니다.
- 이 한계가 10단계(positional encoding + residual + LayerNorm)로 넘어가는 이유입니다. 10단계는 입력에 위치 정보를 명시적으로 주입하고(positional encoding), 학습 안정화를 위한 residual connection과 LayerNorm을 추가합니다.

## 참고

- multi-head attention의 원 논문 근거는 Vaswani et al., "Attention Is All You Need"(2017) 3.2.2절(Multi-Head Attention)입니다. (https://arxiv.org/abs/1706.03762)
