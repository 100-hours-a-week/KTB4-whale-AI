# MLP (Multi-Layer Perceptron)

## 배경

- 2단계에서 XOR 문제를 풀려면 최소 2개의 결정 경계(decision boundary)가 필요한데 뉴런 1개는 경계 1개만 만들었습니다. 이 문제를 해결하는 구조가 MLP(Multi-Layer Perceptron)입니다.
- 은닉층(hidden layer)에 뉴런을 여러 개 두면, 각 뉴런이 서로 다른 결정 경계를 학습하고, 그 결과를 출력층(output layer)에서 조합(combine)해서 복잡한 경계를 만들 수 있습니다.

## 동작 원리

**forward**

```python
# Before — 2단계(NeuronWithActivation)
def forward(self, x1: float, x2: float) -> tuple[float, float]:
    z = self.w1 * x1 + self.w2 * x2 + self.b
    a = sigmoid(z)
    return a, z

# After - 3단계(MLP)
def forward(self, x1: float, x2: float) -> tuple[float, dict]:
    h1, z_h1 = self.hidden1.forward([x1, x2])
    h2, z_h2 = self.hidden2.forward([x1, x2])
    o, z_o = self.output.forward([h1, h2])

    cache = {
        'x1': x1, 'x2': x2,
        'h1': h1, 'z_h1': z_h1,
        'h2': h2, 'z_h2': z_h2,
        'o': o, 'z_o': z_o,
    }
    return o, cache
```

- 2단계는 뉴런 1개가 직접 최종 출력을 계산했습니다.
- 3단계는 이 계산을 2단(stage)로 쪼갠다. 은닉층(hidden layer) 뉴런 2개가 입력(x1, x2)를 각자 처리해서 h1, h2를 만들고, 출력층(output layer) 뉴런 1개가 이 h1, h2를 다시 입력으로 받아 최종 출력 o를 만듭니다.
- cache는 backward 단계에서 `h1, h2, z_h1, z_h2, z_o` 값이 모두 필요하기 때문에 새로 추가되었습니다. 2단계는 뉴런이 1개라 a, z 2개만 반환하면 됐지만, 3단계는 층이 2개로 늘어나면서 중간값을 전달할 대상이 많아져서 딕셔너리로 묶었습니다.

**loss**

```python
# Before — 2단계(NeuronWithActivation)
def compute_loss(self, a: float, y_true: float) -> float:
    return (a - y_true) ** 2

# After - 3단계(MLP)
def compute_loss(self, o: float, y_true: float) -> float:
    return (o - y_true) ** 2
```

- 2단계의 동일한 형태를 그대로 사용합니다.
- 대신, 앞서 forward로부터 전달받은 변수명이 더 이상 a가 아니므로, o로 변경되었습니다.

**backward**

```python
# Before — 2단계(NeuronWithActivation)
def backward(self, x1: float, x2: float, a: float, y_true: float) -> tuple[float, float, float]:
    error = a - y_true

    # (w1) dL/dw1 = dL/da * da/dz * dz/dw1
    dL_da = 2 * error
    da_dz = a * (1 - a)
    dz_dw1 = x1
    dL_dw1 = dL_da * da_dz * dz_dw1

    # (w2) dL/dw2 = dL/da * da/dz * dz/dw2
    dL_da = 2 * error
    da_dz = a * (1 - a)
    dz_dw2 = x2
    dL_dw2 = dL_da * da_dz * dz_dw2

    # (bias) dL/db  = dL/da * da/dz * dz/db
    dL_da = 2 * error
    da_dz = a * (1 - a)
    dz_db = 1
    dL_db = dL_da * da_dz * dz_db

    return dL_dw1, dL_dw2, dL_db

# After - 3단계(MLP)
def backward(self, cache: dict, y_true: float) -> dict:
    x1, x2 = cache['x1'], cache['x2']
    h1, h2 = cache['h1'], cache['h2']
    o = cache['o']

    error = o - y_true

    # ============ output (v1, v2, c) ============
    # (v1) dL/dv1 = dL/do * do/dzo * dzo/dv1
    dL_do = 2 * error
    do_dzo = o * (1 - o)
    dzo_dv1 = h1
    dL_dv1 = dL_do * do_dzo * dzo_dv1

    # (v2) dL/dv2 = dL/do * do/dzo * dzo/dv2
    dL_do = 2 * error
    do_dzo = o * (1 - o)
    dzo_dv2 = h2
    dL_dv2 = dL_do * do_dzo * dzo_dv2

    # (c) dL/dc = dL/do * do/dzo * dzo/dc
    dL_do = 2 * error
    do_dzo = o * (1 - o)
    dzo_dc = 1
    dL_dc = dL_do * do_dzo * dzo_dc

    # ============ hidden1 (w1_1, w1_2, b1) ============
    # (공통, h1) dL/dh1 = dL/do * do/dzo * dzo/dh1 (dzo/dh1 == v1 == self.output.weights[0])
    dL_do = 2 * error
    do_dzo = o * (1 - o)
    dzo_dh1 = self.output.weights[0]  # v1

    # (w1_1) dL/dw1_1 = dL/dh1 * dh1/dzh1 * dzh1/dw1_1
    dL_dh1 = dL_do * do_dzo * dzo_dh1
    dh1_dzh1 = h1 * (1 - h1)
    dzh1_dw1_1 = x1
    dL_dw1_1 = dL_dh1 * dh1_dzh1 * dzh1_dw1_1

    # (w1_2) dL/dw1_2 = dL/dh1 * dh1/dzh1 * dzh1/dw1_2
    dL_dh1 = dL_do * do_dzo * dzo_dh1
    dh1_dzh1 = h1 * (1 - h1)
    dzh1_dw1_2 = x2
    dL_dw1_2 = dL_dh1 * dh1_dzh1 * dzh1_dw1_2

    # (b1) dL/db1 = dL/dh1 * dh1/dzh1 * dzh1/db1
    dL_dh1 = dL_do * do_dzo * dzo_dh1
    dh1_dzh1 = h1 * (1 - h1)
    dzh1_db1 = 1
    dL_db1 = dL_dh1 * dh1_dzh1 * dzh1_db1

    # ============ hidden2 (w2_1, w2_2, b2) ============
    # (공통, h2) dL/dh2 = dL/do * do/dzo * dzo/dh2, (dzo/dh2 == v2 == self.output.weights[1])
    dL_do = 2 * error
    do_dzo = o * (1 - o)
    dzo_dh2 = self.output.weights[1]  # v2

    # (w2_1) dL/dw2_1 = dL/dh2 * dh2/dzh2 * dzh2/dw2_1
    dL_dh2 = dL_do * do_dzo * dzo_dh2
    dh2_dzh2 = h2 * (1 - h2)
    dzh2_dw2_1 = x1
    dL_dw2_1 = dL_dh2 * dh2_dzh2 * dzh2_dw2_1

    # (w2_2) dL/dw2_2 = dL/dh2 * dh2/dzh2 * dzh2/dw2_2
    dL_dh2 = dL_do * do_dzo * dzo_dh2
    dh2_dzh2 = h2 * (1 - h2)
    dzh2_dw2_2 = x2
    dL_dw2_2 = dL_dh2 * dh2_dzh2 * dzh2_dw2_2

    # (b2) dL/db2 = dL/dh2 * dh2/dzh2 * dzh2/db2
    dL_dh2 = dL_do * do_dzo * dzo_dh2
    dh2_dzh2 = h2 * (1 - h2)
    dzh2_db2 = 1
    dL_db2 = dL_dh2 * dh2_dzh2 * dzh2_db2

    return {
        'output': (dL_dv1, dL_dv2, dL_dc),
        'hidden1': (dL_dw1_1, dL_dw1_2, dL_db1),
        'hidden2': (dL_dw2_1, dL_dw2_2, dL_db2),
    }
```

- 2단계는 뉴런이 1개뿐이라, "w1, w2, b"에 대한 미분을 chain rule(연쇄 법칙)으로 1회 적용됐습니다. 3단계는 뉴런이 3개(출력층 1개, 은닉층 2개)로 늘어났고, backward는 이 3개 뉴런 각각에 대해 "2단계와 완전히 동일한 형태의 계산"을 반복합니다. 즉, 2단계에서 확립한 "뉴런 1개짜리 backward 패턴"을 뉴런 개수만큼 재사용합니다.
- 3번의 적용으로 달라진 점은 "패턴에 입력되는 gradient가 어디서 오는가"는 점입니다. 출력층은 2단계와 똑같이 `dL_do = 2*error`로 loss에서 직접 계산합니다. 반면, 은닉층1, 2는 출력층을 거쳐서 loss에 연결되어 있으므로, `dL_dh1`, `dL_dh2`를 출력층으로부터 전파받습니다. (`dzo_dh1`은 출력층의 weight `v1`과 같은 값)
- backward가 "은닉층까지 확장"됐다는 것은 새로운 미분 규칙의 추가라기보다 기존 backward 계산을 네트워크 뉴런 개수만큼 반복하되, 출력에 가까운 뉴런부터 이전 뉴런에 넘겨주는 순서로 진행한다는 뜻입니다.
- 이 "동일 패턴의 반복 적용 + 순서에 따른 전파"가 backpropagation(역전파 알고리즘)의 본질이며, 층이 몇 개로 늘어나든(4단계 이후 다층 신경망, Transformer의 여러 decoder layer 등) 이 원리 자체는 바뀌지 않습니다.

**update**

```python
# Before — 2단계(NeuronWithActivation)
def update(self, dL_dw1: float, dL_dw2: float, dL_db: float, learning_rate: float):
    self.w1 -= learning_rate * dL_dw1
    self.w2 -= learning_rate * dL_dw2
    self.b -= learning_rate * dL_db

# After - 3단계(MLP)
def update(self, gradients: dict, learning_rate: float):
    dL_dv1, dL_dv2, dL_dc = gradients['output']
    self.output.weights[0] -= learning_rate * dL_dv1
    self.output.weights[1] -= learning_rate * dL_dv2
    self.output.bias -= learning_rate * dL_dc

    dL_dw1_1, dL_dw1_2, dL_db1 = gradients['hidden1']
    self.hidden1.weights[0] -= learning_rate * dL_dw1_1
    self.hidden1.weights[1] -= learning_rate * dL_dw1_2
    self.hidden1.bias -= learning_rate * dL_db1

    dL_dw2_1, dL_dw2_2, dL_db2 = gradients['hidden2']
    self.hidden2.weights[0] -= learning_rate * dL_dw2_1
    self.hidden2.weights[1] -= learning_rate * dL_dw2_2
    self.hidden2.bias -= learning_rate * dL_db2
```

- 구조적 변화는 없습니다. `w -= learning_rate * gradient`라는 갱신 공식 자체는 동일하고, 갱신 대상이 뉴런 1개(파라미터 3개)에서 뉴런 3개(파라미터 9개: 은닉층 뉴런 2개 \* 3개 + 출력층 뉴런 1개 \* 3개)로 늘어났을 뿐입니다.

## 구현

- [바로가기](./mlp.py)

## 결과 정리

**표 1. 2단계 vs 3단계 — XOR 문제 해결 여부**
| 단계 | 구조 | XOR 최종 MSE | 4개 케이스 정확도 |
| --- | --- | --- | --- |
| 2단계 (`NeuronWithActivation`) | 뉴런 1개 | 0.257689 | 2/4 |
| 3단계 (`MLP`) | 은닉층 뉴런 2개 + 출력층 뉴런 1개 | 0.000063 | 4/4 |

## 결론 및 한계

- 은닉층을 추가해서 뉴런 2개가 각자 다른 결정 경계를 학습하고, 출력층이 이를 조합한 결과 XOR를 정확히 풀었습니다. 이게 "층 쌓기(layer stacking)로 표현력을 확장"하는 것입니다.

- 한계 1
  - 다만 이 과정에서 backward 코드가 이미 상당히 길어졌습니다. 파라미터 9개 (v1, v2, c, w1_1, w1_2, b1, w2_1, w2_2, b2) 각각에 대해 chain rule을 손으로 나열해야 했고, 은닉층 파라미터는 인수 5개짜리 곱셈까지 직접 유도해야 했습니다. 1단계(파라미터 2개), 2단계(파라미터 3개) 대비 뚜렷하게 증가했습니다.
  - 이 추세는 뉴런과 층이 늘어날수록 계속됩니다. 실제로 transformer 모델은 파라미터가 수백만 개인 모델입니다. 지금처럼 chain rule을 수기로 유도하는 방식이 사실상 불가능합니다. 이게 4단계 backward를 autograd로 전환해야 하는 이유입니다.
  - 수동 역전파가 규모에 따라 복잡성이 비현실적이고, 이를 해결하기 위한 자동 미분(automatic differentiation)의 필요성은 "Automatic Differentiation in Machine Learning: a Survey(2018)"에서 정식으로 다룹니다. [[바로가기](https://arxiv.org/abs/1502.05767)]
  - PyTorch의 autograd의 필요성과 메커니즘은 공식 문서에서 확인됩니다. [[바로가기](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)]

- 한계 2 (참고)
  - 이번 구현 과정에서 처음 시도했을 때 MSE가 0.142에서 정체(plateau)됐고, 4개 중 2개만 맞히는 결과가 나왔습니다. 이는 local minimum (지역 최솟값)에 갇힌 것입니다. 1단계에서 다룬 loss 곡면은 convex(볼록, 최솟값이 하나뿐)했지만, MLP처럼 파라미터가 많아지고 층이 쌓이면 loss 곡면이 non-convex(비볼록, 여러 개의 지역 최솟값이 존재)해집니다.
  - 실제로 seed를 바꿔가며 스캔한 결과 0부터 29이면 대부분은 수렴하지만, 42는 실패했습니다. 이는 초기값에 따라 결과가 크게 갈린다는 점을 시사합니다. "신경망 학습이 초기값에 민감하고, non-convex optimization 문제가 발생하는" 이 현상은 4단계로 개선되는 작업과는 별개입니다. 따라서 바로 사라지지 않고, 다양한 optimizer 또는 초기화 기법(Xavier, He initialization 등)으로 완화하는 방향으로 다뤄지는데, 현재는 학습 범위를 넘어서므로 짚어만 둡니다.
  - 해당 현상은 "Deep Learning(심층 학습)" 8장(Optimization for Training Deep Models)에서 다루고 있습니다. [[바로가기](https://www.deeplearningbook.org/contents/optimization.html)]
