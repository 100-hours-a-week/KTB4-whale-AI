# activation function

## 배경

- single neuron은 forward할 때, 일차함수로 고정되어 있습니다. 그래서 아무리 학습해도 직선 이외의 형태는 표현할 수 없습니다. 이 한계를 보여주는 고전적인 사례로 XOR 문제가 있습니다. XOR 문제는 2개의 이진 입력을 받아 정확히 하나만 1일 때 1을 출력하는 함수입니다. 이 문제는 linearly separable (선형 분리 가능)하지 않습니다.
- Minsky & Papert의 "Perceptrons"(1969)가 이 한계를 수학적으로 처음 증명했습니다.
- activation function은 이 문제를 완전히 해결하지는 못합니다.(이유는 3단계 MLP에서 다루겠습니다.) 하지만 activation function 없이는 여러 층(layer)을 쌓아도 소용없다는 점을 먼저 확인해야, 3단계의 필요성을 이해할 수 있습니다. 이번 단계에서는 activation function 자체를 구현하고, 이걸 추가해도 "뉴런 1개"로는 XOR을 풀 수 없다는 것을 확인해보겠습니다.

## 동작 원리

- (핵심) forward 단계에서 activation function으로 sigmoid를 추가합니다. 이 삽입으로 인해 아래와 같이 코드가 개선됩니다.

**forward**

```python
# Before — 1단계(SingleNeuron)
def forward(self, x: float) -> float:
    return self.w * x + self.b

# After - 2단계(NeuronWithActivation)
def forward(self, x1: float, x2: float) -> tuple[float, float]:
    z = self.w1 * x1 + self.w2 * x2 + self.b
    a = sigmoid(z)
    return a, z
```

- 기존에는 $y_pred = w\*x + b$ 하나로 고정되어 있었습니다. 2단계는 두 가지 구조 변경이 있습니다. 입력을 2개로 확장하고, 선형 결합(linear combination) 결과에 sigmoid를 씌웁니다.
- 입력을 2개로 확장한 이유는 검증 대상으로 삼은 문제(AND, XOR)가 2개의 이진 입력을 받는 논리 게이트이기 때문입니다. sigmoid를 forward에 삽입하는 이유는 비선형성(non-linearity)이 1단계의 한계를 해결하는 핵심이기 때문입니다.

**loss**

```python
# Before — 1단계(SingleNeuron)
def compute_loss(self, y_pred: float, y_true: float) -> float:
    return (y_pred - y_true) ** 2

# After - 2단계(NeuronWithActivation)
def compute_loss(self, a: float, y_true: float) -> float:
    return (a - y_true) ** 2
```

- 1단계의 동일한 형태를 그대로 사용합니다.
- 대신, 앞서 forward로부터 전달받은 변수명이 더 이상 y_pred이 아니므로, a로 변경되었습니다.

**backward**

```python
# Before — 1단계(SingleNeuron)
def backward(self, x: float, y_pred: float, y_true: float) -> tuple[float, float]:
    error = y_pred - y_true
    dL_dw = 2 * error * x
    dL_db = 2 * error
    return dL_dw, dL_db

# After - 2단계(NeuronWithActivation)
def backward(self, x1: float, x2: float, a: float, y_true: float) -> tuple[float, float, float]:
    error = a - y_true

    # (w1) dL/dw1 = dL/da * da/dz * dz/dw1
    dL_da = 2 * error
    da_dz = a * (1 - a)  # sigmoid_derivative(z)를 a로부터 바로 계산 (효율적)
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
```

- 1단계에서는 2단(`y_pred -> loss`)짜리 chain rule이었으나, 2단계에서는 3단짜리로 한 단계 늘어났습니다. (`z -> a -> loss`)
- `dL_dw1`, `dL_dw2`는 1단계의 `dL_dw`와 같은 역할이지만, 공통 계수(`dL_dz`)에 각 입력(`x1`, `x2`)을 곱하는 방식으로 확장됩니다. (`dL/da * da/dz * dz/dw1`, `dL/da * da/dz * dz/dw2`)

**update**

```python
# Before — 1단계(SingleNeuron)
def update(self, dL_dw: float, dL_db: float, learning_rate: float):
    self.w -= learning_rate * dL_dw
    self.b -= learning_rate * dL_db

# After - 2단계(NeuronWithActivation)
def update(self, dL_dw1: float, dL_dw2: float, dL_db: float, learning_rate: float):
    self.w1 -= learning_rate * dL_dw1
    self.w2 -= learning_rate * dL_dw2
    self.b -= learning_rate * dL_db
```

- 구조적 변화는 없습니다. 갱신 대상 파라미터가 `w, b` 2개에서 `w1, w2, b` 3개로 늘어난 것뿐입니다.

## 개선 이유

- sigmoid를 사용하는 3가지 이유
  1. 출력 범위가 항상 (0, 1) 사이입니다. 지금처럼 결과가 0, 1과 같은 이진 분류(binary classification)의 확률처럼 해석이 가능합니다.
  2. **비선형성(non-linearity)을 띕니다.** 즉 1단계에서 "가설 공간(hypothesis space)이 직선의 집합으로 고정되어 있었다"는 근본 한계를, sigmoid가 곡선을 표현 가능한 형태로 확장하는 방식으로 직접 해결하는 지점이 바로 이 특성입니다.
  3. 미분 가능합니다. 모든 지점에서 매끄럽게 미분 가능하기 때문에, 1단계와 동일하게 gradient descent를 적용할 수 있습니다. 1단계에서 요구됐던 "손실 함수가 모든 지점에서 미분 가능해야 한다"는 조건이 activation function에도 동일하게 적용된다는 것을 의미합니다.

## 결론 및 한계

- sigmoid activation을 추가해서 뉴런이 이제 곡선(비선형)을 표현할 수 있게 됐는데도, XOR은 여전히 풀리지 않습니다. 이유는 activation이 "직선을 구부리는" 역할은 하지만, 뉴런이 1개뿐이라 여전히 결정 경계(decision boundary, 분류를 가르는 경계선)가 하나만 만들어지기 때문입니다.
- `z = w1*x1 + w2*x2 + b`라는 하나의 선형 결합(linear combination)에 sigmoid를 씌운 것뿐이라, 그 결정 경계는 여전히 하나의 곡선(sigmoid의 경우 사실상 하나의 직선을 매끄럽게 만든 것)입니다. XOR을 풀려면 (0,1)-(1,0) 그룹과 (0,0)-(1,1) 그룹을 나누는 데 최소 2개의 결정 경계가 필요한데, 뉴런 1개는 경계를 1개만 만들 수 있기 때문입니다.
- 이게 3단계(MLP, 여러 뉴런을 층으로 쌓기)로 넘어가야 하는 이유입니다. 뉴런을 여러 개 병렬로 두면 각자 다른 결정 경계를 학습하고, 그 결과를 다음 층에서 조합(combine)하면 XOR처럼 복잡한 경계도 표현할 수 있게 됩니다. 이 사실(은닉층을 가진 다층 신경망이 XOR을 포함한 임의의 함수를 근사할 수 있다는 것)은 Cybenko(1989)의 [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)으로 공식화되어 있습니다.
