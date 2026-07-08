# vectorization

## 배경

- 4단계에서 지금까지의 MLP(3~4단계)는 XOR 데이터 4개를 학습할 때 매번 반복문으로 샘플 하나씩 순차 처리했다. 파라미터도 w1_1, w1_2, b1처럼 개별 Value로 나뉘어 있어서, 은닉층 뉴런 하나의 계산도 반복문이었다. 이걸 행렬(matrix) 하나로 묶으면, "4개 샘플 × 2개 은닉 뉴런"이라는 계산 전체를 행렬곱(matrix multiplication) 한 번으로 표현할 수 있다.

## 동작 원리

**forward**

```python
# Before — 4단계(mlp_pure_autograd.py)
def forward(self, x1: float, x2: float) -> Value:
    h1 = self.hidden1.forward([x1, x2])
    h2 = self.hidden2.forward([x1, x2])
    o = self.output.forward([h1, h2])
    return o

# After 1 — 5단계(mlp_pure.py, 순수 Python)
def forward(self, X: list) -> PureValue:
    X_val = PureValue([row[:] for row in X])
    Z1 = X_val @ self.W1 + self.b1
    H = Z1.sigmoid()
    Z2 = H @ self.W2 + self.b2
    O = Z2.sigmoid()
    return O

# After 2 — 5단계(mlp_numpy.py, NumPy)
def forward(self, X: np.ndarray) -> NumpyValue:
    X_val = NumpyValue(X)
    Z1 = X_val @ self.W1 + self.b1
    H = Z1.sigmoid()
    Z2 = H @ self.W2 + self.b2
    O = Z2.sigmoid()
    return O
```

- 4단계는 샘플 1개(x1, x2인 스칼라 2개)만 받아서, 은닉 뉴런 각각을 따로 호출했습니다.
- 5단계는 모두 배치 전체를 한 번에 받습니다.
- forward 실행 시, Value 동작 과정
  1. (`X_val @ self.W1`) 곱셈 연산(`__matmul__`)이 실행됩니다.
     1. 새로운 Value(`Z1`)가 생성됩니다.
        1. `Z1`은 초기화 과정을 통해 자신을 만드는 데 쓰인 입력(`X_val`, `self.W1`)을 자식(`_children`)으로 삼아 `self._prev`에 기록합니다.
        2. 또한, `Z1`은 초기화 과정을 통해 `_op`를 `self._op`에 기록합니다.
     2. `Z1._backward`로 이때 실행할 `_backward`(backward 연산의 일부)가 무엇인지 기록합니다.
        1. 해당 과정은 정의만 해둡니다.
        2. 실제 연산은 추후 backward() 메서드가 실행될 때 진행합니다.
     3. `Z1` 객체를 반환합니다.
  2. (`+ self.b1`) 덧셈 연산(`__add__`)이 실행됩니다. `self.b1`의 행 개수가 배치보다 적으므로, 브로드캐스팅(broadcasting)이 함께 적용됩니다.
     1. 이하 동일
  3. (`.sigmoid()`) sigmoid 연산이 실행됩니다.
     1. 이하 동일
  4. (`H @ self.W2 + self.b2`, `.sigmoid()`) 1~3과 동일한 과정이 출력층에서 한 번 더 반복됩니다.

**loss**

```python
# Before — 4단계
diff = o + (-1 * y_true)
loss = diff * diff

# After 1 — 5단계(순수 Python)
def compute_loss(self, O: PureValue, y_true: list) -> PureValue:
    y_neg = PureValue([[-v for v in row] for row in y_true])
    diff = O + y_neg
    squared = diff * diff
    batch_size = len(y_true)
    return squared.sum() * (1.0 / batch_size)

# After 2 — 5단계(NumPy)
def compute_loss(self, O: NumpyValue, y_true: np.ndarray) -> NumpyValue:
    diff = O + (-1 * y_true)
    squared = diff * diff
    batch_size = y_true.shape[0]
    return squared.sum() * (1.0 / batch_size)
```

- 4단계는 샘플 1개짜리 squared error 하나만 계산했습니다.
- 5단계는 모두 squared가 배치 크기만큼의 값을 담은 뒤, batch_size로 나누어 평균을 계산합니다.
- SE에서 MSE로 확장되는 이유
  - 4단계는 샘플 1개만 다루므로 squared error(SE) 자체가 곧 그 스텝의 loss였습니다.
  - 5단계는 배치 전체(4개 샘플)를 한 번에 다루므로, `squared`는 샘플마다 다른 SE 값 4개를 담은 행렬입니다.
  - 이 4개를 대표하는 값 하나로 요약해야 학습이 진행되므로, `.sum()`으로 전부 더한 뒤 `batch_size`로 나눠 평균(MSE)을 냅니다.
  - `.sum()`만 하고 평균을 안 내면, 배치 크기가 커질수록 loss 값 자체가 커져서 `learning_rate`를 배치 크기마다 다시 맞춰야 하는 문제가 생깁니다. 평균을 내면 배치 크기와 무관하게 loss 스케일이 일정하게 유지됩니다.
- loss 실행 시, Value 동작 과정
  1. (`O + y_neg`) 덧셈 연산(`__add__`)이 실행됩니다. (4단계의 `o + (-1 * y_true)`와 동일한 패턴이지만, 이번엔 `y_true` 전체가 행렬이라 원소별로 부호가 반전됩니다.)
     1. 새로운 Value(`diff`)가 생성됩니다.
        1. `diff`는 초기화 과정을 통해 자신을 만드는 데 쓰인 입력(`O`, `y_neg`)을 자식(`_children`)으로 삼아 `self._prev`에 기록합니다.
        2. 또한, `diff`는 초기화 과정을 통해 `_op`를 `self._op`에 기록합니다.
     2. `diff._backward`로 이때 실행할 `_backward`(backward 연산의 일부)가 무엇인지 기록합니다.
        1. 해당 과정은 정의만 해둡니다.
        2. 실제 연산은 추후 backward() 메서드가 실행될 때 진행합니다.
     3. `diff` 객체를 반환합니다.
  2. (`diff * diff`) 곱셈 연산(`__mul__`)이 실행됩니다. (원소별 곱셈이라, 배치 4개 각각의 squared error가 동시에 계산됩니다.)
     1. 이하 동일
  3. (`.sum()`) 배치 전체를 하나의 값으로 합산하는 연산이 실행됩니다.
     1. 새로운 Value(`sum` 결과)가 생성되고, `_prev`에 `squared` 하나만 자식으로 기록됩니다.
     2. `_backward`는 "합산에 참여한 모든 원소에게 같은 gradient를 그대로 나눠준다"는 규칙으로 정의만 해둡니다.
  4. (`* (1.0 / batch_size)`) 스칼라 곱셈이 실행되어 평균을 냅니다. 이 스텝의 최종 loss가 됩니다.

**backward**

```python
# Before — 4단계
loss.backward()

# After 1, After 2 — 5단계 (동일)
loss.backward()
```

- 4, 5단계 backward 계산 공식은 동일한 방식입니다.
- backward 실행 시, Value 동작 과정
  1. `loss.backward()`가 호출되면, 위상 정렬(topological sort)로 계산 그래프를 먼저 만듭니다.
     1. `loss`(최종 노드)부터 시작해서, `_prev`를 따라 자식들을 스택에 쌓습니다.
     2. 각 노드의 자식이 전부 처리된 뒤에야 그 노드 자신을 `topo_order`에 추가합니다. (4단계와 동일한 절차)
  2. `self.grad`(loss 자신의 gradient)를 1.0으로 설정합니다. (After 1은 `[[1.0]]`, After 2는 `np.ones_like(self.data)` — 스칼라 1이 아니라 loss의 모양(shape)에 맞춘 1로 채워진 배열이라는 점이 4단계와 다릅니다.)
  3. `topo_order`를 뒤집어서(`reversed`), 출력에 가까운 노드부터 순서대로 `_backward()`를 호출합니다.
     1. (`sum`의 `_backward`) 4단계에는 없던 신규 연산입니다. 합산에 참여했던 모든 원소에게, 자신(합산 결과)의 gradient를 그대로 복사해서 나눠줍니다.
     2. (`__mul__`, `__add__`, `sigmoid`의 `_backward`) 4단계와 동일한 계산 방식이지만, 원소 하나가 아니라 행렬 전체에 대해(After 1은 반복문으로, After 2는 NumPy 배열 연산으로) 한 번에 적용됩니다.
     3. (`__matmul__`의 `_backward`) 4단계에는 없던 신규 연산입니다. `dL/dA = dL/dC @ B.T`, `dL/dB = A.T @ dL/dC` 공식으로, 행렬곱에 참여했던 두 입력 각각의 gradient를 계산합니다. After 1은 이 공식을 3중 `for` 반복문으로, After 2는 NumPy 행렬곱 두 번으로 계산합니다.
  4. 모든 노드의 `_backward()` 호출이 끝나면, `W1.grad`, `b1.grad`, `W2.grad`, `b2.grad`에 배치 4개 샘플 전체를 반영한 gradient가 채워집니다.

**update**

```python
# Before — 4단계
for p in self.parameters():
    p.data -= learning_rate * p.grad

# After 1 — 5단계(순수 Python)
for p in self.parameters():
    rows, cols = p.shape
    for i in range(rows):
        for j in range(cols):
            p.data[i][j] -= learning_rate * p.grad[i][j]

# After 2 — 5단계(NumPy)
for p in self.parameters():
    p.data -= learning_rate * p.grad
```

- 4단계는 파라미터 하나하나가 스칼라라서, `p.data -= learning_rate * p.grad` 한 줄이 그 파라미터 값 하나를 갱신했습니다.
- 5단계는 파라미터 하나하나가 행렬이라서, 갱신 대상이 "값 하나"에서 "행렬 안의 원소 전체"로 늘어났습니다.
  - After 1(순수 Python)은 `rows`, `cols`를 순회하는 이중 `for` 반복문으로 원소 하나하나 갱신합니다.
  - After 2(NumPy)는 4단계와 코드가 완전히 동일합니다. `p.data`, `p.grad`가 NumPy 배열이라, `-=` 연산자 자체가 배열 전체에 대해 원소별로 알아서 적용됩니다.
- update 실행 시, Value 동작 과정
  1. `p.data -= learning_rate * p.grad`가 실행됩니다.
     1. `p`는 Value 인스턴스지만, `p.data`와 `p.grad`는 둘 다 순수 float(After 1은 중첩 리스트, After 2는 NumPy 배열)입니다.
     2. `learning_rate * p.grad`는 Value의 특수 메서드 연산을 사용하지 않습니다. After 1은 Python의 리스트 순회, After 2는 NumPy의 원소별 곱셈이 대신 계산합니다.
     3. `... -= ...`도 마찬가지로 Value의 특수 메서드 연산을 사용하지 않습니다.
  2. update 단계에서는 연산자 오버로딩을 거치지 않습니다. 4단계에서 확인한 것과 동일한 이유(파라미터 갱신 자체가 새로운 계산 그래프 노드로 기록되는 것을 막기 위함)로, 이번에도 `.data`, `.grad`라는 원시 값을 직접 조작하는 방식을 유지합니다.

## 구현

- [PureValue 클래스 (순수 Python)](./value_pure.py)
- [NumpyValue 클래스 (NumPy)](./value_numpy.py)
- [순수 Python 배치 MLP](./mlp_pure.py)
- [NumPy 배치 MLP](./mlp_numpy.py)

## 결과 정리

**표 1. Value 특수 메서드(add, mul, matmul, sigmoid) 검증 결과**
| 검증 항목 | 대상 | 결과 |
|---|---|---|
| 행렬곱(`__matmul__`) backward vs 수치 미분 | After 1 (순수 Python) | 6개 원소 전부 일치 |
| 행렬곱(`__matmul__`) backward vs 수치 미분 | After 2 (NumPy) | 6개 원소 전부 일치 (`A.grad = [[1. 1. 2.], [1. 1. 2.]]`) |
| 브로드캐스팅(broadcasting)된 bias의 gradient 합산 | After 2 (NumPy) | `bias.grad = [[4. 4.]]` (배치 크기 4만큼 정확히 합산) |
| 스칼라 하위 호환성 (4단계와 동일한 결과) | After 2 (NumPy) | `w.grad=-18.0, b.grad=-6.0` (4단계와 완전 동일) |

**표 2. After 1(순수 Python)과 After 2(NumPy) 동일 입력 결과 대조**
| 항목 | After 1 (순수 Python) | After 2 (NumPy) |
|---|---|---|
| loss | 0.454090 | 0.454090 |
| W1.grad (4개 원소) | 전부 일치 | 전부 일치 |

**표 3. XOR 학습 결과 (10000 epoch, learning_rate=1.0)**
| 버전 | seed | 최종 MSE | 4개 케이스 정확도 |
|---|---|---|---|
| After 1 (순수 Python, BatchMLPPure) | 0 | 0.000318 | 4/4 |
| After 2 (NumPy, BatchMLP) | 0 | 0.000295 (초기화 알고리즘 차이로 미세한 차이) | 4/4 |

**표 4. 행렬 크기에 따른 순수 Python과 NumPy의 속도 역전**
| 실험 | 행렬/파라미터 규모 | 결과 |
|---|---|---|
| 200×200 행렬곱 단독 | 대규모 | NumPy가 73배 빠름 |
| XOR 학습 전체 루프 (2000 epoch) | 소규모 (파라미터 9개) | 순수 Python이 1.6배 빠름 (167.24ms vs 267.04ms) |

## 결론 및 한계

- 5단계에서 확보한 것은 "여러 데이터를 동시에 처리하는 능력"입니다. `X`의 각 행(row)은 서로 완전히 독립적인 샘플이고, `X @ W1`이라는 한 번의 행렬곱이 이 독립적인 샘플들을 병렬로 계산해줬을 뿐입니다. XOR 데이터의 4개 샘플 사이에는 애초에 순서(order)나 선후 관계가 없습니다. `(0,1)`을 계산한 뒤 `(1,0)`을 계산하든, 순서를 완전히 뒤바꿔도 결과는 동일합니다.
- 하지만 언어, 시계열(time series)처럼 순서 자체가 의미를 가지는 데이터는 이 가정이 한계가 있습니다. 문장에서 단어의 순서를 바꾸면 의미가 달라지고, 특정 시점의 값이 그 이전 시점의 값에 의존합니다. 지금까지 만든 배치 MLP는 "배치 안의 샘플들이 서로 무관하다"는 전제 위에서 설계됐기 때문에, 이런 순서 의존성(sequential dependency)을 표현할 방법이 없습니다.
- 구체적으로, 배치 MLP의 forward(`X @ W1 + b1`)는 배치 차원(batch dimension)의 각 행을 동시에, 서로 참조하지 않고 계산합니다. 반면 순서가 있는 데이터를 처리하려면, `t`번째 원소를 계산할 때 `t-1`번째까지의 정보를 어떤 형태로든 전달받아야 합니다. 이 "이전 시점의 정보를 다음 시점으로 넘겨주는 통로"가 지금 구조에는 존재하지 않습니다.
- 이게 6단계(RNN cell)가 필요한 이유입니다. RNN은 hidden state(은닉 상태)라는 값을 두고, 매 시점(time step)마다 "현재 입력"과 "이전 시점의 hidden state"를 함께 받아 새로운 hidden state를 만들어낸다. 이번 5단계에서 확보한 행렬곱·브로드캐스팅 연산은 그대로 재사용되지만, 그 위에 "시간 축을 따라 값을 전달하는 구조"가 새로 추가되는 것이 6단계의 핵심입니다.
