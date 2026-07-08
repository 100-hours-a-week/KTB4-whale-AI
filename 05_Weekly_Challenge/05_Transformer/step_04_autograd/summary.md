# autograd

## 배경

- 3단계에서 `MLP.backward()`는 파라미터 9개에 대해 chain rule을 모두 수기로 나열해야 했습니다. 이 과정은 뉴런과 레이어가 늘어날수록 계속됩니다. Transformer처럼 파라미터가 수백만 개인 모델에서는 이런 방식이 사실상 불가능합니다.
- 4단계에서는 이 문제를 "연산이 일어나는 시점에 그 연산의 local gradient(지역 그레디언트)를 자동으로 기록해두는" Operator Overloading 방식으로 해결하는 과정입니다. 그래서 forward()만 정의하면 backward는 자동으로 계산됩니다.

## 동작 원리

- 3단계의 수동 backward에서, 4단계 실제 PyTorch(torch.Tensor)로 바로 넘어가지 않고 중간 단계를 하나 더 거칩니다. Operator Overloading 자체의 원리(연산 시점에 local gradient를 자동 기록하는 메커니즘)를 직접 구현해보지 않으면, torch.Tensor의 동작이 블랙박스로 남기 때문입니다.
- 이 과정을 이해하기 위해 Operator Overloading 메커니즘을 Value라는 자체 클래스로 직접 구현(After 1)하고, 그 다음 실제 PyTorch(torch.Tensor)로 동일한 계산을 재현(After 2)하는 순서로 진행하겠습니다.

### Neuron, Value 기반 세부 동작 원리

```python
# After 1 — 4단계(mlp_autograd.py, 자체 Value 엔진)
class Neuron:
    def __init__(self, num_inputs: int, seed: int):
        random_generator = random.Random(seed)
        self.weights = [Value(random_generator.uniform(-1.0, 1.0)) for _ in range(num_inputs)]
        self.bias = Value(random_generator.uniform(-1.0, 1.0))
```

- 실행 코드를 살펴보면, 최초 MLP 클래스가 선언될 때 Neuron 클래스가 선언되어 weights, bias가 hidden1, hidden2, output에 할당됩니다. 3단계에서는 이 weights, bias가 순수 float(랜덤값)이었지만 4단계에서는 Value 인스턴스로 생성됩니다.
- Value로 생성된 weights, bias는 생성 직후에는 계산 그래프의 leaf node(부모가 없는 노드)일 뿐입니다. 이후 forward()에서 이 weights, bias가 덧셈, 곱셈, sigmoid 연산에 사용될 때마다, 그 연산에 대응하는 Value의 특수 메서드가 호출되면서 각 연산의 local gradient 계산 방법이 계산 그래프에 자동으로 기록됩니다.

**forward**

```python
# Before — 3단계(mlp.py)
def forward(self, x1: float, x2: float) -> tuple[float, dict]:
    h1, z_h1 = self.hidden1.forward([x1, x2])
    h2, z_h2 = self.hidden2.forward([x1, x2])
    o, z_o = self.output.forward([h1, h2])
    cache = {'x1': x1, 'x2': x2, 'h1': h1, 'z_h1': z_h1, 'h2': h2, 'z_h2': z_h2, 'o': o, 'z_o': z_o}
    return o, cache

# After 1 — 4단계(mlp_autograd.py, 자체 Value 엔진)
def forward(self, x1: float, x2: float) -> Value:
    h1 = self.hidden1.forward([x1, x2])
    h2 = self.hidden2.forward([x1, x2])
    o = self.output.forward([h1, h2])
    return o

# After 2 — 4단계(pytorch_mlp.py, 실제 PyTorch)
def forward(self, x1: float, x2: float) -> torch.Tensor:
    x1_t, x2_t = torch.tensor(x1), torch.tensor(x2)
    h1 = torch.sigmoid(self.w1_1 * x1_t + self.w1_2 * x2_t + self.b1)
    h2 = torch.sigmoid(self.w2_1 * x1_t + self.w2_2 * x2_t + self.b2)
    o = torch.sigmoid(self.v1 * h1 + self.v2 * h2 + self.c)
    return o
```

- 3단계는 backward에 필요한 중간값을 cache 딕셔너리로 모아서 반환했습니다.
- 4단계에서는 `Value`(After 2에서는 `torch.Tensor`)가 연산자 오버로딩으로 자기 자신이 만들어진 경로(계산 그래프, computational graph)를 내부적으로 기억해둡니다. 그래서 사람이 별도로 중간값을 모아 cache로 전달할 필요가 없습니다.
- forward 실행 시, Value의 동작 과정
  1. (`w * x`) 앞서 만들어진 Value 인스턴스에 곱셈 연산(`__mul__`)이 실행됩니다.
     1. 새로운 Value(`out`)가 생성됩니다.
        1. out은 초기화 과정을 통해 자신을 만드는 데 쓰인 입력(w, x)을 자식(`_children`)으로 삼아 `self._prev`에 기록합니다.
        2. 이를 통해 w, x가 out의 자식이 됩니다. (`w.__mul__(x)`이므로, w는 `self`, x는 `other`)
        3. 또한, out은 초기화 과정을 통해 `_op`를 `self._op`에 기록합니다.
     2. `out._backward`로 이때 실행할 \_backward(backward 연산의 일부)가 무엇인지 기록합니다.
        1. 해당 과정은 정의만 해둡니다.
        2. 실제 연산은 추후 backward() 메서드가 실행될 때 진행합니다.
     3. out 객체를 반환합니다.
  2. (`sum(...)`) 앞서 만들어진 Value 인스턴스의 덧셈 연산(`__add__`)이 실행됩니다.
     1. 이하 동일
  3. (`z.sigmoid()`) 앞서 만들어진 Value 인스턴스의 sigmoid 연산이 실행됩니다.
     1. 이하 동일

**loss**

```python
# Before — 3단계(mlp.py)
def compute_loss(self, o: float, y_true: float) -> float:
    return (o - y_true) ** 2

# After 1, After 2 — 4단계 (별도 메서드 없음)
loss = (o - y_true) ** 2
```

- 3단계는 `compute_loss`라는 전용 메서드가 필요했습니다.
- 4단계에서는 `(o - y_true) ** 2` 자체가 Value의 연산자로 오버로딩을 거치는 텐서 연산이라, 이 계산 자체가 계산 그래프에 자동으로 편입됩니다.
- loss 실행 시, Value의 동작 과정
  1. (`o - y_true`) 앞서 만들어진 Value 인스턴스의 뺄셈 연산(`__sub__`)이 실행됩니다.
     1. isinstance로 y_true는 (float이므로) False를 반환하고, 조건 분기 처리로 `-other`가 실행됩니다.
     2. `-y_true`는 순수 float이므로, float의 부호 반전으로 -1.0이 계산됩니다. (`__neg__` 사용 아님)
     3. 뺄셈 연산 메서드는 덧셈 연산으로 구현되므로, 덧셈 연산(`__add__`)이 실행됩니다.
        1. isinstance로 -1.0은 (float이므로) False를 반환하고, 조건 분기 처리로 `Value(-1.0)`로 클래스화됩니다. (promote, 승격)
        2. 이하 동일
     4. 연산 결과를 반환합니다.
  2. (`(...) ** 2`) 앞서 만들어진 Value 인스턴스의 제곱 연산(`__pow__`)이 실행됩니다.
     1. 이하 동일

**backward**

```python
# Before — 3단계(mlp.py), 60줄 이상 (출력층 3개 + 은닉층 2개 x 3개 파라미터,
#          각 파라미터마다 chain rule을 손으로 나열)
def backward(self, cache: dict, y_true: float) -> dict:
    ... # (생략, 3단계 summary.md 참고)

# After 1 — 4단계(mlp_autograd.py)
loss.backward()

# After 2 — 4단계(pytorch_mlp.py)
loss.backward()
```

- 3단계의 backward() 전체가, 4단계에서는 loss.backward() 한 줄로 대체됩니다.
- 4단계에서는 계산 그래프를 위상 정렬(topological sort)로 역순 순회하며, 각 노드가 자신의 local gradient를 이용해 chain rule을 적용하는 과정입니다.
- backward 실행 시, Value 동작 과정
  1. (위상 정렬 준비) 정렬 결과를 담을 리스트(topo_order), 이미 처리한 노드 집합(visited), 처리 대기 중인 노드를 담는 리스트(stack)를 초기화합니다.
  2. (스택 순회) stack이 빌 때까지 반복합니다.
     1. stack.pop()으로 가장 최근에 넣은 노드를 하나 꺼냅니다.
     2. 이미 visited에 있는 노드라면 건너뜁니다. (중복 처리 방지)
     3. children_done이 True(이미 한 번 노드를 만나 자식들을 스택에 넣어둔 상태)라면:
        1. visited에 이 노드를 추가합니다.
        2. topo_order에 이 노드를 추가합니다.(이 시점에 추가되었다는 것은 이 노드의 모든 자식은 이미 topo_order에 들어가 있다는 의미)
     4. children_done이 False(처음 만난 노드)라면:
        1. 같은 노드를 (node, True)로 다시 스택에 넣습니다.(자식을 다 처리하면 다시 나를 처리해야 함을 표시)
        2. `node._prev`(이 노드를 만드는 데 쓰인 자식들, 예: `w * x`의 w, x)를 순회하며, 아직 visited에 없는 자식을 (child, False)로 스택에 넣습니다.
  3. (역순 실행 준비) self.grad = 1.0으로 설정합니다. `dL/dL = 1`이라는 자기 자신에 대한 미분값이며, 모든 gradient 전파의 시작점입니다.
  4. (gradient 전파) topo_order를 reversed하여 순회합니다.
     1. topo_order는 자식이 먼저, 부모가 나중에 쌓인 순서이므로, 이를 뒤집으면 "부모(출력에 가까운 노드)"부터 "자식(입력에 가까운 노드)"까지 순회하는 형태가 됩니다. 즉, loss에서 시작해서 weights, bias 방향으로 흘러가는 순서입니다.
     2. 각 노드의 `node._backward()`를 호출합니다. 이 호출이 forward 시점에 정의만 해뒀던 함수를 실제로 실행하는 지점입니다.
     3. `_backward()` 내부에서는 `self.grad += ...` 형태로 gradient가 누적(accumulate)됩니다. 한 노드가 여러 경로에서 동시에 사용된 경우, 그 모든 경로의 gradient를 합산해야 하기 때문입니다.

**update**

```python
# Before — 3단계(mlp.py)
def update(self, gradients: dict, learning_rate: float):
    dL_dv1, dL_dv2, dL_dc = gradients['output']
    self.output.weights[0] -= learning_rate * dL_dv1
    ... # (생략, 9개 파라미터 각각 수동 갱신)

# After 1 — 4단계(mlp_autograd.py)
for p in self.parameters():
    p.data -= learning_rate * p.grad

# After 2 — 4단계(pytorch_mlp.py)
with torch.no_grad():
    for p in self.parameters():
        p -= learning_rate * p.grad
```

- 3단계는 9개 파라미터 각각을 이름으로 지정하여 갱신했습니다.
- 4단계는 `parameters()`가 반환하는 리스트를 순회하며 동일한 갱신 공식(`p -= learning_rate * p.grad`)을 일괄 적용합니다. 파라미터 개수가 늘어나도 update 코드 자체는 바뀌지 않습니다.
- 4단계는 `zero_grad()`가 각 학습 스텝 시작 전에 필요합니다. `grad`가 `+=`로 누적되는 구조이기 때문입니다. 이전 스텝의 gradient가 남아있으면 현재 스텝 계산과 섞입니다.
- update 실행 시, Value 동작 과정
  1. `p.data -= learning_rate * p.grad`이 실행됩니다.
     1. `p`는 Value 인스턴스지만, `p.data`와 `p.grad`는 둘 다 순수 float입니다.
     2. 따라서, `learning_rate * p.grad`는 Value의 특수 메서드 연산을 사용하지 않습니다.
     3. 마찬가지로 `... -= ...`도 Value의 특수 메서드 연산을 사용하지 않습니다.
- update 단계에서는 연산자 오버로딩을 거치지 않습니다. 이 점은 의도된 설계입니다. 왜냐하면 `p -= learning_rate * p.grad`처럼 Value 객체 `p` 자체를 연산 대상으로 삼았다면, 특수 메서드 연산이 호출되면서 파라미터 갱신이 새로운 계산 그래프 노드에 다시 기록되어 버리기 때문입니다. 같은 이유로 4단계-2에서 `torch.no_grad()`을 사용하면, 컨텍스트로 감싸서 그래프 추적을 끌 수 있습니다.

## 구현

- [Value 클래스](./value.py)
- [Value 기반 MLP](./mlp_pure_autograd.py)
- [PyTorch 기반 MLP](./mlp_pytorch_autograd.py)

## 결과 정리

**표 1. 3단계와 4단계(After 1, After 2)의 gradient 일치 검증 (seed=7, x1=1.0, x2=0.0, y_true=1.0)**
| 파라미터 | 3단계 (수동) | After 1 (Value) | After 2 (PyTorch) |
|---|---|---|---|
| v1 | -0.143725 | -0.143725 | -0.143725 |
| v2 | -0.063457 | -0.063457 | -0.063457 |
| c | -0.294889 | -0.294889 | -0.294889 |
| w1_1 | 0.005451 | 0.005451 | 0.005451 |
| b1 | 0.005451 | 0.005451 | 0.005451 |
| w2_1 | 0.012619 | 0.012619 | 0.012619 |
| b2 | 0.012619 | 0.012619 | 0.012619 |

**표 2. XOR 학습 결과 (10000 epoch, learning_rate=1.0)**
| 버전 | seed | 최종 MSE | 4개 케이스 정확도 |
|---|---|---|---|
| 3단계 (수동) | 0 | 0.000063 | 4/4 |
| After 1 (Value) | 0 | 0.000063 (3단계와 완전 동일) | 4/4 |
| After 2 (PyTorch) | 0 | 0.000066 (RNG 알고리즘 차이로 미세한 차이) | 4/4 |

- 표 1에 `w1_2`, `w2_2`가 없는 이유는, 검증에 쓰인 입력값이 `x2=0.0`이라 두 파라미터의 gradient가 항상 정확히 0이 되기 때문입니다(`w1_2`, `w2_2`는 `x2`에 곱해지는 가중치라, `gradient = (값) * x2 = 0`). 이는 3단계 `mlp.py` 자체를 이 입력으로 단독 실행해도 동일하게 나오는 결과이며, 4단계 두 엔진(Value, PyTorch)의 오류가 아니라 이 테스트 입력값의 성질입니다.

## 결론 및 한계

- 3단계에서 파라미터 9개에 대해 수기로 유도했던 chain rule이 연산자 오버로딩만으로 자동 계산됐고, 그 결과가 수동 backward, PyTorch 양쪽과 소수점 6자리까지 일치했습니다. 이를 통해 forward()만 정의하면 backward는 특별히 작성할 필요가 없어졌습니다.
- 다만 지금 구현한 Value는 스칼라(scalar, 숫자 하나) 단위로만 동작합니다. 실제 PyTorch의 torch.Tensor는 벡터, 행렬 단위의 배치 연산(batch operation)을 지원해 GPU 병렬화가 가능합니다. 스칼라 단위 연산으로는 이 성능 이점을 얻을 수 없습니다.
- GPU 병렬화는 CUDA 드라이버, 커널 연동이라는 별도 계층이 필요해서 직접 구현하기엔 범위가 큽니다. 따라서 학습을 위해 5단계에서 스칼라에서 벡터 및 행렬 단위로의 배치 연산 단위로 확장하는 경로를 Python, Numpy로 구현하는 과정을 진행합니다.
- 연산자 오버로딩 기반 자동 미분의 최소 구현 사례 참고 문서로는 Andrej Karpathy의 micrograd가 참고하기 좋으며, PyTorch의 실제 autograd 메커니즘은 공식문서를 참고했습니다.
  - [micrograd 바로가기](https://github.com/karpathy/micrograd)
  - [autograd 공식 문서 바로가기](https://pytorch.org/docs/stable/notes/autograd.html)
