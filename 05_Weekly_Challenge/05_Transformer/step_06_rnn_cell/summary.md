# RNN cell

## 배경

- 5단계 배치 MLP는 배치 안의 샘플들이 서로 무관하다는 전제 위에 설계됐습니다. XOR 데이터의 4개 샘플 사이에는 순서(order)나 선후 관계가 없어서, X @ W1이라는 한 번의 행렬곱으로 병렬 계산해도 문제가 없었습니다.
- 하지만 언어, 시계열(time series)처럼 순서 자체가 의미를 가지는 데이터는 이 전제가 깨집니다. 특정 시점의 값이 그 이전 시점의 값에 의존하는데, 배치 MLP의 forward는 배치 차원의 각 행을 서로 참조 없이 동시에 계산하므로, 이 순서 의존성(sequential dependency)을 표현할 방법이 없습니다.
- 6단계는 이 문제를 hidden state(은닉 상태)로 해결합니다. RNN cell은 매 시점(time step)마다 "현재 입력"과 "이전 시점의 hidden state"를 함께 받아 새로운 hidden state를 만들어냅니다.

## 동작 원리

- 5단계는 순서가 없을 뿐 아니라, RNN cell과 비교하기에 부적합한 예시입니다. 따라서, 이해하기 쉽도록 비교하기 위해 `MemorylessBaseline`를 비교 클래스로 구현했습니다.

**forward**

```python
# Before — 5단계(mlp_numpy.py)
def forward(self, X: np.ndarray) -> NumpyValue:
    X_val = NumpyValue(X)
    Z1 = X_val @ self.W1 + self.b1
    H = Z1.sigmoid()
    Z2 = H @ self.W2 + self.b2
    O = Z2.sigmoid()
    return O

# After 1 — 6단계(train_parity.py, MemorylessBaseline — hidden state 없음)
def forward(self, sequence: list) -> list:
    outputs = []
    for x_t in sequence:
        x_val = NumpyValue(x_t)
        y_t = (x_val @ self.W + self.b).sigmoid()
        outputs.append(y_t)
    return outputs

# After 2 — 6단계(rnn_cell.py, RNNCell — hidden state 있음)
def step(self, x_t: NumpyValue, h_prev: NumpyValue) -> tuple:
    h_t = (x_t @ self.Wxh + h_prev @ self.Whh + self.bh).sigmoid()
    y_t = (h_t @ self.Why + self.by).sigmoid()
    return y_t, h_t

def forward(self, sequence: list) -> list:
    batch_size = sequence[0].shape[0]
    h = self.init_hidden(batch_size)
    outputs = []
    for x_t in sequence:
        x_val = NumpyValue(x_t)
        y_t, h = self.step(x_val, h)
        outputs.append(y_t)
    return outputs
```

- 5단계는 배치 전체(`X`)를 받아서 한 번의 행렬곱으로 4개 샘플을 동시에 계산하고 끝났습니다. 샘플들은 서로 완전히 무관합니다.
- 6단계 After 1(`MemorylessBaseline`)은 `sequence`를 시점별로 순회하는 반복문을 도입했지만, 매 시점 `x_t @ self.W + self.b`만 계산할 뿐 이전 시점의 결과를 전혀 참조하지 않습니다.
- 6단계 After 2(`RNNCell`)는 반복문 안에서 `step`을 호출하며, 이전 시점의 hidden state(`h`)를 매번 다음 계산에 전달합니다.
- step 실행 시, NumpyValue 동작 과정
  1. `self.step(x_val, h)`가 호출되면, 인자로 받은 `h`(파이썬 변수가 가리키는 현재 시점의 `NumpyValue` 객체)가 함수 안에서 `h_prev`라는 이름으로 넘겨받아집니다.
  2. (`h_prev @ self.Whh`) 곱셈 연산(`__matmul__`)이 실행되어 새로운 `NumpyValue`가 생성됩니다.
     1. 이 노드는 초기화 과정을 통해 자신을 만드는 데 쓰인 입력(`h_prev`, `self.Whh`)을 자식(`_children`)으로 삼아 `self._prev`에 기록합니다. 즉 지금 넘겨받은 `h_prev` 객체 자체가, 새로 만들어질 `h_t`의 자식으로 붙잡힙니다.
  3. (`x_t @ self.Wxh`, `+ self.bh`, `.sigmoid()`) 나머지 연산이 이어지고, 최종적으로 `h_t`가 반환됩니다.
  4. `step`을 호출한 쪽(`forward`)에서 `y_t, h = self.step(x_val, h)`로 반환값을 받으면, 파이썬 변수 `h`는 이제 새로 만들어진 `h_t` 객체를 가리키도록 재할당됩니다. 이전 시점의 `h`(2번에서 `h_prev`로 넘겨졌던 객체)는 변수 이름으로는 더 이상 접근할 수 없게 됩니다.
  5. 다만 2번에서 이미 새 `h_t`의 `_prev`에 이전 `h` 객체가 자식으로 기록되어 있으므로, 이 객체는 메모리에서 사라지지 않고 그래프 안에 그대로 남아있습니다. 파이썬 변수 이름의 재할당과 계산 그래프 안에서의 참조 유지는 서로 다른 층위의 일입니다.
  6. 이 과정이 매 시점 반복되면서, 각 시점의 `h_t`가 바로 이전 `h_t`를 자식으로 붙잡는 관계가 사슬처럼 이어집니다. `backward()`가 최종 `loss`에서 시작해 `_prev`를 따라 거슬러 올라갈 때, 이 사슬을 타고 `t=0`의 최초 hidden state까지 정확히 도달할 수 있습니다.
- `step`은 `x_t`뿐 아니라 `h_prev`(이전 시점 hidden state)를 함께 받아, `h_prev @ self.Whh`라는 항으로 이번 시점의 계산에 반영합니다. 이 `step`이 `forward`의 반복문 안에 들어가면서, `h`라는 변수가 매 반복마다 그 시점에 새로 계산된 값으로 교체되고, 다음 반복에 `h_prev`로 전달됩니다. 그 결과 `t=1`의 계산에는 `t=0`의 결과가, `t=2`의 계산에는 `t=1`의 결과가 관여하게 되어, 전체 시퀀스가 하나로 연결된 계산이 됩니다.

**loss**

```python
# Before — 5단계
def compute_loss(self, O: NumpyValue, y_true: np.ndarray) -> NumpyValue:
    diff = O + (-1 * y_true)
    squared = diff * diff
    batch_size = y_true.shape[0]
    return squared.sum() * (1.0 / batch_size)

# After 1, After 2 — 6단계 (동일한 형태)
def compute_loss(self, outputs: list, targets: list) -> NumpyValue:
    total = None
    count = 0
    for y_t, target_t in zip(outputs, targets):
        diff = y_t + (-1 * target_t)
        squared = diff * diff
        total = squared if total is None else total + squared
        count += target_t.shape[0]
    return total.sum() * (1.0 / count)
```

**backward**

```python
# Before — 5단계
loss.backward()

# After 1, After 2 — 6단계 (동일)
loss.backward()
```

**update**

```python
# Before — 5단계
for p in self.parameters():
    p.data -= learning_rate * p.grad

# After 1, After 2 — 6단계 (동일)
for p in self.parameters():
    p.data -= learning_rate * p.grad
```

## 구현

- [MemorylessBaseline (hidden state 없음, After 1)](./train_parity.py)
- [RNNCell (hidden state 있음, After 2)](./rnn_cell.py)
- [누적 패리티(running parity) 데이터 생성](./parity_task.py)
- [학습 및 정확도 비교 실행 스크립트](./train_parity.py)
- [BPTT gradient 검증 테스트](./test_rnn_cell.py)

## 결과 정리

**표 1. RNNCell vs MemorylessBaseline — 누적 패리티 과제 학습 결과**
| 모델 | hidden state | 1000 epoch 후 정확도 | 특이사항 |
| --- | --- | --- | --- |
| RNNCell | 있음 | 95.8% | epoch이 진행될수록 계속 개선 |
| MemorylessBaseline | 없음 | 56.2% | 400 epoch 이후 완전히 정체(더 이상 개선 없음) |

## 결론 및 한계

- Whh는 모든 시점에서 동일한 파라미터가 재사용되는데, 4단계에서 확인했던 "공유 파라미터의 gradient는 여러 경로의 합"이라는 원리가 여기서는 "여러 시점의 합"으로 확장되었습니다. 지금까지의 `NumpyValue.backward()` (위상 정렬 기반 역순회)를 코드 한 줄도 바꾸지 않았는데, 시퀀스 4개 시점을 거치는 계산 그래프에 그대로 적용해서 정확한 gradient가 나왔습니다. 이게 automatic differentiation의 일반성을 보여주는 지점입니다.
- 다만, RNN cell 구조 자체에는 두 가지 한계가 있습니다.
  - **장거리 의존성(long-range dependency) 약화**: `t=0`의 정보가 `t=T`에 영향을 미치려면, backward 시 `sigmoid` 미분(`s*(1-s)`, 최댓값 0.25)과 `Whh`가 시퀀스 길이만큼 반복해서 곱해집니다. 1보다 작은 값이 계속 곱해지므로, 시퀀스가 길어질수록 초반 시점의 gradient가 지수적으로 작아지는 vanishing gradient(기울기 소실) 문제가 구조적으로 발생합니다. 이번 6단계는 시퀀스 길이 6으로 짧게 유지해서 이 문제가 드러나지 않았지만, 로드맵(`docs.md`)의 6단계 한계 항목에 적힌 문제가 바로 이것입니다.
  - **병렬화 불가**: 5단계 배치 MLP는 샘플들이 서로 무관해서 `X @ W1` 한 번의 행렬곱으로 전부 동시에 계산할 수 있었습니다. 반면 RNN cell은 `h_t`가 `h_{t-1}`을 반드시 필요로 하므로, `step()`을 시간 순서대로 하나씩 실행해야 합니다. 시퀀스가 길어질수록 이 순차 실행 자체가 학습 속도의 병목이 됩니다.
- 이 두 한계가 7단계(attention score)로 넘어가는 이유입니다. attention은 hidden state를 시간 순서대로 전달하는 사슬 없이, 각 위치가 다른 모든 위치와 직접 관계를 계산합니다. 정보가 시점 수만큼 사슬을 거치지 않고 한 번에 오가므로 장거리 의존성이 약화되지 않고, 각 위치의 계산이 서로를 기다리지 않으므로 병렬화도 가능해집니다.
