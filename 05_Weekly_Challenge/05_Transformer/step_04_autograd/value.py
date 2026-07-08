"""
연산자 오버로딩(operator overloading)을 이용한 자동 미분 최소 구현.

핵심 아이디어:
    a + b, a * b 같은 일반 연산을 수행하는 시점에,
    - 그 연산의 결과값
    - 그 연산에 어떤 입력(들)이 쓰였는지
    - 그 연산의 local gradient(지역 그래디언트) 계산 방법
    을 함께 기록해둔다. 이후 backward()가 이 기록을 역순으로 순회하며
    chain rule(연쇄 법칙)을 자동으로 적용한다.

지원 연산: 덧셈(+), 곱셈(*), 거듭제곱(**), 뺄셈(-), sigmoid.
MLP를 재구현하는 데 필요한 만큼만 확장했다.
"""

import math


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = ''):
        self.data = data
        self.grad = 0.0  # 이 값에 대한 최종 loss의 gradient (처음엔 0으로 초기화)

        # 계산 그래프(computational graph) 구성을 위한 내부 정보
        self._prev = set(_children)  # 이 Value를 만드는 데 쓰인 입력들 (부모 노드)
        self._op = _op              # 어떤 연산으로 만들어졌는지 (디버깅용)
        self._backward = lambda: None  # 이 연산의 local gradient를 부모에게 전달하는 함수

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  (덧셈의 미분)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self (other가 Value가 아닌 일반 숫자일 때)
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data  (곱셈의 미분)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "지수는 상수(int/float)만 지원한다"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(self^n)/d(self) = n * self^(n-1)  (power rule, 거듭제곱 법칙)
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z))
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        위상 정렬(topological sort)로 계산 그래프를 역순 순회하며
        각 노드의 _backward()를 호출해 gradient를 전파한다.
        """
        topo_order = []
        visited = set()

        # 방법 1 - while loop (Stack 명시)
        stack = [(self, False)] # (node, 자식까지 처리 완료했는지)
        while stack:
            node, children_done = stack.pop()
            if node in visited:
                continue
            if children_done:
                visited.add(node)
                topo_order.append(node)
            else:
                stack.append((node, True))
                for child in node._prev:
                    if child not in visited:
                        stack.append((child, False))
        
        # 방법 2 - Recursive loop (Stack 암시)
        # def build_topo(node):
        #     if node not in visited:
        #         visited.add(node)
        #         for child in node._prev:
        #             build_topo(child)
        #         topo_order.append(node)
        # build_topo(self)

        self.grad = 1.0  # dL/dL = 1 (자기 자신에 대한 미분은 항상 1)
        for node in reversed(topo_order):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"