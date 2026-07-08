"""
5단계 순수 Python 버전: NumPy 없이 중첩 리스트(nested list)로 행렬 연산을 구현.

tensor_value.py(NumPy 버전)와의 차이:
    tensor_value.py: self.data가 np.ndarray, 모든 연산이 NumPy의 벡터화된 C 코드로 계산됨
    matrix_value.py: self.data가 list[list[float]], 모든 연산이 Python for 반복문으로 계산됨

두 버전의 연산자 오버로딩 구조(__add__, __mul__, __matmul__, sigmoid, backward)는
동일하다 -- 차이는 "그 안에서 실제 숫자를 계산하는 방식"뿐이다.
"""

import math


def _zeros(rows: int, cols: int) -> list:
    return [[0.0] * cols for _ in range(rows)]


def _shape(data: list) -> tuple:
    return (len(data), len(data[0]) if data else 0)


class PureValue:
    def __init__(self, data: list, _children: tuple = (), _op: str = ''):
        # data: list[list[float]] 형태로 통일 (스칼라도 [[값]] 형태의 1x1 행렬로 취급)
        self.data = data
        self.shape = _shape(data)
        self.grad = _zeros(*self.shape)

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, PureValue) else PureValue(other)
        rows, cols = self.shape
        # brodcasting: other가 1행이면 self의 모든 행에 공통 적용 (bias 브로드캐스팅)
        out_data = [
            [self.data[i][j] + other.data[i % other.shape[0]][j] for j in range(cols)]
            for i in range(rows)
        ]
        out = PureValue(out_data, (self, other), '+')

        def _backward():
            for i in range(rows):
                for j in range(cols):
                    self.grad[i][j] += out.grad[i][j]
            # other가 브로드캐스팅됐다면(행 개수가 적으면), 여러 행의 gradient를 합산해야 함
            for i in range(rows):
                oi = i % other.shape[0]
                for j in range(cols):
                    other.grad[oi][j] += out.grad[i][j]
        out._backward = _backward
        return out

    def __mul__(self, other):
        """원소별(element-wise) 곱셈. other가 스칼라(순수 숫자)인 경우도 지원."""
        if isinstance(other, (int, float)):
            rows, cols = self.shape
            out_data = [[self.data[i][j] * other for j in range(cols)] for i in range(rows)]
            out = PureValue(out_data, (self,), '*scalar')

            def _backward():
                for i in range(rows):
                    for j in range(cols):
                        self.grad[i][j] += other * out.grad[i][j]
            out._backward = _backward
            return out

        other = other if isinstance(other, PureValue) else PureValue(other)
        rows, cols = self.shape
        out_data = [[self.data[i][j] * other.data[i][j] for j in range(cols)] for i in range(rows)]
        out = PureValue(out_data, (self, other), '*')

        def _backward():
            for i in range(rows):
                for j in range(cols):
                    self.grad[i][j] += other.data[i][j] * out.grad[i][j]
                    other.grad[i][j] += self.data[i][j] * out.grad[i][j]
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, PureValue) else PureValue(other)
        return self + (other * -1)

    def __matmul__(self, other):
        """
        행렬곱. C = A @ B
        dL/dA = dL/dC @ B.T
        dL/dB = A.T @ dL/dC
        전부 순수 Python 3중 for 반복문으로 계산한다 (NumPy 미사용).
        """
        m, k = self.shape
        k2, n = other.shape
        assert k == k2, f"행렬곱 모양 불일치: {self.shape} @ {other.shape}"

        out_data = _zeros(m, n)
        for i in range(m):
            for j in range(n):
                total = 0.0
                for p in range(k):
                    total += self.data[i][p] * other.data[p][j]
                out_data[i][j] = total

        out = PureValue(out_data, (self, other), '@')

        def _backward():
            # dL/dA = dL/dC @ B.T
            for i in range(m):
                for p in range(k):
                    total = 0.0
                    for j in range(n):
                        total += out.grad[i][j] * other.data[p][j]
                    self.grad[i][p] += total
            # dL/dB = A.T @ dL/dC
            for p in range(k):
                for j in range(n):
                    total = 0.0
                    for i in range(m):
                        total += self.data[i][p] * out.grad[i][j]
                    other.grad[p][j] += total
        out._backward = _backward
        return out

    def sum(self):
        rows, cols = self.shape
        total = sum(self.data[i][j] for i in range(rows) for j in range(cols))
        out = PureValue([[total]], (self,), 'sum')

        def _backward():
            g = out.grad[0][0]
            for i in range(rows):
                for j in range(cols):
                    self.grad[i][j] += g
        out._backward = _backward
        return out

    def sigmoid(self):
        rows, cols = self.shape
        s_data = [[1 / (1 + math.exp(-self.data[i][j])) for j in range(cols)] for i in range(rows)]
        out = PureValue(s_data, (self,), 'sigmoid')

        def _backward():
            for i in range(rows):
                for j in range(cols):
                    s = s_data[i][j]
                    self.grad[i][j] += s * (1 - s) * out.grad[i][j]
        out._backward = _backward
        return out

    def backward(self):
        topo_order = []
        visited = set()
        stack = [(self, False)]
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

        rows, cols = self.shape
        self.grad = [[1.0] * cols for _ in range(rows)]
        for node in reversed(topo_order):
            node._backward()

    def __repr__(self):
        return f"MatrixValue(shape={self.shape}, data={self.data})"