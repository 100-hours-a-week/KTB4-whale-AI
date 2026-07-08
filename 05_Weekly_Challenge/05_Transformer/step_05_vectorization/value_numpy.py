"""
5단계: 4단계 Value를 확장하여 스칼라뿐 아니라 NumPy 배열(행렬)도 다룰 수 있게 한다.

4단계 Value와의 차이:
    - data: float -> np.ndarray로 확장 (스칼라도 여전히 지원 -- np.float64로 자동 승격)
    - grad 초기화: 0.0 -> np.zeros_like(data) (배열 모양에 맞춰 0으로 채워진 배열)
    - sigmoid: math.exp -> np.exp (배열 전체에 대해 한 번에 계산, "벡터화"의 핵심)
    - __matmul__ 신규 추가: 행렬곱(matrix multiplication) 연산과 그 backward

이 확장으로, 지금까지 "샘플 1개 x 파라미터 1개"씩 반복문으로 처리하던 계산을
"샘플 여러 개 x 파라미터 여러 개"를 행렬 하나로 묶어 한 번에 처리할 수 있게 된다.
"""

import numpy as np


class NumpyValue:
    def __init__(self, data, _children: tuple = (), _op: str = ''):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, NumpyValue) else NumpyValue(other)
        out = NumpyValue(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, NumpyValue) else NumpyValue(other)
        out = NumpyValue(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, NumpyValue) else -other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = NumpyValue(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """
        행렬곱(matrix multiplication). Python의 @ 연산자에 대응하는 특수 메서드.

        C = A @ B
        dL/dA = dL/dC @ B.T
        dL/dB = A.T @ dL/dC
        """
        other = other if isinstance(other, NumpyValue) else NumpyValue(other)
        out = NumpyValue(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def transpose(self):
        """전치(transpose). backward는 gradient도 함께 전치해서 되돌려주면 된다."""
        out = NumpyValue(self.data.T, (self,), 'transpose')

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    def mean(self, axis: int = -1, keepdims: bool = True):
        """
        axis 방향의 평균. LayerNorm이 "각 위치(행)마다 feature 축 평균"을
        구하는 데 필요하다.

        mean = sum / n 이므로, sum의 backward(모든 원소에 동일하게 분배)에
        1/n을 추가로 곱하는 것과 같다.
        """
        n = self.data.shape[axis]
        m = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = NumpyValue(m, (self,), 'mean')

        def _backward():
            grad = out.grad
            if not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.ones_like(self.data) * grad / n
        out._backward = _backward
        return out

    def sqrt(self):
        """제곱근. d(sqrt(x))/dx = 1/(2*sqrt(x))"""
        s = np.sqrt(self.data)
        out = NumpyValue(s, (self,), 'sqrt')

        def _backward():
            self.grad += (0.5 / s) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        """나눗셈. self * (1/other)로 처리하되, other가 0에 가까울 때를 대비해
        직접 나눗셈 backward를 정의한다 (역수의 미분보다 수치적으로 안정적)."""
        other = other if isinstance(other, NumpyValue) else NumpyValue(other)
        out = NumpyValue(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.data.shape)
            other.grad += _unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1):
        """
        softmax(x_i) = e^(x_i) / sum_j(e^(x_j))   (axis 방향으로 정규화)

        지금까지의 sigmoid, transpose와 달리, softmax는 한 행(row) 안의
        모든 원소가 서로 영향을 주는 연산이다 -- 원소 하나를 바꾸면 그 행의
        나머지 원소들의 정규화 결과도 함께 바뀐다. 그래서 backward도
        원소별(element-wise)이 아니라 행 전체를 고려해서 계산해야 한다.

        backward 공식 (softmax의 Jacobian-vector product):
            dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
            (y = softmax(x), 같은 행 안에서 계산)
        """
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)  # overflow 방지
        exp = np.exp(shifted)
        y = exp / np.sum(exp, axis=axis, keepdims=True)

        out = NumpyValue(y, (self,), 'softmax')

        def _backward():
            weighted_sum = np.sum(out.grad * y, axis=axis, keepdims=True)
            self.grad += y * (out.grad - weighted_sum)
        out._backward = _backward
        return out

    @staticmethod
    def concat(values: list, axis: int = -1):
        """
        여러 NumpyValue를 axis 방향으로 이어 붙인다.
        multi-head attention에서 각 head의 출력을 하나로 합칠 때 사용한다.

        backward는 반대로, 이어 붙여진 gradient를 원래 조각들의 크기만큼 잘라서
        각자에게 되돌려주면 된다.
        """
        sizes = [v.data.shape[axis] for v in values]
        out_data = np.concatenate([v.data for v in values], axis=axis)
        out = NumpyValue(out_data, tuple(values), 'concat')

        def _backward():
            offset = 0
            for v, size in zip(values, sizes):
                slicer = [slice(None)] * out.grad.ndim
                slicer[axis] = slice(offset, offset + size)
                v.grad += out.grad[tuple(slicer)]
                offset += size
        out._backward = _backward
        return out

    def sum(self):
        """배치 전체의 loss를 스칼라 하나로 합산할 때 필요"""
        out = NumpyValue(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = NumpyValue(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1 - s)) * out.grad
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

        self.grad = np.ones_like(self.data)
        for node in reversed(topo_order):
            node._backward()

    def __repr__(self):
        return f"NumpyValue(shape={self.data.shape}, data={self.data})"


def _unbroadcast(grad, target_shape):
    """
    브로드캐스팅(broadcasting, 모양이 다른 배열끼리 자동으로 크기를 맞춰 연산하는 NumPy 기능)이
    forward에서 적용됐다면, backward에서는 그만큼 gradient를 합산해서 원래 모양으로 되돌려야 한다.
    예: bias(모양 (1,3))가 배치 전체(모양 (4,3))에 더해졌다면,
        bias의 gradient는 배치 4개에 대한 gradient를 전부 더한 값이어야 한다.
    """
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad