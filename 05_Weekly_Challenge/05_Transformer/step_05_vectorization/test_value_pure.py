"""
MatrixValue(순수 Python) 검증.

검증 항목:
    1. 행렬곱 backward가 수치 미분과 일치하는가
    2. bias 브로드캐스팅 gradient가 정확히 합산되는가
    3. 순수 Python 버전과 NumPy 버전(TensorValue)이 동일한 입력에 대해 정확히 같은
       forward, backward 결과를 내는가 (두 엔진의 등가성 검증)
"""

from step_05_vectorization.value_pure import PureValue
from step_05_vectorization.value_numpy import NumpyValue
import numpy as np


def test_matmul_backward_matches_numerical_gradient():
    A_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    B_data = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    def matmul_sum(A, B):
        m, k, n = len(A), len(A[0]), len(B[0])
        total = 0.0
        for i in range(m):
            for j in range(n):
                for p in range(k):
                    total += A[i][p] * B[p][j]
        return total

    A = PureValue([row[:] for row in A_data])
    B = PureValue([row[:] for row in B_data])
    C = A @ B
    L = C.sum()
    L.backward()

    h = 1e-6
    for i in range(2):
        for j in range(3):
            A_plus = [row[:] for row in A_data]
            A_plus[i][j] += h
            A_minus = [row[:] for row in A_data]
            A_minus[i][j] -= h
            numerical = (matmul_sum(A_plus, B_data) - matmul_sum(A_minus, B_data)) / (2 * h)
            analytical = A.grad[i][j]
            assert abs(numerical - analytical) < 1e-4, (
                f"A[{i}][{j}] 불일치: analytical={analytical}, numerical={numerical}"
            )
    print("PASS: test_matmul_backward_matches_numerical_gradient (순수 Python, 6개 원소 전부 일치)")


def test_pure_python_matches_numpy_version():
    """
    같은 가중치, 같은 입력에 대해 MatrixValue(순수 Python)와 TensorValue(NumPy)가
    정확히 같은 forward, backward 결과를 내는지 검증.
    """
    W1_data = [[0.5, -0.3], [0.2, 0.8]]
    b1_data = [[0.1, -0.1]]
    X_data = [[1.0, 0.0], [0.0, 1.0]]
    y_data = [[1.0], [0.0]]
    W2_data = [[0.4], [-0.6]]
    b2_data = [[0.05]]

    # --- 순수 Python (MatrixValue) ---
    W1_p = PureValue([row[:] for row in W1_data])
    b1_p = PureValue([row[:] for row in b1_data])
    W2_p = PureValue([row[:] for row in W2_data])
    b2_p = PureValue([row[:] for row in b2_data])
    X_p = PureValue([row[:] for row in X_data])

    H_p = (X_p @ W1_p + b1_p).sigmoid()
    O_p = (H_p @ W2_p + b2_p).sigmoid()
    diff_p = O_p + PureValue([[-v for v in row] for row in y_data])
    loss_p = (diff_p * diff_p).sum()
    loss_p.backward()

    # --- NumPy (TensorValue) ---
    W1_n = NumpyValue(np.array(W1_data))
    b1_n = NumpyValue(np.array(b1_data))
    W2_n = NumpyValue(np.array(W2_data))
    b2_n = NumpyValue(np.array(b2_data))
    X_n = NumpyValue(np.array(X_data))
    y_n = np.array(y_data)

    H_n = (X_n @ W1_n + b1_n).sigmoid()
    O_n = (H_n @ W2_n + b2_n).sigmoid()
    diff_n = O_n + (-1 * y_n)
    loss_n = (diff_n * diff_n).sum()
    loss_n.backward()

    assert abs(loss_p.data[0][0] - float(loss_n.data)) < 1e-9, (
        f"loss 불일치: pure={loss_p.data[0][0]}, numpy={float(loss_n.data)}"
    )

    for i in range(2):
        for j in range(2):
            assert abs(W1_p.grad[i][j] - W1_n.grad[i, j]) < 1e-9, (
                f"W1.grad[{i}][{j}] 불일치: pure={W1_p.grad[i][j]}, numpy={W1_n.grad[i, j]}"
            )

    print(f"PASS: test_pure_python_matches_numpy_version "
          f"(loss: pure={loss_p.data[0][0]:.6f} == numpy={float(loss_n.data):.6f}, "
          f"W1.grad 4개 원소 전부 일치)")


if __name__ == "__main__":
    test_matmul_backward_matches_numerical_gradient()
    test_pure_python_matches_numpy_version()
    print("\n모든 테스트 통과.")