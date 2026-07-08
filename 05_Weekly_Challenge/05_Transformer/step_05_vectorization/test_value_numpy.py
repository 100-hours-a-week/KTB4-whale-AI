"""
TensorValue 검증.

검증 항목:
    1. 스칼라(shape=())로 써도 4단계 Value와 동일하게 동작하는가 (하위 호환성)
    2. sigmoid가 배열 전체에 대해 정확히 계산되는가
    3. 행렬곱(__matmul__)의 backward가 수치 미분(numerical differentiation)과 일치하는가
    4. bias처럼 브로드캐스팅되는 파라미터의 gradient가 배치 전체에 대해 정확히 합산되는가
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue


def test_scalar_backward_compatible_with_step4():
    """스칼라로 사용해도 4단계 Value의 결과(dL_dw=-18.0, dL_db=-6.0)와 동일해야 한다"""
    w = NumpyValue(2.0)
    b = NumpyValue(1.0)
    x_val, y_true_val = 3.0, 10.0

    y_pred = w * x_val + b
    L = (y_pred - y_true_val) ** 2
    L.backward()

    assert abs(float(w.grad) - (-18.0)) < 1e-9, f"dL_dw 불일치: {w.grad}"
    assert abs(float(b.grad) - (-6.0)) < 1e-9, f"dL_db 불일치: {b.grad}"
    print(f"PASS: test_scalar_backward_compatible_with_step4 (w.grad={float(w.grad)}, b.grad={float(b.grad)})")


def test_sigmoid_on_array():
    z = NumpyValue(np.array([0.0, 1.0, -1.0]))
    a = z.sigmoid()

    expected = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
    assert np.allclose(a.data, expected), f"배열 sigmoid 계산 오류: {a.data} vs {expected}"
    print(f"PASS: test_sigmoid_on_array (a.data={a.data})")


def numerical_gradient_matmul(A_data, B_data, entry_index, h=1e-6):
    """A의 특정 원소를 h만큼 움직여서, sum(A@B) 값의 변화로 수치 gradient를 근사"""
    i, j = entry_index
    A_plus = A_data.copy()
    A_plus[i, j] += h
    loss_plus = (A_plus @ B_data).sum()

    A_minus = A_data.copy()
    A_minus[i, j] -= h
    loss_minus = (A_minus @ B_data).sum()

    return (loss_plus - loss_minus) / (2 * h)


def test_matmul_backward_matches_numerical_gradient():
    """
    C = A @ B, L = sum(C) 형태에서, A의 각 원소에 대한 gradient가
    수치 미분과 일치하는지 확인. (2,3) @ (3,2) 행렬곱 사용.
    """
    A_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B_data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    A = NumpyValue(A_data.copy())
    B = NumpyValue(B_data.copy())
    C = A @ B
    L = C.sum()
    L.backward()

    for i in range(2):
        for j in range(3):
            numerical = numerical_gradient_matmul(A_data, B_data, (i, j))
            analytical = A.grad[i, j]
            assert abs(numerical - analytical) < 1e-4, (
                f"A[{i},{j}] gradient 불일치: analytical={analytical}, numerical={numerical}"
            )
    print(f"PASS: test_matmul_backward_matches_numerical_gradient (A.grad 전체 6개 원소 수치 미분과 일치)")
    print(f"  A.grad =\n{A.grad}")


def test_broadcast_bias_gradient_sums_over_batch():
    """
    bias(모양 (1,2))가 배치(모양 (4,2))에 브로드캐스팅되어 더해질 때,
    bias의 gradient가 배치 4개에 대한 gradient의 합이어야 한다.
    """
    batch = NumpyValue(np.ones((4, 2)))  # 배치 4개, 각 2차원
    bias = NumpyValue(np.zeros((1, 2)))

    out = batch + bias
    loss = out.sum()
    loss.backward()

    # 각 배치 샘플이 sum()에 동일하게 1씩 기여하므로, bias.grad는 배치 크기(4)만큼 누적되어야 함
    expected_bias_grad = np.array([[4.0, 4.0]])
    assert np.allclose(bias.grad, expected_bias_grad), f"bias.grad 불일치: {bias.grad}"
    print(f"PASS: test_broadcast_bias_gradient_sums_over_batch (bias.grad={bias.grad})")


if __name__ == "__main__":
    test_scalar_backward_compatible_with_step4()
    test_sigmoid_on_array()
    test_matmul_backward_matches_numerical_gradient()
    test_broadcast_bias_gradient_sums_over_batch()
    print("\n모든 테스트 통과.")