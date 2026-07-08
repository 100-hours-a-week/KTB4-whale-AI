"""
순수 Python(BatchMLPPure)과 NumPy(BatchMLP) 버전 검증.

검증 항목:
    1. (정확성) BatchMLP(NumPy)의 배치 forward가 개별 스칼라 계산과 일치하는가
    2. (정확성) BatchMLPPure(순수 Python)가 XOR을 정확히 학습하는가
    3. (속도) 단순 행렬곱(200x200) 단독 비교 -- 순수 Python vs NumPy
    4. (속도) XOR 학습 전체 루프(파라미터 9개, 소규모) 비교 -- 순수 Python vs NumPy

3, 4를 하나의 파일에 함께 둔 이유: 둘 다 "순수 Python과 NumPy 중 어느 쪽이 빠른가"라는
같은 질문을 다루며, 계산 규모에 따라 결과가 정반대로 나온다는 것을 대조하기 위함이다.
"""

import time
import numpy as np
from mlp_numpy import BatchMLP, XOR_X as XOR_X_NUMPY, XOR_Y as XOR_Y_NUMPY
from mlp_pure import BatchMLPPure, XOR_X as XOR_X_PURE, XOR_Y as XOR_Y_PURE


def test_batch_forward_matches_manual_per_sample_computation():
    """BatchMLP(NumPy)의 배치 forward가 개별 스칼라 계산과 일치하는지 확인"""
    mlp = BatchMLP(seed=1)
    O_batch = mlp.forward(XOR_X_NUMPY)

    W1, b1, W2, b2 = mlp.W1.data, mlp.b1.data, mlp.W2.data, mlp.b2.data

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for i in range(4):
        x1, x2 = XOR_X_NUMPY[i]
        h1 = sigmoid(x1 * W1[0, 0] + x2 * W1[1, 0] + b1[0, 0])
        h2 = sigmoid(x1 * W1[0, 1] + x2 * W1[1, 1] + b1[0, 1])
        o = sigmoid(h1 * W2[0, 0] + h2 * W2[1, 0] + b2[0, 0])
        assert abs(o - O_batch.data[i, 0]) < 1e-9, (
            f"샘플 {i}: 배치 계산={O_batch.data[i, 0]}, 개별 계산={o} 불일치"
        )
    print("PASS: test_batch_forward_matches_manual_per_sample_computation "
          "(배치 처리 결과가 개별 스칼라 계산과 4개 샘플 모두 정확히 일치)")


def test_pure_python_batch_mlp_converges_xor():
    """BatchMLPPure(순수 Python)가 XOR을 정확히 학습하는지 확인"""
    mlp = BatchMLPPure(seed=0)
    for _ in range(10000):
        mlp.train_step(XOR_X_PURE, XOR_Y_PURE, learning_rate=1.0)

    O = mlp.forward(XOR_X_PURE)
    correct = sum(
        1 for i in range(4)
        if (O.data[i][0] >= 0.5) == bool(XOR_Y_PURE[i][0])
    )
    assert correct == 4, f"순수 Python 버전이 XOR을 다 못 맞춤: {correct}/4"
    print(f"PASS: test_pure_python_batch_mlp_converges_xor ({correct}/4)")


def pure_python_matmul(A: list, B: list) -> list:
    """순수 Python 중첩 리스트로 구현한 행렬곱 (NumPy 없이)"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            total = 0.0
            for k in range(cols_A):
                total += A[i][k] * B[k][j]
            result[i][j] = total
    return result


def test_speed_large_matmul_numpy_wins():
    """대규모 행렬곱(200x200) 단독 비교 -- NumPy가 압도적으로 빠를 것으로 예상"""
    n = 200
    rng = np.random.default_rng(0)
    A_np = rng.uniform(-1, 1, size=(n, n))
    B_np = rng.uniform(-1, 1, size=(n, n))
    A_list, B_list = A_np.tolist(), B_np.tolist()

    start = time.perf_counter()
    result_pure = pure_python_matmul(A_list, B_list)
    pure_time = time.perf_counter() - start

    start = time.perf_counter()
    result_numpy = A_np @ B_np
    numpy_time = time.perf_counter() - start

    assert np.allclose(np.array(result_pure), result_numpy, atol=1e-6), "두 방식의 계산 결과가 다름"

    speedup = pure_time / numpy_time
    print(f"PASS: test_speed_large_matmul_numpy_wins")
    print(f"  {n}x{n} 행렬곱 기준")
    print(f"  순수 Python 반복문: {pure_time*1000:.2f}ms")
    print(f"  NumPy 벡터화:       {numpy_time*1000:.4f}ms")
    print(f"  속도 차이: {speedup:.0f}배 (NumPy가 더 빠름)")
    assert speedup > 10, f"NumPy가 예상만큼 빠르지 않음 (속도 차이: {speedup:.0f}배)"


def test_speed_small_xor_training_pure_python_competitive():
    """소규모 XOR 학습(파라미터 9개) 전체 루프 비교 -- 결과가 역전될 수 있음을 확인"""
    epochs = 2000

    mlp_pure = BatchMLPPure(seed=0)
    start = time.perf_counter()
    for _ in range(epochs):
        mlp_pure.train_step(XOR_X_PURE, XOR_Y_PURE, learning_rate=1.0)
    pure_time = time.perf_counter() - start

    mlp_numpy = BatchMLP(seed=0)
    start = time.perf_counter()
    for _ in range(epochs):
        mlp_numpy.train_step(XOR_X_NUMPY, XOR_Y_NUMPY, learning_rate=1.0)
    numpy_time = time.perf_counter() - start

    print(f"PASS: test_speed_small_xor_training_pure_python_competitive")
    print(f"  {epochs} epoch, XOR(4개 샘플, 파라미터 9개) 학습 기준")
    print(f"  순수 Python (MatrixValue): {pure_time*1000:.2f}ms")
    print(f"  NumPy (TensorValue):       {numpy_time*1000:.2f}ms")
    print(f"  속도 비율: {pure_time/numpy_time:.2f}배")


if __name__ == "__main__":
    test_batch_forward_matches_manual_per_sample_computation()
    test_pure_python_batch_mlp_converges_xor()
    test_speed_large_matmul_numpy_wins()
    test_speed_small_xor_training_pure_python_competitive()
    print("\n모든 테스트 통과.")