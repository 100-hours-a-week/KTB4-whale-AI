"""
PyTorch 버전 MLP가 XOR을 정확히 학습하는지 검증.
(3단계/Value 버전과 초기 seed의 RNG 알고리즘이 달라 완전히 같은 수치는 아니지만,
 수렴 여부와 정확도는 동일해야 한다.)
"""

from mlp_pytorch_autograd import MLP, train, XOR_DATA


def test_xor_converges_with_pytorch():
    mlp = MLP(seed=0)
    train(mlp, XOR_DATA, epochs=10000, learning_rate=1.0)

    total_squared_error = 0.0
    correct_count = 0
    for x1, x2, y_true in XOR_DATA:
        o = mlp.forward(x1, x2)
        total_squared_error += (o.item() - y_true) ** 2
        if (o.item() >= 0.5) == bool(y_true):
            correct_count += 1

    mse = total_squared_error / len(XOR_DATA)
    assert mse < 0.01, f"XOR 문제인데 MSE가 충분히 안 줄어듦: {mse}"
    assert correct_count == 4, f"XOR 문제인데 4개 모두 정확히 분류하지 못함: {correct_count}"
    print(f"PASS: test_xor_converges_with_pytorch (MSE={mse:.6f}, {correct_count}/4개 정확히 분류)")


if __name__ == "__main__":
    test_xor_converges_with_pytorch()
    print("\n모든 테스트 통과.")