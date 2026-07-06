"""
NeuronWithActivation(sigmoid 추가된 뉴런) 검증.

검증 항목:
    1. forward()가 sigmoid(w1*x1+w2*x2+b)를 정확히 계산하는가
    2. backward()의 chain rule이 수치 미분과 일치하는가
    3. AND(linearly separable)는 충분히 낮은 MSE로 수렴하는가
    4. XOR(not linearly separable)은 activation을 추가해도 여전히 풀리지 않는가
       (activation만으로는 부족하고, 여러 뉴런을 쌓아야 한다는 것 -> 3단계로 이어짐)
"""

from neuron_with_activation import NeuronWithActivation, train, AND_DATA, XOR_DATA
from activation import sigmoid


def numerical_gradient(neuron: NeuronWithActivation, x1: float, x2: float, y_true: float,
                        param_name: str, h: float = 1e-6) -> float:
    """
    특정 파라미터(w1, w2, b)에 대한 수치 gradient 계산.
    해당 파라미터만 h만큼 움직여서 loss 변화를 관찰.
    """
    original_value = getattr(neuron, param_name)

    setattr(neuron, param_name, original_value + h)
    a_plus, _ = neuron.forward(x1, x2)
    loss_plus = neuron.compute_loss(a_plus, y_true)

    setattr(neuron, param_name, original_value - h)
    a_minus, _ = neuron.forward(x1, x2)
    loss_minus = neuron.compute_loss(a_minus, y_true)

    setattr(neuron, param_name, original_value)  # 원상복구

    return (loss_plus - loss_minus) / (2 * h)


def test_forward_computes_sigmoid_of_weighted_sum():
    neuron = NeuronWithActivation(seed=0)
    neuron.w1, neuron.w2, neuron.b = 1.0, 2.0, 0.5

    a, z = neuron.forward(x1=1.0, x2=1.0)

    expected_z = 1.0 * 1.0 + 2.0 * 1.0 + 0.5  # 3.5
    expected_a = sigmoid(expected_z)

    assert abs(z - expected_z) < 1e-9, f"z 계산 오류: 기대값 {expected_z}, 실제값 {z}"
    assert abs(a - expected_a) < 1e-9, f"a 계산 오류: 기대값 {expected_a}, 실제값 {a}"
    print(f"PASS: test_forward_computes_sigmoid_of_weighted_sum (z={z}, a={a:.6f})")


def test_backward_matches_numerical_gradient():
    neuron = NeuronWithActivation(seed=0)
    neuron.w1, neuron.w2, neuron.b = 0.5, -0.3, 0.1

    x1, x2, y_true = 1.0, 0.0, 1.0
    a, z = neuron.forward(x1, x2)

    dL_dw1, dL_dw2, dL_db = neuron.backward(x1, x2, a, y_true)

    numerical_dw1 = numerical_gradient(neuron, x1, x2, y_true, 'w1')
    numerical_dw2 = numerical_gradient(neuron, x1, x2, y_true, 'w2')
    numerical_db = numerical_gradient(neuron, x1, x2, y_true, 'b')

    assert abs(dL_dw1 - numerical_dw1) < 1e-4, f"dw1 불일치: analytical={dL_dw1}, numerical={numerical_dw1}"
    assert abs(dL_dw2 - numerical_dw2) < 1e-4, f"dw2 불일치: analytical={dL_dw2}, numerical={numerical_dw2}"
    assert abs(dL_db - numerical_db) < 1e-4, f"db 불일치: analytical={dL_db}, numerical={numerical_db}"

    print(f"PASS: test_backward_matches_numerical_gradient "
          f"(dw1: {dL_dw1:.6f} vs {numerical_dw1:.6f}, "
          f"dw2: {dL_dw2:.6f} vs {numerical_dw2:.6f}, "
          f"db: {dL_db:.6f} vs {numerical_db:.6f})")


def test_and_problem_converges():
    """AND는 linearly separable하므로, activation 추가된 뉴런 1개로 충분히 풀려야 한다"""
    neuron = NeuronWithActivation(seed=42)
    train(neuron, AND_DATA, epochs=3000, learning_rate=1.0)

    total_squared_error = 0.0
    all_correct = True
    for x1, x2, y_true in AND_DATA:
        a, _ = neuron.forward(x1, x2)
        total_squared_error += (a - y_true) ** 2
        predicted_class = 1.0 if a >= 0.5 else 0.0
        if predicted_class != y_true:
            all_correct = False

    mse = total_squared_error / len(AND_DATA)
    assert mse < 0.01, f"AND 문제인데 MSE가 충분히 안 줄어듦: {mse}"
    assert all_correct, "AND 문제인데 일부 케이스를 잘못 분류함"
    print(f"PASS: test_and_problem_converges (MSE={mse:.6f}, 4개 케이스 모두 정확히 분류)")


def test_xor_problem_cannot_converge():
    """
    핵심 테스트: XOR은 not linearly separable하므로,
    activation을 추가해도 뉴런 1개로는 여전히 풀리지 않아야 한다.
    """
    neuron = NeuronWithActivation(seed=42)
    train(neuron, XOR_DATA, epochs=3000, learning_rate=1.0)

    total_squared_error = 0.0
    correct_count = 0
    for x1, x2, y_true in XOR_DATA:
        a, _ = neuron.forward(x1, x2)
        total_squared_error += (a - y_true) ** 2
        predicted_class = 1.0 if a >= 0.5 else 0.0
        if predicted_class == y_true:
            correct_count += 1

    mse = total_squared_error / len(XOR_DATA)

    # AND의 기준(0.01)과 달리, XOR은 MSE가 충분히 크게 남아있어야 한다 (풀리지 않았다는 증거)
    assert mse > 0.1, f"XOR 문제인데 MSE가 너무 작음 (뉴런 1개로 풀렸다는 뜻이라 모순): {mse}"
    # 4개 중 4개를 다 맞히지는 못해야 함 (완전히 못 푸는 게 XOR의 특징)
    assert correct_count < 4, f"XOR 문제인데 4개 모두 정확히 분류함 (모순): correct_count={correct_count}"

    print(f"PASS: test_xor_problem_cannot_converge "
          f"(MSE={mse:.6f}, {correct_count}/4개만 정확히 분류 -> 뉴런 1개로는 XOR 풀 수 없음 확인)")


if __name__ == "__main__":
    test_forward_computes_sigmoid_of_weighted_sum()
    test_backward_matches_numerical_gradient()
    test_and_problem_converges()
    test_xor_problem_cannot_converge()
    print("\n모든 테스트 통과.")