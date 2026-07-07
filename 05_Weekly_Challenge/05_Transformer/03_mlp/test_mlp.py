"""
MLP 검증.

검증 항목:
    1. forward()가 은닉층 -> 출력층을 정확히 계산하는가
    2. backward()가 수치 미분(numerical differentiation)과 일치하는가
       (은닉층까지 전파된 gradient가 정확한지가 핵심 — backpropagation 검증)
    3. XOR(2단계에서 실패)이 MLP로는 수렴하는가
    4. AND(2단계에서도 성공했던 문제)가 MLP로도 여전히 잘 풀리는가 (회귀 방지)
"""

from mlp import MLP, train, AND_DATA, XOR_DATA
from activation import sigmoid


def numerical_gradient_for_param(mlp: MLP, x1: float, x2: float, y_true: float,
                                  neuron_name: str, param_index, h: float = 1e-6) -> float:
    """
    특정 뉴런의 특정 파라미터에 대한 수치 gradient 계산.

    neuron_name: 'output', 'hidden1', 'hidden2'
    param_index: 'weights'면 (인덱스, 'weights') 형태로, bias면 'bias' 문자열로 지정
    """
    neuron = getattr(mlp, neuron_name)

    if param_index == 'bias':
        original = neuron.bias
        neuron.bias = original + h
        o_plus, _ = mlp.forward(x1, x2)
        loss_plus = mlp.compute_loss(o_plus, y_true)

        neuron.bias = original - h
        o_minus, _ = mlp.forward(x1, x2)
        loss_minus = mlp.compute_loss(o_minus, y_true)

        neuron.bias = original
    else:
        idx = param_index
        original = neuron.weights[idx]
        neuron.weights[idx] = original + h
        o_plus, _ = mlp.forward(x1, x2)
        loss_plus = mlp.compute_loss(o_plus, y_true)

        neuron.weights[idx] = original - h
        o_minus, _ = mlp.forward(x1, x2)
        loss_minus = mlp.compute_loss(o_minus, y_true)

        neuron.weights[idx] = original

    return (loss_plus - loss_minus) / (2 * h)


def test_forward_computes_hidden_then_output():
    mlp = MLP(seed=0)
    mlp.hidden1.weights, mlp.hidden1.bias = [1.0, 1.0], 0.0
    mlp.hidden2.weights, mlp.hidden2.bias = [1.0, -1.0], 0.0
    mlp.output.weights, mlp.output.bias = [1.0, 1.0], 0.0

    o, cache = mlp.forward(1.0, 1.0)

    expected_h1 = sigmoid(1.0 * 1.0 + 1.0 * 1.0 + 0.0)   # sigmoid(2.0)
    expected_h2 = sigmoid(1.0 * 1.0 + (-1.0) * 1.0 + 0.0)  # sigmoid(0.0)
    expected_o = sigmoid(1.0 * expected_h1 + 1.0 * expected_h2 + 0.0)

    assert abs(cache['h1'] - expected_h1) < 1e-9, f"h1 계산 오류: {cache['h1']} vs {expected_h1}"
    assert abs(cache['h2'] - expected_h2) < 1e-9, f"h2 계산 오류: {cache['h2']} vs {expected_h2}"
    assert abs(o - expected_o) < 1e-9, f"최종 출력 계산 오류: {o} vs {expected_o}"
    print(f"PASS: test_forward_computes_hidden_then_output (h1={cache['h1']:.4f}, h2={cache['h2']:.4f}, o={o:.4f})")


def test_backward_matches_numerical_gradient_all_layers():
    """
    출력층뿐 아니라 은닉층 파라미터까지, 모든 gradient가 수치 미분과 일치하는지 검증.
    이게 통과해야 backpropagation(은닉층으로의 gradient 전파)이 정확하다는 증거가 된다.
    """
    mlp = MLP(seed=0)
    x1, x2, y_true = 1.0, 0.0, 1.0

    o, cache = mlp.forward(x1, x2)
    gradients = mlp.backward(cache, y_true)

    checks = [
        ('output', 0, gradients['output'][0], 'dL_dv1'),
        ('output', 1, gradients['output'][1], 'dL_dv2'),
        ('output', 'bias', gradients['output'][2], 'dL_dc'),
        ('hidden1', 0, gradients['hidden1'][0], 'dL_dw1_1'),
        ('hidden1', 1, gradients['hidden1'][1], 'dL_dw1_2'),
        ('hidden1', 'bias', gradients['hidden1'][2], 'dL_db1'),
        ('hidden2', 0, gradients['hidden2'][0], 'dL_dw2_1'),
        ('hidden2', 1, gradients['hidden2'][1], 'dL_dw2_2'),
        ('hidden2', 'bias', gradients['hidden2'][2], 'dL_db2'),
    ]

    for neuron_name, param_index, analytical, label in checks:
        numerical = numerical_gradient_for_param(mlp, x1, x2, y_true, neuron_name, param_index)
        diff = abs(analytical - numerical)
        assert diff < 1e-4, f"{label} 불일치: analytical={analytical}, numerical={numerical}"
        print(f"PASS: {label} (neuron={neuron_name}) -> analytical={analytical:.6f}, numerical={numerical:.6f}")


def test_and_still_converges():
    """2단계에서도 성공했던 AND가, MLP로도 여전히 잘 풀리는지 확인 (회귀 방지)"""
    mlp = MLP(seed=0)
    train(mlp, AND_DATA, epochs=5000, learning_rate=1.0)

    total_squared_error = 0.0
    all_correct = True
    for x1, x2, y_true in AND_DATA:
        o, _ = mlp.forward(x1, x2)
        total_squared_error += (o - y_true) ** 2
        if (o >= 0.5) != bool(y_true):
            all_correct = False

    mse = total_squared_error / len(AND_DATA)
    assert mse < 0.01, f"AND 문제인데 MSE가 충분히 안 줄어듦: {mse}"
    assert all_correct, "AND 문제인데 일부 케이스를 잘못 분류함"
    print(f"PASS: test_and_still_converges (MSE={mse:.6f})")


def test_xor_converges_with_mlp():
    """
    핵심 테스트: 2단계에서 실패했던 XOR이, MLP(은닉층 2개 뉴런)로는 수렴해야 한다.
    """
    mlp = MLP(seed=0)
    train(mlp, XOR_DATA, epochs=10000, learning_rate=1.0)

    total_squared_error = 0.0
    correct_count = 0
    for x1, x2, y_true in XOR_DATA:
        o, _ = mlp.forward(x1, x2)
        total_squared_error += (o - y_true) ** 2
        if (o >= 0.5) == bool(y_true):
            correct_count += 1

    mse = total_squared_error / len(XOR_DATA)
    assert mse < 0.01, f"XOR 문제인데 MSE가 충분히 안 줄어듦 (MLP로도 실패): {mse}"
    assert correct_count == 4, f"XOR 문제인데 4개 모두 정확히 분류하지 못함: correct_count={correct_count}"
    print(f"PASS: test_xor_converges_with_mlp (MSE={mse:.6f}, {correct_count}/4개 정확히 분류)")


if __name__ == "__main__":
    test_forward_computes_hidden_then_output()
    test_backward_matches_numerical_gradient_all_layers()
    test_and_still_converges()
    test_xor_converges_with_mlp()
    print("\n모든 테스트 통과.")