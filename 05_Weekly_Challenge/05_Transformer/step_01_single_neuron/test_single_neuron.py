from single_neuron import SingleNeuron, generate_data, train

def test_forward_computes_linear_equation():
    neuron = SingleNeuron(seed=0)
    neuron.w = 2.0
    neuron.b = 1.0

    y_pred = neuron.forward(3.0)

    # y = 2*3 + 1 = 7
    assert y_pred == 7.0, f"forward 계산 오류: 기대값 7.0, 실제값 {y_pred}"

    print("PASS: test_forward_computes_linear_equation")

def test_backward_gradient_matches_manual_derivative():
    neuron = SingleNeuron(seed=0)
    neuron.w = 2.0
    neuron.b = 1.0

    x, y_true = 3.0, 10.0
    y_pred = neuron.forward(x) # 7.0

    dL_dw, dL_db = neuron.backward(x, y_pred, y_true)

    # error = y_pred - y_true = 7.0 - 10.0 = -3.0
    # dL/dw = 2 * error * x = 2 * (-3.0) * 3.0 = -18.0
    # dL/db = 2 * error       = 2 * (-3.0)        = -6.0
    expected_dw = -18.0
    expected_db = -6.0

    assert abs(dL_dw - expected_dw) < 1e-9, f"dL/dw 오류: 기대값 {expected_dw}, 실제값 {dL_dw}"
    assert abs(dL_db - expected_db) < 1e-9, f"dL/db 오류: 기대값 {expected_db}, 실제값 {dL_db}"

    print("PASS: test_backward_gradient_matches_manual_derivative")

def test_loss_decreases_after_single_step():
    neuron = SingleNeuron(seed=0)
    x, y_true = 3.0, 10.0
 
    y_pred_before = neuron.forward(x)
    loss_before = neuron.compute_loss(y_pred_before, y_true)
 
    neuron.train_step(x, y_true, learning_rate=0.01)
 
    y_pred_after = neuron.forward(x)
    loss_after = neuron.compute_loss(y_pred_after, y_true)
 
    assert loss_after < loss_before, (
        f"1 step 학습 후 loss가 감소해야 함: before={loss_before}, after={loss_after}"
    )
    print("PASS: test_loss_decreases_after_single_step")
 
def test_converges_to_target_weight_and_bias():
    """
    핵심 테스트: y = 2x + 1 데이터를 100 epoch 학습시켰을 때,
    w가 2.0에, b가 1.0에 충분히 가까워지는지 확인.
    """
    TRUE_W, TRUE_B = 2.0, 1.0
    TOLERANCE = 0.1  # 허용 오차
 
    neuron = SingleNeuron(seed=42)
    data = generate_data(n=20, true_w=TRUE_W, true_b=TRUE_B, seed=0)
 
    train(neuron, data, epochs=100, learning_rate=0.001)
 
    w_error = abs(neuron.w - TRUE_W)
    b_error = abs(neuron.b - TRUE_B)
 
    assert w_error < TOLERANCE, f"w가 수렴하지 않음: w={neuron.w:.4f}, 목표={TRUE_W}, 오차={w_error:.4f}"
    assert b_error < TOLERANCE, f"b가 수렴하지 않음: b={neuron.b:.4f}, 목표={TRUE_B}, 오차={b_error:.4f}"
    print(f"PASS: test_converges_to_target_weight_and_bias (w={neuron.w:.4f}, b={neuron.b:.4f})")

if __name__ == "__main__":
    test_forward_computes_linear_equation()
    test_backward_gradient_matches_manual_derivative()
    test_loss_decreases_after_single_step()
    test_converges_to_target_weight_and_bias()
    print("\n모든 테스트 통과.")