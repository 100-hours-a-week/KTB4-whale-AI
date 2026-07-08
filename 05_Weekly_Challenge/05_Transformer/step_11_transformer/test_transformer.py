"""
11단계 검증.

검증 항목:
    1. ReLU의 backward가 수치 미분과 일치하는가
    2. FFN이 없으면(선형 연산만으로는) 위치별 비선형 함수를 학습할 수 없는가
       (FFN이 없으면 실패, 있으면 성공한다는 것을 대조)
    3. FFN을 포함한 DecoderLayer 전체의 backward가 정확한가
    4. (핵심) 여러 DecoderLayer를 쌓았을 때, gradient가 첫 번째 층까지
       정확히 도달하는가 (층 사이의 계산 그래프가 끊기지 않는지 확인)
    5. TransformerDecoder 전체가 실제로 무언가를 학습할 수 있는가
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue
from feed_forward import FeedForward
from decoder_layer import DecoderLayer
from transformer_decoder import TransformerDecoder


def test_relu_backward_matches_numerical_gradient():
    x_data = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    x = NumpyValue(x_data.copy())
    loss = (x.relu() * x.relu()).sum()
    loss.backward()

    def numgrad(i, h=1e-6):
        xp = x_data.copy(); xp[0, i] += h
        xm = x_data.copy(); xm[0, i] -= h
        def L(xd):
            v = NumpyValue(xd)
            return float((v.relu() * v.relu()).sum().data)
        return (L(xp) - L(xm)) / (2 * h)

    for i in range(5):
        num = numgrad(i)
        ana = x.grad[0, i]
        assert abs(num - ana) < 1e-3, f"x[{i}] 불일치: analytical={ana}, numerical={num}"
    print("PASS: test_relu_backward_matches_numerical_gradient (5개 지점 전부 일치)")


def test_without_ffn_cannot_fit_positionwise_nonlinear_function():
    """
    선형 변환(W, b)만으로는 위치별 비선형 함수(y=x^2)를 학습할 수 없음을 재확인.
    (1단계 XOR 실패와 같은 종류의 한계 -- attention까지의 모든 연산이 선형이라는 증거)
    """
    x_data = np.array([[1.0], [2.0], [3.0], [-1.0], [-2.0]])
    y_true = x_data ** 2

    W = NumpyValue(np.array([[0.5]]))
    b = NumpyValue(np.array([[0.0]]))
    x_val = NumpyValue(x_data)

    for _ in range(2000):
        W.grad = np.zeros_like(W.data)
        b.grad = np.zeros_like(b.data)
        pred = x_val @ W + b
        diff = pred + (-1 * y_true)
        loss = (diff * diff).sum() * (1.0 / 5)
        loss.backward()
        W.data -= 0.01 * W.grad
        b.data -= 0.01 * b.grad

    final_pred = (x_val @ W + b).data
    mse = float(((final_pred - y_true) ** 2).mean())
    assert mse > 1.0, f"선형식만으로 x^2을 학습해버림 (모순): mse={mse}"
    print(f"PASS: test_without_ffn_cannot_fit_positionwise_nonlinear_function "
          f"(선형식만으로는 MSE={mse:.4f}, 수렴 실패 확인)")


def test_with_ffn_can_fit_positionwise_nonlinear_function():
    """FFN(ReLU 포함)을 쓰면 같은 위치별 비선형 함수(y=x^2)를 학습할 수 있는지 확인"""
    x_data = np.array([[1.0], [2.0], [3.0], [-1.0], [-2.0]])
    y_true = x_data ** 2

    ffn = FeedForward(d_model=1, d_ff=16, seed=0)
    x_val = NumpyValue(x_data)

    for _ in range(3000):
        ffn.zero_grad()
        pred = ffn.forward(x_val)
        diff = pred + (-1 * y_true)
        loss = (diff * diff).sum() * (1.0 / 5)
        loss.backward()
        for p in ffn.parameters():
            p.data -= 0.02 * p.grad

    final_pred = ffn.forward(x_val).data
    mse = float(((final_pred - y_true) ** 2).mean())
    assert mse < 1.0, f"FFN을 썼는데도 MSE가 큼: mse={mse}"
    print(f"PASS: test_with_ffn_can_fit_positionwise_nonlinear_function "
          f"(FFN 사용 시 MSE={mse:.4f}, 목표={y_true.flatten()}, 예측={final_pred.flatten()})")


def numerical_gradient_for_ffn_W1(layer: DecoderLayer, X: np.ndarray, entry_index: tuple, h=1e-6) -> float:
    i, j = entry_index
    W1 = layer.ffn.W1
    original = W1.data[i, j]

    W1.data[i, j] = original + h
    out_plus = layer.forward(X)
    loss_plus = float((out_plus * out_plus).sum().data)

    W1.data[i, j] = original - h
    out_minus = layer.forward(X)
    loss_minus = float((out_minus * out_minus).sum().data)

    W1.data[i, j] = original
    return (loss_plus - loss_minus) / (2 * h)


def test_decoder_layer_with_ffn_backward_matches_numerical_gradient():
    """FFN을 포함한 DecoderLayer 전체의 backward가 정확한지, ffn.W1로 검증"""
    layer = DecoderLayer(d_model=4, num_heads=2, d_ff=6, seed=0)
    X = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    layer.zero_grad()
    output = layer.forward(X)
    loss = (output * output).sum()
    loss.backward()

    d_model, d_ff = layer.ffn.W1.data.shape
    for i in range(d_model):
        for j in range(d_ff):
            numerical = numerical_gradient_for_ffn_W1(layer, X, (i, j))
            analytical = layer.ffn.W1.grad[i, j]
            assert abs(numerical - analytical) < 1e-3, (
                f"ffn.W1[{i},{j}] 불일치: analytical={analytical}, numerical={numerical}"
            )
    print(f"PASS: test_decoder_layer_with_ffn_backward_matches_numerical_gradient "
          f"(ffn.W1 {d_model*d_ff}개 원소 전부 일치)")


def test_gradient_flows_through_all_stacked_layers():
    """
    핵심 검증: 여러 DecoderLayer를 쌓았을 때, 첫 번째(가장 먼) 층의
    파라미터까지 gradient가 정확히 도달하는지 확인 (as_value로 그래프 보존).
    """
    model = TransformerDecoder(d_model=8, num_heads=2, d_ff=16, num_layers=3, seed=0)
    X = np.random.default_rng(1).uniform(-1, 1, size=(5, 8))

    model.zero_grad()
    output = model.forward(X)
    loss = (output * output).sum()
    loss.backward()

    gradients_per_layer = [
        np.abs(layer.attention.heads[0].Wq.grad).sum() for layer in model.layers
    ]
    for i, g in enumerate(gradients_per_layer):
        assert g > 0, f"{i}번째 층까지 gradient가 도달하지 못함 (그래프 단절)"

    print(f"PASS: test_gradient_flows_through_all_stacked_layers "
          f"(층별 Wq gradient 절댓값 합: {[f'{g:.6f}' for g in gradients_per_layer]})")


if __name__ == "__main__":
    test_relu_backward_matches_numerical_gradient()
    test_without_ffn_cannot_fit_positionwise_nonlinear_function()
    test_with_ffn_can_fit_positionwise_nonlinear_function()
    test_decoder_layer_with_ffn_backward_matches_numerical_gradient()
    test_gradient_flows_through_all_stacked_layers()
    print("\n모든 테스트 통과.")