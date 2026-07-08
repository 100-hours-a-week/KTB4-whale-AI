"""
검증 항목:
    1. positional encoding이 서로 다른 위치마다 실제로 다른 패턴을 만드는가
       (9단계까지의 한계 -- "순서를 모른다" -- 가 실제로 해결되는지)
    2. LayerNorm 결과가 각 행마다 평균 0, 분산 1이 되는가
    3. LayerNorm의 backward가 수치 미분과 일치하는가
    4. DecoderLayer 전체(positional + attention + residual + LayerNorm)의
       forward-backward가 정확한가
"""

import numpy as np
from positional_residual_layernorm import positional_encoding, LayerNorm
from decoder_layer import DecoderLayer


def test_positional_encoding_differs_by_position():
    """서로 다른 위치의 PE 벡터가 실제로 다른지 확인 -- 이게 없으면 9단계까지와 다를 게 없다"""
    pe = positional_encoding(seq_len=6, d_model=8)

    for i in range(6):
        for j in range(i + 1, 6):
            assert not np.allclose(pe[i], pe[j]), f"위치 {i}와 {j}의 PE가 동일함 -- 위치 구분 실패"
    print(f"PASS: test_positional_encoding_differs_by_position (6개 위치 전부 서로 다른 패턴)")


def test_layernorm_output_has_zero_mean_unit_variance():
    """LayerNorm 결과가 각 행마다 평균 0, 분산 1에 가까운지 확인"""
    from step_05_vectorization.value_numpy import NumpyValue

    ln = LayerNorm(d_model=4)
    x_data = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 100.0, 5.0, 8.0]])
    x = NumpyValue(x_data)

    out = ln.forward(x)

    row_means = out.data.mean(axis=-1)
    row_vars = out.data.var(axis=-1)
    assert np.allclose(row_means, 0.0, atol=1e-6), f"평균이 0이 아님: {row_means}"
    assert np.allclose(row_vars, 1.0, atol=1e-3), f"분산이 1이 아님: {row_vars}"
    print(f"PASS: test_layernorm_output_has_zero_mean_unit_variance "
          f"(행별 평균={row_means}, 행별 분산={row_vars})")


def numerical_gradient_for_gamma(ln: LayerNorm, x_data: np.ndarray, index: int, h=1e-6) -> float:
    from step_05_vectorization.value_numpy import NumpyValue
    original = ln.gamma.data[0, index]

    ln.gamma.data[0, index] = original + h
    out_plus = ln.forward(NumpyValue(x_data))
    loss_plus = float(out_plus.sum().data)

    ln.gamma.data[0, index] = original - h
    out_minus = ln.forward(NumpyValue(x_data))
    loss_minus = float(out_minus.sum().data)

    ln.gamma.data[0, index] = original
    return (loss_plus - loss_minus) / (2 * h)


def test_layernorm_backward_matches_numerical_gradient():
    """LayerNorm의 gamma에 대한 backward가 수치 미분과 일치하는지 확인"""
    from step_05_vectorization.value_numpy import NumpyValue

    ln = LayerNorm(d_model=4)
    x_data = np.array([[10.0, 20.0, 30.0, 40.0]])
    x = NumpyValue(x_data)

    out = ln.forward(x)
    loss = out.sum()
    loss.backward()

    for i in range(4):
        numerical = numerical_gradient_for_gamma(ln, x_data, i)
        analytical = ln.gamma.grad[0, i]
        assert abs(numerical - analytical) < 1e-3, (
            f"gamma[{i}] 불일치: analytical={analytical}, numerical={numerical}"
        )
    print(f"PASS: test_layernorm_backward_matches_numerical_gradient (gamma 4개 원소 전부 일치)")


def test_decoder_layer_forward_backward_end_to_end():
    """DecoderLayer 전체(positional + attention + residual + LayerNorm)가 정상 동작하는지 확인"""
    layer = DecoderLayer(d_model=8, num_heads=2, seed=0)
    X = np.random.default_rng(1).uniform(-1, 1, size=(5, 8))

    layer.zero_grad()
    output = layer.forward(X)
    loss = output.sum()
    loss.backward()

    assert output.data.shape == (5, 8), f"출력 모양이 다름: {output.data.shape}"

    row_means = output.data.mean(axis=-1)
    row_vars = output.data.var(axis=-1)
    assert np.allclose(row_means, 0.0, atol=1e-5), f"최종 출력의 행별 평균이 0이 아님: {row_means}"
    assert np.allclose(row_vars, 1.0, atol=1e-3), f"최종 출력의 행별 분산이 1이 아님: {row_vars}"

    has_gradient = any(np.abs(p.grad).sum() > 0 for p in layer.parameters())
    assert has_gradient, "전체 파라미터의 gradient가 전부 0임 -- backward가 그래프를 타지 못함"

    print(f"PASS: test_decoder_layer_forward_backward_end_to_end "
          f"(output.shape={output.data.shape}, LayerNorm 정규화 유지, gradient 전파 확인)")


if __name__ == "__main__":
    test_positional_encoding_differs_by_position()
    test_layernorm_output_has_zero_mean_unit_variance()
    test_layernorm_backward_matches_numerical_gradient()
    test_decoder_layer_forward_backward_end_to_end()
    print("\n모든 테스트 통과.")