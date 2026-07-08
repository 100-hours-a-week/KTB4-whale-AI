"""
MultiHeadAttention 검증.

검증 항목:
    1. 출력 O의 모양이 (seq_len, d_model)로, 입력과 동일한 차원을 유지하는가
       (다음 층에 그대로 쌓을 수 있어야 하므로 중요)
    2. 전체 forward-backward가 수치 미분과 일치하는가
    3. (핵심) 서로 다른 head가 실제로 서로 다른 attention weight(관점)를 학습하는가
       -- 8단계의 한계("head 1개로는 한 가지 관점만 포착")가 해결됐는지 확인
"""

import numpy as np
from multi_head_attention import MultiHeadAttention


def test_output_shape_matches_input_dimension():
    """multi-head의 출력이 입력과 같은 d_model 차원을 유지하는지 확인 (층을 쌓기 위한 전제조건)"""
    d_model = 8
    model = MultiHeadAttention(d_model=d_model, num_heads=4, seed=0)
    X = np.random.default_rng(1).uniform(-1, 1, size=(5, d_model))

    O, head_outputs = model.forward(X)

    assert O.data.shape == (5, d_model), f"출력 모양이 다름: {O.data.shape}, 기대값: (5, {d_model})"
    assert len(head_outputs) == 4, f"head 개수가 다름: {len(head_outputs)}"
    print(f"PASS: test_output_shape_matches_input_dimension "
          f"(O.shape={O.data.shape}, head 개수={len(head_outputs)})")


def numerical_gradient_for_head0_Wq(model: MultiHeadAttention, X: np.ndarray,
                                     entry_index: tuple, h=1e-6) -> float:
    """head 0의 Wq 특정 원소를 h만큼 움직여서, sum(O) 변화로 수치 gradient를 근사"""
    i, j = entry_index
    Wq = model.heads[0].Wq
    original = Wq.data[i, j]

    Wq.data[i, j] = original + h
    O_plus, _ = model.forward(X)
    loss_plus = float(O_plus.sum().data)

    Wq.data[i, j] = original - h
    O_minus, _ = model.forward(X)
    loss_minus = float(O_minus.sum().data)

    Wq.data[i, j] = original
    return (loss_plus - loss_minus) / (2 * h)


def test_multi_head_backward_matches_numerical_gradient():
    """전체 MultiHeadAttention의 backward가 정확한지, head 0의 Wq 전체 원소로 검증"""
    d_model = 6
    model = MultiHeadAttention(d_model=d_model, num_heads=2, seed=0)
    X = np.array([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]])

    model.zero_grad()
    O, _ = model.forward(X)
    loss = O.sum()
    loss.backward()

    d_k = model.d_k
    for i in range(d_model):
        for j in range(d_k):
            numerical = numerical_gradient_for_head0_Wq(model, X, (i, j))
            analytical = model.heads[0].Wq.grad[i, j]
            assert abs(numerical - analytical) < 1e-3, (
                f"head0.Wq[{i},{j}] 불일치: analytical={analytical}, numerical={numerical}"
            )
    print(f"PASS: test_multi_head_backward_matches_numerical_gradient "
          f"(head 0의 Wq 전체 {d_model*d_k}개 원소가 수치 미분과 일치)")


def test_different_heads_learn_different_attention_patterns():
    """
    핵심 검증: 서로 다른 head가 실제로 서로 다른 attention weight(A)를 만들어내는지 확인.
    8단계는 head가 1개뿐이라 "관점이 여러 개"라는 개념 자체가 없었다.
    """
    d_model = 8
    model = MultiHeadAttention(d_model=d_model, num_heads=4, seed=0)
    X = np.random.default_rng(2).uniform(-1, 1, size=(4, d_model))

    O, head_outputs = model.forward(X)
    attention_weights = [A.data for _, A in head_outputs]

    # 서로 다른 head의 attention weight가 전부 동일하지는 않은지 확인
    all_same = all(
        np.allclose(attention_weights[0], attention_weights[k])
        for k in range(1, len(attention_weights))
    )
    assert not all_same, "모든 head의 attention weight가 동일함 -- head가 서로 다른 관점을 학습하지 못함"

    print("PASS: test_different_heads_learn_different_attention_patterns")
    for i, A in enumerate(attention_weights):
        print(f"  head {i} attention weight (첫 행): {A[0]}")


if __name__ == "__main__":
    test_output_shape_matches_input_dimension()
    test_multi_head_backward_matches_numerical_gradient()
    test_different_heads_learn_different_attention_patterns()
    print("\n모든 테스트 통과.")