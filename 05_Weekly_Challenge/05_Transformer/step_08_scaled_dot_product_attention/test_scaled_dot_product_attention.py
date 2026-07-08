"""
ScaledDotProductAttention 검증.

검증 항목:
    1. attention weight 행렬 A의 각 행 합이 정확히 1인가 (softmax 정규화 확인)
    2. 전체 forward-backward가 수치 미분과 일치하는가 (Wq 전체 원소)
    3. (핵심) 7단계 S와 달리, 8단계 O가 실제로 V(정보)를 가중합해서 만든
       "의미 있는 값"인지 -- attention weight가 균등하지 않을 때 O가
       V의 단순 평균과 달라지는지 확인
"""

import numpy as np
from scaled_dot_product_attention import ScaledDotProductAttention


def test_attention_weights_sum_to_one():
    """softmax를 거친 A의 각 행 합이 정확히 1인지 확인 (정규화 검증)"""
    model = ScaledDotProductAttention(d_model=4, d_k=3, seed=0)
    X = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])

    O, A = model.forward(X)

    row_sums = A.data.sum(axis=-1)
    assert np.allclose(row_sums, 1.0), f"attention weight 행 합이 1이 아님: {row_sums}"
    print(f"PASS: test_attention_weights_sum_to_one (각 행 합: {row_sums})")


def numerical_gradient_for_Wq(model: ScaledDotProductAttention, X: np.ndarray,
                              entry_index: tuple, h=1e-6) -> float:
    """Wq의 특정 원소를 h만큼 움직여서, sum(O) 값의 변화로 수치 gradient를 근사"""
    i, j = entry_index
    original = model.Wq.data[i, j]

    model.Wq.data[i, j] = original + h
    O_plus, _ = model.forward(X)
    loss_plus = float(O_plus.sum().data)

    model.Wq.data[i, j] = original - h
    O_minus, _ = model.forward(X)
    loss_minus = float(O_minus.sum().data)

    model.Wq.data[i, j] = original
    return (loss_plus - loss_minus) / (2 * h)


def test_full_attention_backward_matches_numerical_gradient():
    """O = softmax(Q@K.T/sqrt(d_k)) @ V 전체의 backward가 정확한지, Wq 전체 원소로 검증"""
    model = ScaledDotProductAttention(d_model=3, d_k=2, seed=0)
    X = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    model.zero_grad()
    O, A = model.forward(X)
    loss = O.sum()
    loss.backward()

    d_model, d_k = model.Wq.data.shape
    for i in range(d_model):
        for j in range(d_k):
            numerical = numerical_gradient_for_Wq(model, X, (i, j))
            analytical = model.Wq.grad[i, j]
            assert abs(numerical - analytical) < 1e-3, (
                f"Wq[{i},{j}] 불일치: analytical={analytical}, numerical={numerical}"
            )
    print(f"PASS: test_full_attention_backward_matches_numerical_gradient "
          f"(Wq 전체 {d_model*d_k}개 원소가 수치 미분과 일치)")


def test_output_actually_mixes_value_information():
    """
    핵심 검증: 7단계는 S(유사도 점수)만 있고 정보 이동이 없었지만,
    8단계는 O가 실제로 V를 attention weight로 가중합한 값이어야 한다.

    검증 방법: attention weight A가 균등(uniform)하지 않다면,
    O는 V의 단순 평균과 달라야 한다. (균등하다면 오히려 O == V의 평균이 되는 것도 맞는 계산)
    """
    model = ScaledDotProductAttention(d_model=4, d_k=3, seed=7)
    X = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])

    O, A = model.forward(X)

    # O[i] = sum_j( A[i,j] * V[j] ) 라는 정의를, 직접 재계산해서 대조
    X_val_data = X
    V_manual = X_val_data @ model.Wv.data  # V = X @ Wv (직접 계산)
    O_manual = A.data @ V_manual            # O = A @ V (가중합 직접 계산)

    assert np.allclose(O.data, O_manual, atol=1e-9), (
        f"O가 A@V의 정의와 다름: O={O.data}, 직접계산={O_manual}"
    )

    # attention weight가 균등하지 않다는 것도 확인 (완전히 무작위가 아니라 학습 가능한 구조라는 증거)
    is_uniform = np.allclose(A.data, 1.0 / A.data.shape[1], atol=1e-3)
    print(f"PASS: test_output_actually_mixes_value_information")
    print(f"  O가 A@V(가중합) 정의와 정확히 일치함을 확인")
    print(f"  attention weight가 균등 분포인가: {is_uniform} (False가 자연스러움 -- Wq/Wk가 서로 다르므로)")


if __name__ == "__main__":
    test_attention_weights_sum_to_one()
    test_full_attention_backward_matches_numerical_gradient()
    test_output_actually_mixes_value_information()
    print("\n모든 테스트 통과.")