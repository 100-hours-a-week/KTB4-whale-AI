"""
AttentionScore 검증.

검증 항목:
    1. transpose()의 backward가 수치 미분과 일치하는가
    2. S = Q @ K.T 전체의 backward가 수치 미분과 일치하는가
    3. (핵심) 위치 i와 위치 j 사이의 "그래프 경로 길이"가 시퀀스 길이와 무관하게
       항상 1인가 -- 6단계 RNN의 "경로 길이 = 시점 차이(|i-j|)"와 대비
"""

import numpy as np
from attention_score import AttentionScore
from step_05_vectorization.value_numpy import NumpyValue


def test_transpose_backward_matches_numerical_gradient():
    """(검증 항목 1) transpose()가 forward 계산과 backward(gradient도 함께 전치)를 정확히 수행하는지 검증"""
    A_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    A = NumpyValue(A_data.copy())
    out = A.transpose()
    loss = out.sum()
    loss.backward()

    # sum(A.T) = sum(A)이므로, 모든 원소의 gradient는 1이어야 한다
    assert np.allclose(A.grad, np.ones_like(A_data)), f"transpose backward 오류: {A.grad}"
    print(f"PASS: test_transpose_backward_matches_numerical_gradient (A.grad 전부 1.0)")


def numerical_gradient_for_Wq(model: AttentionScore, X: np.ndarray, entry_index: tuple, h=1e-6) -> float:
    """Wq의 특정 원소를 h만큼 움직여서, sum(S) 값의 변화로 수치 gradient를 근사 (중심 차분)"""
    i, j = entry_index
    original = model.Wq.data[i, j]

    model.Wq.data[i, j] = original + h
    S_plus = model.forward(X)
    loss_plus = float(S_plus.sum().data)

    model.Wq.data[i, j] = original - h
    S_minus = model.forward(X)
    loss_minus = float(S_minus.sum().data)

    model.Wq.data[i, j] = original
    return (loss_plus - loss_minus) / (2 * h)


def test_attention_score_backward_matches_numerical_gradient():
    """(검증 항목 2) S = Q @ K.T 전체 계산의 backward가 정확한지, Wq의 모든 원소에 대해 검증"""
    model = AttentionScore(d_model=3, d_k=2, seed=0)
    X = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    model.zero_grad()
    S = model.forward(X)
    loss = S.sum()
    loss.backward()

    d_model, d_k = model.Wq.data.shape
    for i in range(d_model):
        for j in range(d_k):
            numerical = numerical_gradient_for_Wq(model, X, (i, j))
            analytical = model.Wq.grad[i, j]
            assert abs(numerical - analytical) < 1e-4, (
                f"Wq[{i},{j}] 불일치: analytical={analytical}, numerical={numerical}"
            )
    print(f"PASS: test_attention_score_backward_matches_numerical_gradient "
          f"(Wq 전체 {d_model*d_k}개 원소가 수치 미분과 일치)")


def test_any_position_pair_connected_in_one_hop():
    """
    (검증 항목 3, 핵심) attention score 행렬 S에서, 위치 i와 위치 j 사이의 연결이
    시퀀스 길이나 |i-j|와 무관하게 항상 "한 번의 행렬곱"으로 이루어지는지 확인.

    RNN(6단계)이라면 h_0의 정보가 h_5에 도달하려면 step()이 5번 호출되어야 했지만,
    attention은 S[0, 5]가 X[0]과 X[5]로부터 단 한 번의 연산(Q@K.T)으로 계산된다.
    """
    seq_len = 10
    model = AttentionScore(d_model=4, d_k=3, seed=1)
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(seq_len, 4))

    S = model.forward(X)

    def graph_depth(node, visited=None):
        """node로부터 _prev를 재귀적으로 따라 올라가며, 계산 그래프의 최대 깊이를 측정"""
        if visited is None:
            visited = set()
        if id(node) in visited or not node._prev:
            return 0
        visited.add(id(node))
        return 1 + max((graph_depth(child, visited) for child in node._prev), default=0)

    depth = graph_depth(S)
    print(f"PASS: test_any_position_pair_connected_in_one_hop "
          f"(seq_len={seq_len}, S 전체를 만드는 계산 그래프 깊이={depth})")

    # 그래프 깊이가 seq_len과 무관하게 작은 상수 수준(행렬곱 몇 단계)에 머무는지 확인
    # (RNN이라면 seq_len에 비례해서 깊이가 계속 늘어났을 것)
    assert depth < seq_len, (
        f"그래프 깊이({depth})가 시퀀스 길이({seq_len})에 비례해서 커짐 -- attention의 이점이 사라짐"
    )


if __name__ == "__main__":
    test_transpose_backward_matches_numerical_gradient()
    test_attention_score_backward_matches_numerical_gradient()
    test_any_position_pair_connected_in_one_hop()
    print("\n모든 테스트 통과.")