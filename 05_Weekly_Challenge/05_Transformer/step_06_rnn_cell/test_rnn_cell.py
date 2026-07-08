"""
RNNCell의 backward가 정확한지 검증.

핵심 검증 대상: Whh(hidden-to-hidden weight)의 gradient.
이 파라미터는 모든 시점(time step)에서 반복 사용되므로, 그 gradient는
여러 시점에서 온 기여가 전부 합산된 값이어야 한다 (BPTT, Backpropagation Through Time).
"""

import numpy as np
from rnn_cell import RNNCell


def numerical_gradient_for_Whh(rnn: RNNCell, sequence: list, targets: list,
                                entry_index: tuple, h: float = 1e-5) -> float:
    i, j = entry_index
    original = rnn.Whh.data[i, j]

    rnn.Whh.data[i, j] = original + h
    outputs_plus = rnn.forward(sequence)
    loss_plus = float(rnn.compute_loss(outputs_plus, targets).data)

    rnn.Whh.data[i, j] = original - h
    outputs_minus = rnn.forward(sequence)
    loss_minus = float(rnn.compute_loss(outputs_minus, targets).data)

    rnn.Whh.data[i, j] = original
    return (loss_plus - loss_minus) / (2 * h)


def test_rnn_backward_through_time_matches_numerical_gradient():
    """
    시퀀스 길이 4짜리 입력에 대해, Whh의 모든 원소의 gradient가
    수치 미분과 일치하는지 확인. 이게 통과해야 BPTT가 정확하다는 증거가 된다.
    """
    rnn = RNNCell(input_size=1, hidden_size=2, output_size=1, seed=1)

    sequence = [np.array([[1.0]]), np.array([[0.0]]), np.array([[1.0]]), np.array([[1.0]])]
    targets = [np.array([[1.0]]), np.array([[1.0]]), np.array([[0.0]]), np.array([[1.0]])]

    rnn.zero_grad()
    outputs = rnn.forward(sequence)
    loss = rnn.compute_loss(outputs, targets)
    loss.backward()

    hidden_size = rnn.hidden_size
    for i in range(hidden_size):
        for j in range(hidden_size):
            numerical = numerical_gradient_for_Whh(rnn, sequence, targets, (i, j))
            analytical = rnn.Whh.grad[i, j]
            assert abs(numerical - analytical) < 1e-3, (
                f"Whh[{i},{j}] 불일치: analytical={analytical}, numerical={numerical}"
            )
    print(f"PASS: test_rnn_backward_through_time_matches_numerical_gradient "
          f"(Whh 전체 {hidden_size*hidden_size}개 원소가 4개 시점에 걸친 BPTT gradient와 일치)")


if __name__ == "__main__":
    test_rnn_backward_through_time_matches_numerical_gradient()
    print("\n모든 테스트 통과.")