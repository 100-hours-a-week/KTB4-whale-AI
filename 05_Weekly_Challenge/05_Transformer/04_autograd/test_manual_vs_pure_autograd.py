"""
핵심 검증: 3단계(mlp.py, 수동 backward)와 4단계(mlp_autograd.py, Value 기반 autograd)가
동일한 초기 가중치, 동일한 입력에 대해 정확히 같은 gradient를 계산하는지 대조한다.

이 테스트가 통과해야 "backward를 손으로 유도한 결과"와 "autograd가 자동 계산한 결과"가
동일하다는 것이 증명되고, 4단계에서 autograd로 전환해도 안전하다는 근거가 된다.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'step3_mlp'))

from mlp import MLP as ManualMLP
from mlp_pure_autograd import MLP as AutogradMLP


def sync_weights(manual: ManualMLP, auto: AutogradMLP):
    """두 MLP가 정확히 같은 가중치를 갖도록 autograd MLP에 manual MLP의 값을 복사"""
    auto.hidden1.weights[0].data = manual.hidden1.weights[0]
    auto.hidden1.weights[1].data = manual.hidden1.weights[1]
    auto.hidden1.bias.data = manual.hidden1.bias

    auto.hidden2.weights[0].data = manual.hidden2.weights[0]
    auto.hidden2.weights[1].data = manual.hidden2.weights[1]
    auto.hidden2.bias.data = manual.hidden2.bias

    auto.output.weights[0].data = manual.output.weights[0]
    auto.output.weights[1].data = manual.output.weights[1]
    auto.output.bias.data = manual.output.bias


def test_manual_and_autograd_gradients_match():
    manual = ManualMLP(seed=7)
    auto = AutogradMLP(seed=7)
    sync_weights(manual, auto)  # 혹시 모를 초기화 차이를 제거하고 완전히 동일한 가중치로 맞춤

    x1, x2, y_true = 1.0, 0.0, 1.0

    # --- 3단계: 수동 backward ---
    o_manual, cache = manual.forward(x1, x2)
    manual_gradients = manual.backward(cache, y_true)

    # --- 4단계: autograd ---
    auto.zero_grad()
    o_auto = auto.forward(x1, x2)
    loss = (o_auto - y_true) ** 2
    loss.backward()

    # forward 결과 자체가 같은지 먼저 확인 (가중치가 정말 동일했는지 검증)
    assert abs(o_manual - o_auto.data) < 1e-9, f"forward 결과 불일치: {o_manual} vs {o_auto.data}"

    # output layer gradient 비교
    dL_dv1, dL_dv2, dL_dc = manual_gradients['output']
    assert abs(dL_dv1 - auto.output.weights[0].grad) < 1e-9, \
        f"dL_dv1 불일치: manual={dL_dv1}, autograd={auto.output.weights[0].grad}"
    assert abs(dL_dv2 - auto.output.weights[1].grad) < 1e-9, \
        f"dL_dv2 불일치: manual={dL_dv2}, autograd={auto.output.weights[1].grad}"
    assert abs(dL_dc - auto.output.bias.grad) < 1e-9, \
        f"dL_dc 불일치: manual={dL_dc}, autograd={auto.output.bias.grad}"

    # hidden1 gradient 비교
    dL_dw1_1, dL_dw1_2, dL_db1 = manual_gradients['hidden1']
    assert abs(dL_dw1_1 - auto.hidden1.weights[0].grad) < 1e-9, \
        f"dL_dw1_1 불일치: manual={dL_dw1_1}, autograd={auto.hidden1.weights[0].grad}"
    assert abs(dL_dw1_2 - auto.hidden1.weights[1].grad) < 1e-9, \
        f"dL_dw1_2 불일치: manual={dL_dw1_2}, autograd={auto.hidden1.weights[1].grad}"
    assert abs(dL_db1 - auto.hidden1.bias.grad) < 1e-9, \
        f"dL_db1 불일치: manual={dL_db1}, autograd={auto.hidden1.bias.grad}"

    # hidden2 gradient 비교
    dL_dw2_1, dL_dw2_2, dL_db2 = manual_gradients['hidden2']
    assert abs(dL_dw2_1 - auto.hidden2.weights[0].grad) < 1e-9, \
        f"dL_dw2_1 불일치: manual={dL_dw2_1}, autograd={auto.hidden2.weights[0].grad}"
    assert abs(dL_dw2_2 - auto.hidden2.weights[1].grad) < 1e-9, \
        f"dL_dw2_2 불일치: manual={dL_dw2_2}, autograd={auto.hidden2.weights[1].grad}"
    assert abs(dL_db2 - auto.hidden2.bias.grad) < 1e-9, \
        f"dL_db2 불일치: manual={dL_db2}, autograd={auto.hidden2.bias.grad}"

    print("PASS: test_manual_and_autograd_gradients_match")
    print(f"  output:  v1={dL_dv1:.6f}=={auto.output.weights[0].grad:.6f}, "
          f"v2={dL_dv2:.6f}=={auto.output.weights[1].grad:.6f}, "
          f"c={dL_dc:.6f}=={auto.output.bias.grad:.6f}")
    print(f"  hidden1: w1_1={dL_dw1_1:.6f}=={auto.hidden1.weights[0].grad:.6f}, "
          f"w1_2={dL_dw1_2:.6f}=={auto.hidden1.weights[1].grad:.6f}, "
          f"b1={dL_db1:.6f}=={auto.hidden1.bias.grad:.6f}")
    print(f"  hidden2: w2_1={dL_dw2_1:.6f}=={auto.hidden2.weights[0].grad:.6f}, "
          f"w2_2={dL_dw2_2:.6f}=={auto.hidden2.weights[1].grad:.6f}, "
          f"b2={dL_db2:.6f}=={auto.hidden2.bias.grad:.6f}")


if __name__ == "__main__":
    test_manual_and_autograd_gradients_match()
    print("\n모든 테스트 통과: 3단계 수동 backward == 4단계 autograd (완전 일치)")