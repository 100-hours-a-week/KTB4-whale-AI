"""
(After 2) 실제 PyTorch의 autograd로 동일한 계산을 재현하고,
3단계 수동 backward, 4단계 자체 구현 Value 엔진과 gradient가 일치하는지 확인한다.

PyTorch가 라이브러리 내부에서 하는 일이, 우리가 value.py에 구현한 것과
원리적으로 동일하다는 것(연산자 오버로딩 + 계산 그래프 역순 순회)을 확인하는 게 목적이다.
"""

import torch

from step_03_mlp.mlp import MLP as ManualMLP


def test_pytorch_matches_manual_backward():
    manual = ManualMLP(seed=7)
    x1, x2, y_true = 1.0, 0.0, 1.0

    # --- 3단계: 수동 backward ---
    o_manual, cache = manual.forward(x1, x2)
    manual_gradients = manual.backward(cache, y_true)

    # --- PyTorch: 동일한 가중치로 텐서 구성, requires_grad=True로 자동 미분 활성화 ---
    w1_1 = torch.tensor(manual.hidden1.weights[0], requires_grad=True)
    w1_2 = torch.tensor(manual.hidden1.weights[1], requires_grad=True)
    b1 = torch.tensor(manual.hidden1.bias, requires_grad=True)

    w2_1 = torch.tensor(manual.hidden2.weights[0], requires_grad=True)
    w2_2 = torch.tensor(manual.hidden2.weights[1], requires_grad=True)
    b2 = torch.tensor(manual.hidden2.bias, requires_grad=True)

    v1 = torch.tensor(manual.output.weights[0], requires_grad=True)
    v2 = torch.tensor(manual.output.weights[1], requires_grad=True)
    c = torch.tensor(manual.output.bias, requires_grad=True)

    x1_t = torch.tensor(x1)
    x2_t = torch.tensor(x2)

    # forward pass — 3단계 mlp.py의 forward()와 완전히 동일한 수식
    z_h1 = w1_1 * x1_t + w1_2 * x2_t + b1
    h1 = torch.sigmoid(z_h1)

    z_h2 = w2_1 * x1_t + w2_2 * x2_t + b2
    h2 = torch.sigmoid(z_h2)

    z_o = v1 * h1 + v2 * h2 + c
    o = torch.sigmoid(z_o)

    loss = (o - y_true) ** 2
    loss.backward()  # PyTorch autograd가 전체 gradient를 자동 계산

    # forward 결과 일치 확인
    assert abs(o_manual - o.item()) < 1e-6, f"forward 결과 불일치: {o_manual} vs {o.item()}"

    # output layer gradient 비교
    dL_dv1, dL_dv2, dL_dc = manual_gradients['output']
    assert abs(dL_dv1 - v1.grad.item()) < 1e-6, f"dL_dv1 불일치: manual={dL_dv1}, pytorch={v1.grad.item()}"
    assert abs(dL_dv2 - v2.grad.item()) < 1e-6, f"dL_dv2 불일치: manual={dL_dv2}, pytorch={v2.grad.item()}"
    assert abs(dL_dc - c.grad.item()) < 1e-6, f"dL_dc 불일치: manual={dL_dc}, pytorch={c.grad.item()}"

    # hidden1 gradient 비교
    dL_dw1_1, dL_dw1_2, dL_db1 = manual_gradients['hidden1']
    assert abs(dL_dw1_1 - w1_1.grad.item()) < 1e-6, f"dL_dw1_1 불일치: {dL_dw1_1} vs {w1_1.grad.item()}"
    assert abs(dL_dw1_2 - w1_2.grad.item()) < 1e-6, f"dL_dw1_2 불일치: {dL_dw1_2} vs {w1_2.grad.item()}"
    assert abs(dL_db1 - b1.grad.item()) < 1e-6, f"dL_db1 불일치: {dL_db1} vs {b1.grad.item()}"

    # hidden2 gradient 비교
    dL_dw2_1, dL_dw2_2, dL_db2 = manual_gradients['hidden2']
    assert abs(dL_dw2_1 - w2_1.grad.item()) < 1e-6, f"dL_dw2_1 불일치: {dL_dw2_1} vs {w2_1.grad.item()}"
    assert abs(dL_dw2_2 - w2_2.grad.item()) < 1e-6, f"dL_dw2_2 불일치: {dL_dw2_2} vs {w2_2.grad.item()}"
    assert abs(dL_db2 - b2.grad.item()) < 1e-6, f"dL_db2 불일치: {dL_db2} vs {b2.grad.item()}"

    print("PASS: test_pytorch_matches_manual_backward")
    print(f"  output:  v1={dL_dv1:.6f}=={v1.grad.item():.6f}, "
          f"v2={dL_dv2:.6f}=={v2.grad.item():.6f}, c={dL_dc:.6f}=={c.grad.item():.6f}")
    print(f"  hidden1: w1_1={dL_dw1_1:.6f}=={w1_1.grad.item():.6f}, "
          f"w1_2={dL_dw1_2:.6f}=={w1_2.grad.item():.6f}, b1={dL_db1:.6f}=={b1.grad.item():.6f}")
    print(f"  hidden2: w2_1={dL_dw2_1:.6f}=={w2_1.grad.item():.6f}, "
          f"w2_2={dL_dw2_2:.6f}=={w2_2.grad.item():.6f}, b2={dL_db2:.6f}=={b2.grad.item():.6f}")


if __name__ == "__main__":
    test_pytorch_matches_manual_backward()
    print("\n모든 테스트 통과: 3단계 수동 backward == PyTorch autograd (완전 일치)")