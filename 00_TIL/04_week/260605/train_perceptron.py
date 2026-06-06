import numpy as np
from types import SimpleNamespace

def train_perceptron(index, inputs, outputs, weights, bias, learning_rate, activate_f):
    # 1. Forward
    total_input = np.dot(inputs[index], weights) + bias # 총 입력(선형 함수) 계산
    prediction = activate_f(total_input) # 예측값 계산

    # Loss Signal
    loss = outputs[index] - prediction # 손실(실제값 - 예측값) 계산
    # loss > 0: 예측이 부족, weight, bias 증가
    # loss < 0: 예측이 과함, weight, bias 감소
    print(f"\n[Sample {index+1}] 입력: {inputs[index]} | 정답: {outputs[index]}")
    print(f"  현재 weights = {weights.round(4)}, bias = {bias.round(4)}")
    print(f"  total_input = {total_input.round(4)}")
    print(f"  prediction  = {prediction}")
    print(f"  loss        = {loss:.4f}")

    # Weight(with Bias) Update (Backward + Optimizer)
    # Optimizer: 고정된 learning_rate를 사용한 확률적 경사하강법(SGD) 스타일
    update_weights = weights + learning_rate * loss * inputs[index]
    update_bias = bias + learning_rate * loss
    print(f"→ weights += {learning_rate} * {loss} * {inputs[index]}")
    print(f"→ bias += {learning_rate} * {loss}")
    print(f"→ 업데이트 후 new_weights = {update_weights.round(4)}, new_bias = {update_bias.round(4)}")

    return SimpleNamespace(
        weights=update_weights,
        bias=update_bias
    )