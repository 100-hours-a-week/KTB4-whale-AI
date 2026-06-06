import numpy as np
from types import SimpleNamespace
from preprocess import _sigmoid_function

def train_perceptron(index, inputs, outputs, W1, b1, W2, b2, learning_rate, activate_f):
    x = inputs[index:index+1]                    # (1, 2)
    y = outputs[index:index+1].reshape(1, 1)    # (1, 1)

    # 1. Forward
    z1 = np.dot(x, W1) + b1 # (1, hidden_size)
    a1 = activate_f(z1)
    
    z2 = np.dot(a1, W2) + b2 # (1, 1)
    a2 = activate_f(z2) # 출력

    # Loss (MSE)
    loss = np.mean((a2 - y) ** 2) # 손실(실제값 - 예측값) 계산

    # Backward (Chain Rule)
    # 출력층 gradient
    dL_da2 = 2 * (a2 - y) / y.shape[0] # 출력층에서 손실(Loss)을 a2(예측값)에 대해 미분 → 오차 크기 계산
    dL_dz2 = dL_da2 * _sigmoid_derivative(z2) # dL/da2 × sigmoid 미분
    dW2 = np.dot(a1.T, dL_dz2) # 출력층 가중치 W2에 대한 gradient 계산
    db2 = np.sum(dL_dz2, axis=0, keepdims=True) # 출력층 편향 b2에 대한 gradient 계산

    # 은닉층 gradient
    dL_da1 = np.dot(dL_dz2, W2.T) # 은닉층으로 오차를 전달 (Backpropagation 핵심)
    dL_dz1 = dL_da1 * _sigmoid_derivative(z1) # 은닉층에서 gradient 계산 (Chain Rule)
    dW1 = np.dot(x.T, dL_dz1) # 입력 → 은닉층 가중치 W1에 대한 gradient
    db1 = np.sum(dL_dz1, axis=0, keepdims=True) # 은닉층 편향 b1에 대한 gradient
    

    # Weight, Bias Update (Backward + Optimizer)
    W1 = W1 - learning_rate * dW1   # W1 업데이트
    b1 = b1 - learning_rate * db1   # b1 업데이트
    W2 = W2 - learning_rate * dW2   # W2 업데이트
    b2 = b2 - learning_rate * db2   # b2 업데이트

    return SimpleNamespace(W1=W1, b1=b1, W2=W2, b2=b2)

def _sigmoid_derivative(x):
    s = _sigmoid_function(x)
    return s * (1 - s)