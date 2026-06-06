import numpy as np
from preprocess import preprocess
from train import train
from evaluate import evaluate

"""
    데이터 정의 (AND 연산 추론하기)
    - 입력/출력 데이터 입수
    - 가중치와 편향 크기 설정
    - 학습률, 에포크 설정
    - 활성화 함수 설정
"""
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 입력 데이터 입수
outputs = np.array([0, 0, 0, 1]) # 출력 데이터 입수
weights_num = 2 # 가중치 크기 설정
bias_num = 1 # 편항 크기 설정
learning_rate = 0.1 # 학습률 설정, 각 학습 단계에서 오차(에러)로부터 변경할 가중치와 편향의 크기를 결정하는 하이퍼파라미터
epochs = 5 # 에포크 설정, 전체 데이터셋을 한 번 학습하는 단위
activate = 'step' # 활성화 함수 설정

# =================== Work Process (작업 과정) ===================
# 1. Preprocess
pre = preprocess(inputs, outputs, weights_num, bias_num, learning_rate, activate)

# 2. Training
trained = train(
        inputs=pre.train_inputs,
        outputs=pre.train_outputs,
        epochs=epochs,
        weights=pre.weights,
        bias=pre.bias,
        learning_rate=pre.learning_rate,
        activate_f=pre.activate_f
    )

# 3. Evaluate
evaluate(pre.test_inputs, pre.test_outputs, trained.weights, trained.bias)