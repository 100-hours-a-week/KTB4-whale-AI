import numpy as np
from preprocess import preprocess
from train import train
from evaluate import evaluate

"""
    데이터 정의 (XOR 연산 추론하기)
    - 입력/출력 데이터 입수
    - 가중치와 편향 크기 설정
    - 학습률, 에포크 설정
    - 활성화 함수 설정
"""
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 입력 데이터 입수
outputs = np.array([0, 1, 1, 0]) # 출력 데이터 입수
weights_num = 2
bias_num = 1
hidden_size = 2 # 은닉층 추가
learning_rate = 0.5
epochs = 20000
activate = 'sigmoid' # 활성화 함수 설정

# =================== Work Process (작업 과정) ===================
# 1. Preprocess
pre = preprocess(inputs, outputs, weights_num, bias_num, hidden_size, learning_rate, activate)

# 2. Training
trained = train(
        inputs=pre.train_inputs,
        outputs=pre.train_outputs,
        epochs=epochs,
        W1=pre.W1, b1=pre.b1,                 # [변경]
        W2=pre.W2, b2=pre.b2,                 # [변경]
        learning_rate=pre.learning_rate,
        activate_f=pre.activate_f
    )

# 3. Evaluate
evaluate(pre.test_inputs, pre.test_outputs, trained.W1, trained.b1, trained.W2, trained.b2)