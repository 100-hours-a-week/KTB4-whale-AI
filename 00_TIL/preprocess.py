import numpy as np
from types import SimpleNamespace

def preprocess(inputs, outputs, weights_num, bias_num, hidden_size, learning_rate, activate = 'step'):
    """
        데이터 전처리
        - 데이터셋 분할
        - 가중치, 편향 초기화
        - 활성화 함수 선택
            - 인공신경망에서 뉴런의 출력을 결정하는 "비선형" 함수
    """

    # inputs, outputs
    # TODO: train, test 데이터셋 분할
    train_inputs = inputs
    train_outputs = outputs
    test_inputs = inputs
    test_outputs = outputs
    
    # weights, bias 초기화
    np.random.seed(42) # 재현성을 위해 시드 고정
    # 입력 → 은닉층
    W1 = np.random.randn(weights_num, hidden_size) * 0.5
    b1 = np.zeros((1, hidden_size))

    # 은닉층 → 출력층
    W2 = np.random.randn(hidden_size, 1) * 0.5
    b2 = np.zeros((1, 1))

    # activate_f
    activate_f = _sigmoid_function

    return SimpleNamespace(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        W1=W1, b1=b1,                     # [변경]
        W2=W2, b2=b2,                     # [변경]
        learning_rate=learning_rate,
        activate_f=activate_f
    )

def _step_function(x):
    return 1 if x >= 0 else 0

def _sigmoid_function(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # overflow 방지

# TODO: 변경 필요
# def _tanh_function(x):
#     return 1 if x >= 0 else 0

# def _relu_function(x):
#     return 1 if x >= 0 else 0

# def _leaky_relu_function(x):
#     return 1 if x >= 0 else 0

# def _gelu_function(x):
#     return 1 if x >= 0 else 0

# def _silu_function(x):
#     return 1 if x >= 0 else 0