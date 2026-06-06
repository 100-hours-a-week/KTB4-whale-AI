import numpy as np
from types import SimpleNamespace

def preprocess(inputs, outputs, weights_num, bias_num, learning_rate, activate = 'step'):
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
    weights = np.random.rand(weights_num) # [0, 1) 사이 균등 분포(Uniform Distribution)에서 난수 선택
    bias = np.random.rand(bias_num)

    # activate_f
    activate_f = _step_function

    return SimpleNamespace(
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        weights=weights,
        bias=bias,
        learning_rate=learning_rate,
        activate_f=activate_f
    )

def _step_function(x):

    return 1 if x >= 0 else 0

# TODO: 변경 필요
# def _sigmoid_function(x):
#     return 1 if x >= 0 else 0

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