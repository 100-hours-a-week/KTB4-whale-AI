import numpy as np
from types import SimpleNamespace

class Preprocessor:
    """
        데이터 전처리
        - 가중치, 편향 초기화
        - 활성화 함수 선택
    """

    def __init__(self):
        pass

    def activate_function(self, x, activate_type = 'sigmoid'):
        if activate_type == 'sigmoid':
            return self._sigmoid(x)
        else: # sigmoid
            return self._sigmoid(x)
    
    def derivative_function(self, x, type = 'sigmoid'):
        if type == 'sigmoid':
            return self._sigmoid_derivative(x)
        else:
            return self._sigmoid_derivative(x)
    
    def initialize_weights(self, input_size, hidden_size, output_size):
        np.random.seed(42) # 재현성을 위해 시드 고정
        SCALING_SIZE = 0.5
        # 입력 → 은닉층
        W1 = np.random.randn(input_size, hidden_size) * SCALING_SIZE
        b1 = np.zeros((1, hidden_size))

        # 은닉층 → 출력층
        W2 = np.random.randn(hidden_size, output_size) * SCALING_SIZE
        b2 = np.zeros((1, output_size))

        return { "W1": W1, "b1": b1, "W2": W2, "b2": b2 }

    # activate
    def _step_function(self, x):
        return 1 if x >= 0 else 0
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Overflow 방지를 위해 (-500 ~ +500)

    # derivative
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)
    
    # TODO: 변경 필요
    # def _tanh(x):
    #     return 1 if x >= 0 else 0

    # def _relu(x):
    #     return 1 if x >= 0 else 0

    # def _leaky_relu(x):
    #     return 1 if x >= 0 else 0

    # def _gelu(x):
    #     return 1 if x >= 0 else 0

    # def _silu(x):
    #     return 1 if x >= 0 else 0

    