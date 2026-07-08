"""
2단계 최소 단위: sigmoid activation function(활성화 함수)과 그 미분(derivative) 구현.

라이브러리 의존성: math (지수함수 계산용) 외 없음.
"""

import math


def sigmoid(z: float) -> float:
    """
    sigmoid(z) = 1 / (1 + e^(-z))

    출력 범위: 항상 (0, 1) 사이.
    """
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(z: float) -> float:
    """
    sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))

    입력을 z 그대로 받아 내부에서 sigmoid(z)를 계산한다.
    (backward 단계에서 activation 이후 값(a)을 이미 갖고 있다면
     a*(1-a)로 바로 계산하는 것이 더 효율적이지만,
     여기서는 함수의 정의를 명확히 하기 위해 z를 입력으로 받는다.)
    """
    s = sigmoid(z)
    return s * (1 - s)