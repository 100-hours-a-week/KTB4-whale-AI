"""
10단계: positional encoding + residual connection + LayerNorm.

9단계 MultiHeadAttention까지의 한계:
    - 위치 정보(순서)를 전혀 반영하지 않음 -> positional encoding으로 해결
    - 층을 깊게 쌓으면(이후 11단계) 학습이 불안정해짐 -> residual + LayerNorm으로 완화

세 요소는 서로 다른 문제를 풀며, 서로 독립적으로 이해할 수 있다.
"""

import numpy as np
import math
from step_05_vectorization.value_numpy import NumpyValue


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    개념 요약:
        attention은 순서를 모른다 -- 입력을 뒤섞어도 attention score 자체는
        (행 순서만 바뀔 뿐) 똑같이 계산된다. positional encoding은 위치마다
        고유한 패턴(sin/cos 파형)을 만들어서, 입력에 "몇 번째 위치인지"라는
        정보를 실어 보낸다.

    수식:
        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
        (짝수 차원은 sin, 홀수 차원은 cos -- 서로 다른 주기의 파형을 섞어서
         위치마다 겹치지 않는 고유한 패턴을 만든다)
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


class LayerNorm:
    """
    개념 요약:
        층이 깊어질수록(11단계에서 여러 층을 쌓으면), 각 층을 통과할 때마다
        값의 크기(scale)가 들쭉날쭉해져서 학습이 불안정해질 수 있다.
        LayerNorm은 각 위치(행)마다, feature 축(열) 방향으로 평균 0, 분산 1이
        되도록 다시 맞춰서 이 문제를 완화한다.

    수식:
        y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
        (gamma, beta는 학습 가능한 파라미터 -- 정규화 후 다시 적절한 크기로
         조정할 여지를 모델에게 남겨준다)
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = NumpyValue(np.ones((1, d_model)))
        self.beta = NumpyValue(np.zeros((1, d_model)))
        self.eps = eps

    def parameters(self) -> list:
        return [self.gamma, self.beta]

    def forward(self, x: NumpyValue) -> NumpyValue:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x + (-1 * mean)
        variance = (centered * centered).mean(axis=-1, keepdims=True)
        std = (variance + self.eps).sqrt()
        normalized = centered / std
        return normalized * self.gamma + self.beta


def residual_connection(x: NumpyValue, sublayer_output: NumpyValue) -> NumpyValue:
    """
    개념 요약:
        층이 깊어지면, backward 시 gradient가 여러 층을 거치며 점점 작아지거나
        (vanishing) 커지는(exploding) 문제가 생길 수 있다. residual connection은
        "이 층이 계산한 것"에 "원래 입력"을 그대로 더해서, 입력에서 출력으로
        가는 지름길(shortcut)을 하나 더 만들어준다.

    수식:
        output = x + sublayer(x)
        (sublayer가 아무것도 유용한 걸 학습하지 못해도, 최소한 x는 그대로
         보존되어 다음 층으로 전달된다는 게 핵심)
    """
    return x + sublayer_output