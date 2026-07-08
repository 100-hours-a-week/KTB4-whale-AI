"""
11단계: FFN (Feed-Forward Network, 위치별 순방향 신경망).

핵심 개념 요약:
    attention(O = A @ V)은 "다른 위치의 정보를 얼마나 섞을지"만 정할 뿐,
    섞고 난 정보를 비선형적으로 재가공하지는 못한다 (A@V, X@Wv 전부 선형 연산).
    FFN은 이 문제를, 3단계에서 만든 MLP를 각 위치에 독립적으로(위치끼리는
    서로 무관하게) 적용해서 해결한다.

수식:
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    (3단계 MLP의 hidden layer -> output layer 구조와 완전히 동일,
     차이는 이 계산이 시퀀스의 각 위치마다 독립적으로, 동일한 W1/W2로 적용된다는 것)
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue


class FeedForward:
    def __init__(self, d_model: int, d_ff: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.5
        self.W1 = NumpyValue(rng.uniform(-scale, scale, size=(d_model, d_ff)))
        self.b1 = NumpyValue(np.zeros((1, d_ff)))
        self.W2 = NumpyValue(rng.uniform(-scale, scale, size=(d_ff, d_model)))
        self.b2 = NumpyValue(np.zeros((1, d_model)))

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: NumpyValue) -> NumpyValue:
        hidden = (x @ self.W1 + self.b1).relu()
        output = hidden @ self.W2 + self.b2
        return output

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)