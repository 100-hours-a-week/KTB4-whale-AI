"""
8단계: scaled dot-product attention (Q, K, V 전부 사용).

7단계 AttentionScore와의 핵심 차이:
    7단계: S = Q @ K.T                         -- 유사도 점수만 계산, 정보 이동 없음
    8단계: O = softmax(S / sqrt(d_k)) @ V      -- 유사도에 비례해 V(실제 정보)를 섞음

추가된 3가지 요소:
    1. V = X @ Wv       -- 각 위치가 "실제로 전달할 정보" (7단계에는 없었음)
    2. scaling (1/sqrt(d_k))  -- d_k가 커질수록 내적값의 분산이 커져 softmax가
                                 한쪽으로 치우치는(saturate) 문제를 완화
    3. softmax          -- S를 "가중치처럼 합이 1이 되는 분포"로 정규화
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue


class ScaledDotProductAttention:
    def __init__(self, d_model: int, d_k: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.5
        self.Wq = NumpyValue(rng.uniform(-scale, scale, size=(d_model, d_k)))
        self.Wk = NumpyValue(rng.uniform(-scale, scale, size=(d_model, d_k)))
        self.Wv = NumpyValue(rng.uniform(-scale, scale, size=(d_model, d_k)))
        self.d_k = d_k

    def parameters(self) -> list:
        return [self.Wq, self.Wk, self.Wv]

    def forward(self, X: np.ndarray) -> tuple:
        """
        X: (seq_len, d_model)

        Returns:
            (O, A): O -- 최종 출력 (seq_len, d_k), A -- attention weight 행렬 (seq_len, seq_len)
                    A를 함께 반환하는 이유는, "각 위치가 다른 위치를 얼마나 참조했는지"를
                    직접 눈으로 확인하기 위함이다 (7단계의 S와 달리, 이제 합이 1인 가중치다).
        """
        X_val = X if isinstance(X, NumpyValue) else NumpyValue(X)
        Q = X_val @ self.Wq
        K = X_val @ self.Wk
        V = X_val @ self.Wv

        S = Q @ K.transpose()
        S_scaled: NumpyValue = S * (1.0 / np.sqrt(self.d_k))   # scaling
        A = S_scaled.softmax(axis=-1)               # softmax -> 가중치 (각 행의 합 = 1)
        O = A @ V                                    # 가중합으로 V(실제 정보)를 섞음

        return O, A

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)