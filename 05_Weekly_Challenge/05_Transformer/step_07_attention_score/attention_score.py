"""
7단계: attention score 계산 (Q, K만 사용, V·마스킹 없음).

6단계 RNNCell.step()과의 핵심 차이:
    6단계: h_t 계산에 h_{t-1}이 필요 -> t번째 위치가 0번째 위치의 정보를 얻으려면
           t번의 순차 step()을 거쳐야 함 (사슬, chain)
    7단계: 모든 위치 쌍의 유사도를 Q @ K.T 한 번의 행렬곱으로 동시에 계산
           -> i번째 위치가 j번째 위치의 정보를 "직접" 참조 (사슬 없음)

핵심 수식:
    Q = X @ Wq   (query, 각 위치가 "무엇을 찾고 있는지")
    K = X @ Wk   (key, 각 위치가 "무엇을 제공하는지")
    S = Q @ K.T  (attention score, i번째 query와 j번째 key의 내적 유사도)

이번 단계는 V(value)와 softmax, masking을 포함하지 않는다 -- 순수하게
"내적으로 유사도를 재는" 개념만 분리해서 확인하는 게 목적이다 (8단계에서 V·softmax 추가).
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue


class AttentionScore:
    def __init__(self, d_model: int, d_k: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.5
        self.Wq = NumpyValue(rng.uniform(-scale, scale, size=(d_model, d_k)))
        self.Wk = NumpyValue(rng.uniform(-scale, scale, size=(d_model, d_k)))

    def parameters(self) -> list:
        return [self.Wq, self.Wk]

    def forward(self, X: np.ndarray) -> NumpyValue:
        """
        X: (seq_len, d_model) -- 시퀀스 전체를 한 번에 받는다 (6단계처럼 시점별 반복 없음).

        Returns:
            S: (seq_len, seq_len) -- S[i, j] = i번째 위치와 j번째 위치의 유사도.
        """
        X_val = NumpyValue(X)
        Q = X_val @ self.Wq
        K = X_val @ self.Wk
        S = Q @ K.transpose()
        return S

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)