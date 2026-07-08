"""
10단계: DecoderLayer -- 9단계 MultiHeadAttention에 위치 정보와 학습 안정화 장치를 추가.

구성 순서:
    1. X + positional_encoding(X)        -- 위치 정보 주입
    2. attention_output = MultiHeadAttention(X_with_pos)
    3. X_with_pos + attention_output     -- residual connection
    4. LayerNorm(위 결과)                -- 정규화
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue
from step_09_multi_head_attention.multi_head_attention import MultiHeadAttention
from positional_residual_layernorm import positional_encoding, residual_connection, LayerNorm


class DecoderLayer:
    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=seed)
        self.norm = LayerNorm(d_model=d_model)
        self.d_model = d_model

    def parameters(self) -> list:
        return self.attention.parameters() + self.norm.parameters()

    def forward(self, X: np.ndarray) -> NumpyValue:
        seq_len = X.shape[0]
        pe = positional_encoding(seq_len, self.d_model)

        X_with_pos_data = X + pe        # 위치 정보 주입 (아직 순수 NumPy 배열 단계)
        X_with_pos = NumpyValue(X_with_pos_data)

        attention_output, _ = self.attention.forward(X_with_pos_data)
        residual_output = residual_connection(X_with_pos, attention_output)
        output = self.norm.forward(residual_output)

        return output

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)