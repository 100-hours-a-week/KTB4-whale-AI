"""
11단계: DecoderLayer -- 9단계 MultiHeadAttention에 위치 정보, 학습 안정화 장치,
FFN(위치별 비선형 재가공)을 추가.

구성 순서:
    1. X + positional_encoding(X)        -- 위치 정보 주입
    2. attention_output = MultiHeadAttention(X_with_pos)
    3. X_with_pos + attention_output     -- residual connection 1
    4. LayerNorm(위 결과)                -- 정규화 1
    5. ffn_output = FeedForward(정규화 결과)
    6. 정규화 결과 + ffn_output           -- residual connection 2
    7. LayerNorm(위 결과)                -- 정규화 2
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue
from step_09_multi_head_attention.multi_head_attention import MultiHeadAttention
from step_10_decoder_layer.positional_residual_layernorm import positional_encoding, residual_connection, LayerNorm
from feed_forward import FeedForward


class DecoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seed: int = 42):
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=seed)
        self.norm1 = LayerNorm(d_model=d_model)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, seed=seed + 1000)
        self.norm2 = LayerNorm(d_model=d_model)
        self.d_model = d_model

    def parameters(self) -> list:
        return (self.attention.parameters() + self.norm1.parameters()
                + self.ffn.parameters() + self.norm2.parameters())

    def forward(self, X) -> NumpyValue:
        """
        X: np.ndarray(첫 번째 층인 경우) 또는 NumpyValue(이전 층의 출력인 경우) 둘 다 받는다.
        isinstance 체크로, 이미 NumpyValue(계산 그래프 포함)면 그대로 쓰고,
        아니면 새로 잎 노드를 만든다 -- 여러 층을 쌓을 때 그래프가 끊기지 않도록 한다.
        """
        X_val = X if isinstance(X, NumpyValue) else NumpyValue(X)
        seq_len = X_val.data.shape[0]
        pe = positional_encoding(seq_len, self.d_model)

        X_with_pos = X_val + pe   # NumpyValue.__add__가 raw pe를 자동으로 NumpyValue(pe)로 승격

        attention_output, _ = self.attention.forward(X_with_pos)
        residual1 = residual_connection(X_with_pos, attention_output)
        normed1 = self.norm1.forward(residual1)

        ffn_output = self.ffn.forward(normed1)
        residual2 = residual_connection(normed1, ffn_output)
        output = self.norm2.forward(residual2)

        return output

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)