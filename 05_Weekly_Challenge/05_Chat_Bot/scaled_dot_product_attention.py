from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot Product Attention
    - Attention 메커니즘의 가장 기본 단위
    - Q, K, V를 받아서 Attention을 계산
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        
        d_k = query.size(-1) # Key의 마지막 차원 크기

        # 1. Q와 K의 유사도 계산 (dot product)
        # (batch, seq, d_k) @ (batch, d_k, seq) -> (batch, seq, seq)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # 2. Scaling (Scores 안정화)
        scores = scores / math.sqrt(d_k)

        # 3. Mask(Causal Mask) 적용 (미래 단어 차단)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax로 Attention 가중치 계산
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Softmax 결과 * V -> 가중 평균
        # (batch, seq, seq) @ (batch, seq, d_v) -> (batch, seq, d_v)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights
