"""
11단계: TransformerDecoder -- DecoderLayer를 여러 개 쌓은 전체 구조.

지금까지 1~10단계에서 만든 요소가 전부 여기에 조합된다:
    1단계 forward-loss-backward-update 사이클
    2단계 activation (sigmoid, 여기서는 ReLU)
    3단계 MLP (FFN이 곧 3단계 MLP를 위치별로 적용한 것)
    4단계 연산자 오버로딩 기반 자동 미분 (NumpyValue)
    5단계 벡터/행렬 배치 연산 (NumPy)
    6단계 hidden state 개념은 attention으로 대체됨 (7~9단계)
    7~9단계 attention score, scaled dot-product attention, multi-head attention
    10단계 positional encoding, residual, LayerNorm
    11단계 (지금) 층 쌓기
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue
from decoder_layer import DecoderLayer


class TransformerDecoder:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, seed: int = 42):
        self.layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, seed=seed + i * 100)
            for i in range(num_layers)
        ]
        self.num_layers = num_layers

    def parameters(self) -> list:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, X: np.ndarray) -> NumpyValue:
        """
        층을 순차적으로 통과시킨다. 각 층의 출력(NumpyValue, 계산 그래프 포함)을
        다음 층에 그대로 전달한다.
        """
        current = X
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)