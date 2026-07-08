"""
9단계: multi-head attention.

8단계 ScaledDotProductAttention과의 핵심 차이:
    8단계: Wq, Wk, Wv 한 세트(head 1개)만 존재 -> 한 가지 관점의 관련성만 포착
    9단계: 세트를 num_heads개 병렬로 두고, 각자 독립적으로 attention을 계산한 뒤
           결과를 이어 붙이고(concat), 최종 선형 변환(Wo)으로 하나로 합침

핵심 수식:
    head_i = ScaledDotProductAttention_i(X)         (i = 1 ... num_heads, 각자 다른 Wq/Wk/Wv)
    O_concat = concat(head_1, ..., head_num_heads)  (열 방향으로 이어 붙임)
    O = O_concat @ Wo                                (모든 head의 결과를 다시 하나로 섞음)
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue
from step_08_scaled_dot_product_attention.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int, seed: int = 42):
        assert d_model % num_heads == 0, "d_model은 num_heads로 나누어떨어져야 한다"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # head 1개가 담당하는 차원 (8단계보다 작아짐)

        # head마다 서로 다른 초기값을 갖도록 seed를 다르게 준다.
        # (2단계에서 은닉층 뉴런 2개에 서로 다른 seed를 준 것과 같은 이유 --
        #  같은 seed면 모든 head가 완전히 동일하게 시작해서 서로 다른 관점을 학습할 수 없다)
        self.heads = [
            ScaledDotProductAttention(d_model=d_model, d_k=self.d_k, seed=seed + i)
            for i in range(num_heads)
        ]
        self.Wo = NumpyValue(
            np.random.default_rng(seed + num_heads).uniform(-0.5, 0.5, size=(d_model, d_model))
        )

    def parameters(self) -> list:
        params = []
        for head in self.heads:
            params.extend(head.parameters())
        params.append(self.Wo)
        return params

    def forward(self, X: np.ndarray) -> tuple:
        """
        X: (seq_len, d_model)

        Returns:
            (O, head_outputs): O -- 최종 출력 (seq_len, d_model)
                               head_outputs -- [(O_1, A_1), (O_2, A_2), ...] 각 head의 결과
                               (각 head가 실제로 다른 attention 패턴을 보이는지 확인하기 위함)
        """
        head_outputs = [head.forward(X) for head in self.heads]
        head_O_values = [O for O, A in head_outputs]

        O_concat = NumpyValue.concat(head_O_values, axis=-1)  # (seq_len, d_model)
        O = O_concat @ self.Wo

        return O, head_outputs

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)