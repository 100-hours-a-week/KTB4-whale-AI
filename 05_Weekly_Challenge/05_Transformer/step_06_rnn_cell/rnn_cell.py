"""
6단계: RNN cell (Recurrent Neural Network cell, 순환 신경망 셀) 최소 구현.

5단계 BatchMLP와의 핵심 차이:
    5단계: forward(X) -- 배치의 각 행(샘플)을 서로 무관하게 한 번에 처리
    6단계: forward(sequence) -- 시퀀스를 시간 순서대로 하나씩 처리하되,
           이전 시점의 hidden state를 다음 시점 계산에 반드시 사용

핵심 수식:
    h_t = sigmoid(x_t @ Wxh + h_{t-1} @ Whh + bh)   -- 새 hidden state 계산
    y_t = sigmoid(h_t @ Why + by)                    -- 이 시점의 출력 계산

h_{t-1} @ Whh 항이 5단계에는 없던 신규 항이다. 이게 "이전 시점의 정보를
다음 시점으로 넘겨주는 통로"이며, RNN의 정의 그 자체다.
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue


class RNNCell:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.5
        self.Wxh = NumpyValue(rng.uniform(-scale, scale, size=(input_size, hidden_size)))
        self.Whh = NumpyValue(rng.uniform(-scale, scale, size=(hidden_size, hidden_size)))
        self.bh = NumpyValue(np.zeros((1, hidden_size)))
        self.Why = NumpyValue(rng.uniform(-scale, scale, size=(hidden_size, output_size)))
        self.by = NumpyValue(np.zeros((1, output_size)))
        self.hidden_size = hidden_size

    def parameters(self) -> list:
        return [self.Wxh, self.Whh, self.bh, self.Why, self.by]

    def init_hidden(self, batch_size: int = 1) -> NumpyValue:
        """시퀀스 시작 시점의 hidden state. 아직 아무것도 못 본 상태이므로 0으로 초기화."""
        return NumpyValue(np.zeros((batch_size, self.hidden_size)))

    def step(self, x_t: NumpyValue, h_prev: NumpyValue) -> tuple:
        """
        시퀀스 한 시점(time step)을 처리한다.

        Returns:
            (y_t, h_t): 이 시점의 출력과, 다음 시점으로 넘겨줄 새 hidden state.
        """
        h_t = (x_t @ self.Wxh + h_prev @ self.Whh + self.bh).sigmoid()
        y_t = (h_t @ self.Why + self.by).sigmoid()
        return y_t, h_t

    def forward(self, sequence: list) -> list:
        """
        sequence: [x_0, x_1, ..., x_{T-1}] 형태의 리스트. 각 x_t는 (batch, input_size) 배열.

        Returns:
            outputs: [y_0, y_1, ..., y_{T-1}] -- 각 시점의 출력 리스트.
        """
        batch_size = sequence[0].shape[0]
        h = self.init_hidden(batch_size)
        outputs = []
        for x_t in sequence:
            x_val = NumpyValue(x_t)
            y_t, h = self.step(x_val, h)
            outputs.append(y_t)
        return outputs

    def compute_loss(self, outputs: list, targets: list) -> NumpyValue:
        """시퀀스 전체 시점에 대한 평균 제곱 오차(MSE)"""
        total = None
        count = 0
        for y_t, target_t in zip(outputs, targets):
            diff = y_t + (-1 * target_t)
            squared = diff * diff
            total = squared if total is None else total + squared
            count += target_t.shape[0]
        return total.sum() * (1.0 / count)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def train_step(self, sequence: list, targets: list, learning_rate: float) -> float:
        self.zero_grad()
        outputs = self.forward(sequence)
        loss = self.compute_loss(outputs, targets)
        loss.backward()
        for p in self.parameters():
            p.data -= learning_rate * p.grad
        return float(loss.data)