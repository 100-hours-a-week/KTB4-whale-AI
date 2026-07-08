"""
5단계: 배치(batch) 연산 기반 MLP.

3~4단계와의 핵심 차이:
    3~4단계: for x1, x2, y_true in data: ... 로 4개 샘플을 하나씩 순차 처리
             각 은닉 뉴런도 zip으로 짝지어 곱하고 더하는 반복문
    5단계:   4개 샘플 전체를 (4, 2) 행렬 하나로 묶고,
             은닉층 가중치도 (2, 2) 행렬 하나로 묶어서,
             X @ W1 + b1 행렬곱 한 번으로 4개 샘플의 은닉층 출력을 동시에 계산

행렬 모양(shape) 정리:
    X:  (4, 2)  -- 배치 4개, 각 샘플은 입력 2개(x1, x2)
    W1: (2, 2)  -- 입력 2개 -> 은닉 뉴런 2개
    b1: (1, 2)  -- 은닉 뉴런 2개의 bias (브로드캐스팅으로 4개 샘플에 공통 적용)
    H:  (4, 2)  -- 배치 4개, 각 샘플의 은닉층 출력 2개
    W2: (2, 1)  -- 은닉 뉴런 2개 -> 출력 1개
    b2: (1, 1)  -- 출력의 bias
    O:  (4, 1)  -- 배치 4개, 각 샘플의 최종 출력
"""

import numpy as np
from step_05_vectorization.value_numpy import NumpyValue


class BatchMLP:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = NumpyValue(rng.uniform(-1, 1, size=(2, 2)))
        self.b1 = NumpyValue(np.zeros((1, 2)))
        self.W2 = NumpyValue(rng.uniform(-1, 1, size=(2, 1)))
        self.b2 = NumpyValue(np.zeros((1, 1)))

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray) -> NumpyValue:
        """
        X: (batch_size, 2) 형태의 NumPy 배열.
        3~4단계처럼 샘플마다 forward()를 호출하는 게 아니라,
        배치 전체를 한 번의 호출로 처리한다.
        """
        X_val = NumpyValue(X)
        Z1 = X_val @ self.W1 + self.b1     # (batch, 2) @ (2, 2) + (1, 2) -> (batch, 2)
        H = Z1.sigmoid()                    # (batch, 2)
        Z2 = H @ self.W2 + self.b2          # (batch, 2) @ (2, 1) + (1, 1) -> (batch, 1)
        O = Z2.sigmoid()                    # (batch, 1)
        return O

    def compute_loss(self, O: NumpyValue, y_true: np.ndarray) -> NumpyValue:
        """배치 전체의 MSE(Mean Squared Error). 3~4단계처럼 개별 squared_error를 나중에
        합산하는 게 아니라, 배치 전체 차이를 한 번에 계산하고 평균 낸다."""
        diff = O + (-1 * y_true)
        squared = diff * diff
        batch_size = y_true.shape[0]
        return squared.sum() * (1.0 / batch_size)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def train_step(self, X: np.ndarray, y_true: np.ndarray, learning_rate: float) -> float:
        self.zero_grad()
        O = self.forward(X)
        loss = self.compute_loss(O, y_true)
        loss.backward()
        for p in self.parameters():
            p.data -= learning_rate * p.grad
        return float(loss.data)


def train(mlp: BatchMLP, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float):
    for epoch in range(epochs):
        mse = mlp.train_step(X, y, learning_rate)
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:5d}/{epochs}] | MSE: {mse:.6f}")


XOR_X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
XOR_Y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])


if __name__ == "__main__":
    print("=== XOR 문제, 배치 연산(BatchMLP)으로 학습 ===")
    mlp = BatchMLP(seed=0)
    train(mlp, XOR_X, XOR_Y, epochs=10000, learning_rate=1.0)

    print("\nXOR 최종 예측:")
    O = mlp.forward(XOR_X)
    for i in range(4):
        print(f"  {XOR_X[i]} -> 예측={O.data[i, 0]:.4f} (정답={XOR_Y[i, 0]})")