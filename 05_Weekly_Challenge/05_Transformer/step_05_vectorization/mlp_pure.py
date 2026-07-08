"""
5단계 순수 Python 버전 배치 MLP.

batch_mlp.py(NumPy 버전)와 구조는 완전히 동일하다 -- 차이는 TensorValue 대신
MatrixValue를 쓴다는 것, 그리고 가중치 초기화를 numpy.random 대신
표준 라이브러리 random으로 한다는 것뿐이다.
"""

import random
from step_05_vectorization.value_pure import PureValue


class BatchMLPPure:
    def __init__(self, seed: int = 42):
        rng = random.Random(seed)
        self.W1 = PureValue([[rng.uniform(-1, 1) for _ in range(2)] for _ in range(2)])
        self.b1 = PureValue([[0.0, 0.0]])
        self.W2 = PureValue([[rng.uniform(-1, 1)] for _ in range(2)])
        self.b2 = PureValue([[0.0]])

    def parameters(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: list) -> PureValue:
        X_val = PureValue([row[:] for row in X])
        Z1 = X_val @ self.W1 + self.b1
        H = Z1.sigmoid()
        Z2 = H @ self.W2 + self.b2
        O = Z2.sigmoid()
        return O

    def compute_loss(self, O: PureValue, y_true: list) -> PureValue:
        y_neg = PureValue([[-v for v in row] for row in y_true])
        diff = O + y_neg
        squared = diff * diff
        batch_size = len(y_true)
        return squared.sum() * (1.0 / batch_size)

    def zero_grad(self):
        for p in self.parameters():
            rows, cols = p.shape
            p.grad = [[0.0] * cols for _ in range(rows)]

    def train_step(self, X: list, y_true: list, learning_rate: float) -> float:
        self.zero_grad()
        O = self.forward(X)
        loss = self.compute_loss(O, y_true)
        loss.backward()
        for p in self.parameters():
            rows, cols = p.shape
            for i in range(rows):
                for j in range(cols):
                    p.data[i][j] -= learning_rate * p.grad[i][j]
        return loss.data[0][0]


def train(mlp: BatchMLPPure, X: list, y: list, epochs: int, learning_rate: float):
    for epoch in range(epochs):
        mse = mlp.train_step(X, y, learning_rate)
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:5d}/{epochs}] | MSE: {mse:.6f}")


XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [[0.0], [1.0], [1.0], [0.0]]


if __name__ == "__main__":
    print("=== XOR 문제, 순수 Python 배치 연산(BatchMLPPure)으로 학습 ===")
    mlp = BatchMLPPure(seed=0)
    train(mlp, XOR_X, XOR_Y, epochs=10000, learning_rate=1.0)

    print("\nXOR 최종 예측:")
    O = mlp.forward(XOR_X)
    for i in range(4):
        print(f"  {XOR_X[i]} -> 예측={O.data[i][0]:.4f} (정답={XOR_Y[i][0]})")