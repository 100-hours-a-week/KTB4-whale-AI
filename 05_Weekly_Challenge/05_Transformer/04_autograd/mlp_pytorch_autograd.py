"""
(After 2) 실제 PyTorch로 MLP 전체 학습 루프를 구현.

3단계(mlp.py)의 forward만 그대로 가져오고, backward는 PyTorch autograd에 위임한다.
compute_loss()라는 별도 메서드도 필요 없다 — loss 계산 자체가 텐서 연산이라
계산 그래프에 자동으로 편입되기 때문이다.
"""

import torch


class MLP:
    def __init__(self, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        self.w1_1 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()
        self.w1_2 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()
        self.b1 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()

        self.w2_1 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()
        self.w2_2 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()
        self.b2 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()

        self.v1 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()
        self.v2 = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()
        self.c = torch.empty(()).uniform_(-1.0, 1.0, generator=generator).requires_grad_()

    def parameters(self):
        return [self.w1_1, self.w1_2, self.b1, self.w2_1, self.w2_2, self.b2, self.v1, self.v2, self.c]

    def forward(self, x1: float, x2: float) -> torch.Tensor:
        """3단계 mlp.py의 forward()와 동일한 수식. backward()는 정의하지 않는다."""
        x1_t, x2_t = torch.tensor(x1), torch.tensor(x2)

        h1 = torch.sigmoid(self.w1_1 * x1_t + self.w1_2 * x2_t + self.b1)
        h2 = torch.sigmoid(self.w2_1 * x1_t + self.w2_2 * x2_t + self.b2)
        o = torch.sigmoid(self.v1 * h1 + self.v2 * h2 + self.c)
        return o

    def train_step(self, x1: float, x2: float, y_true: float, learning_rate: float) -> float:
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()  # zero_grad — 이전 스텝의 gradient 누적 방지

        o = self.forward(x1, x2)
        loss = (o - y_true) ** 2  # loss 계산도 텐서 연산이라 계산 그래프에 자동 편입

        loss.backward()  # 3단계 backward() 60줄 이상을 대체하는 한 줄

        with torch.no_grad():  # update 단계는 계산 그래프 기록이 필요 없으므로 no_grad로 명시
            for p in self.parameters():
                p -= learning_rate * p.grad

        return loss.item()


def train(mlp: MLP, data: list, epochs: int, learning_rate: float):
    for epoch in range(epochs):
        total_squared_error = 0.0
        for x1, x2, y_true in data:
            total_squared_error += mlp.train_step(x1, x2, y_true, learning_rate)
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            mse = total_squared_error / len(data)
            print(f"Epoch [{epoch+1:5d}/{epochs}] | MSE: {mse:.6f}")


XOR_DATA = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]


if __name__ == "__main__":
    print("=== XOR 문제, PyTorch autograd로 학습 ===")
    mlp = MLP(seed=0)
    train(mlp, XOR_DATA, epochs=10000, learning_rate=1.0)

    print("\nXOR 최종 예측:")
    for x1, x2, y_true in XOR_DATA:
        o = mlp.forward(x1, x2)
        print(f"  ({x1}, {x2}) -> 예측={o.item():.4f} (정답={y_true})")