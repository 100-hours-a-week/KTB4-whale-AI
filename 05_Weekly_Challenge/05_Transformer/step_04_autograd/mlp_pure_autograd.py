"""
3단계 MLP를 Value(자동 미분 엔진) 기반으로 재구현.

3단계와의 핵심 차이:
    3단계: forward()와 backward()를 각각 따로 구현 (backward는 손으로 유도한 chain rule)
    4단계: forward()만 구현하면, backward()는 autograd(자동 미분)가 전부 대신한다.
           MLP 클래스 자체에는 backward() 메서드가 존재하지 않는다.
"""

import random
from value import Value


class Neuron:
    def __init__(self, num_inputs: int, seed: int):
        random_generator = random.Random(seed)
        self.weights = [Value(random_generator.uniform(-1.0, 1.0)) for _ in range(num_inputs)]
        self.bias = Value(random_generator.uniform(-1.0, 1.0))

    def forward(self, inputs: list) -> Value:
        z = sum((w * x for w, x in zip(self.weights, inputs)), self.bias)
        a = z.sigmoid()
        return a

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]


class MLP:
    def __init__(self, seed: int = 42):
        self.hidden1 = Neuron(num_inputs=2, seed=seed)
        self.hidden2 = Neuron(num_inputs=2, seed=seed + 1)
        self.output = Neuron(num_inputs=2, seed=seed + 2)

    def forward(self, x1: float, x2: float) -> Value:
        """
        forward pass만 정의한다. backward()는 없다 —
        Value의 연산자 오버로딩이 이 계산 과정을 자동으로 계산 그래프에 기록하고,
        o.backward() 호출 시 이 그래프를 역순 순회하며 gradient를 계산한다.
        """
        h1 = self.hidden1.forward([x1, x2])
        h2 = self.hidden2.forward([x1, x2])
        o = self.output.forward([h1, h2])
        return o

    def parameters(self) -> list[Value]:
        return self.hidden1.parameters() + self.hidden2.parameters() + self.output.parameters()

    def zero_grad(self):
        """
        Value.grad는 +=로 누적되므로, 매 학습 스텝 시작 전에 0으로 초기화해야
        이전 스텝의 gradient가 섞이지 않는다. (1단계부터 강조해온 zero_grad의 필요성과 동일)
        """
        for p in self.parameters():
            p.grad = 0.0

    def train_step(self, x1: float, x2: float, y_true: float, learning_rate: float) -> float:
        self.zero_grad()

        o = self.forward(x1, x2)
        loss = (o - y_true) ** 2  # squared error — compute_loss()를 별도로 안 만들어도 Value 연산만으로 충분

        loss.backward()  # 이 한 줄이 3단계의 backward() 메서드 전체(60줄 이상)를 대체한다

        for p in self.parameters():
            p.data -= learning_rate * p.grad

        return loss.data


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
    print("=== XOR 문제, Value 기반 autograd로 학습 ===")
    mlp = MLP(seed=42)
    train(mlp, XOR_DATA, epochs=10000, learning_rate=1.0)

    print("\nXOR 최종 예측:")
    for x1, x2, y_true in XOR_DATA:
        o = mlp.forward(x1, x2)
        print(f"  ({x1}, {x2}) -> 예측={o.data:.4f} (정답={y_true})")