"""
2단계: activation function을 추가한 뉴런.

1단계 SingleNeuron과의 차이:
    1단계: y_pred = w*x + b                (입력 1개, activation 없음)
    2단계: a = sigmoid(w1*x1 + w2*x2 + b)   (입력 2개, sigmoid activation 추가)

입력을 2개로 확장한 이유:
    XOR, AND 같은 논리 게이트(logic gate) 문제는 입력이 2개(x1, x2)이기 때문이다.

수식 정리:
    z = w1*x1 + w2*x2 + b     (activation 적용 전, 1단계의 y_pred에 해당)
    a = sigmoid(z)            (activation 적용 후, 최종 출력)
    L = (a - y_true)^2        (loss)

backward 시 chain rule 확장:
    dL/dw1 = dL/da * da/dz * dz/dw1
    dL/da = 2*(a - y_true)
    da/dz = sigmoid(z)*(1-sigmoid(z)) = a*(1-a)   (activation 미분, 이번 단계에서 새로 추가된 연결고리)
    dz/dw1 = x1
    따라서 dL/dw1 = 2*(a-y_true) * a*(1-a) * x1
    (dw2, db도 동일한 방식, dz/dw2=x2, dz/db=1)
"""

import random
from activation import sigmoid


class NeuronWithActivation:
    def __init__(self, seed: int = 42):
        random_generator = random.Random(seed)
        self.w1 = random_generator.uniform(-1.0, 1.0)
        self.w2 = random_generator.uniform(-1.0, 1.0)
        self.b = random_generator.uniform(-1.0, 1.0)

    def forward(self, x1: float, x2: float) -> tuple[float, float]:
        """
        forward pass.

        Returns:
            (a, z): 최종 출력(a)과 activation 적용 전 값(z).
            z를 함께 반환하는 이유는 backward에서 da/dz = a*(1-a) 계산에
            a만 있으면 되지만, 1단계와의 구조적 대응을 명확히 하기 위해
            z도 함께 넘긴다.
        """
        z = self.w1 * x1 + self.w2 * x2 + self.b
        a = sigmoid(z)
        return a, z

    def compute_loss(self, a: float, y_true: float) -> float:
        """squared error (제곱오차) — 샘플 1개에 대한 값"""
        return (a - y_true) ** 2

    def backward(self, x1: float, x2: float, a: float, y_true: float) -> tuple[float, float, float]:
        """
        chain rule을 activation까지 확장하여 gradient 계산.

        dL/dw1 = dL/da * da/dz * dz/dw1
        dL/dw2 = dL/da * da/dz * dz/dw2
        dL/db  = dL/da * da/dz * dz/db

        Returns:
            (dL_dw1, dL_dw2, dL_db)
        """
        error = a - y_true

        # (w1) dL/dw1 = dL/da * da/dz * dz/dw1
        dL_da = 2 * error
        da_dz = a * (1 - a)  # sigmoid_derivative(z)를 a로부터 바로 계산 (효율적)
        dz_dw1 = x1
        dL_dw1 = dL_da * da_dz * dz_dw1

        # (w2) dL/dw2 = dL/da * da/dz * dz/dw2
        dL_da = 2 * error
        da_dz = a * (1 - a)
        dz_dw2 = x2
        dL_dw2 = dL_da * da_dz * dz_dw2

        # (bias) dL/db  = dL/da * da/dz * dz/db
        dL_da = 2 * error
        da_dz = a * (1 - a)
        dz_db = 1
        dL_db = dL_da * da_dz * dz_db

        return dL_dw1, dL_dw2, dL_db

    def update(self, dL_dw1: float, dL_dw2: float, dL_db: float, learning_rate: float):
        self.w1 -= learning_rate * dL_dw1
        self.w2 -= learning_rate * dL_dw2
        self.b -= learning_rate * dL_db

    def train_step(self, x1: float, x2: float, y_true: float, learning_rate: float) -> float:
        a, z = self.forward(x1, x2)
        squared_error = self.compute_loss(a, y_true)
        dL_dw1, dL_dw2, dL_db = self.backward(x1, x2, a, y_true)
        self.update(dL_dw1, dL_dw2, dL_db, learning_rate)
        return squared_error


def train(neuron: NeuronWithActivation, data: list[tuple[float, float, float]],
          epochs: int, learning_rate: float):
    """data: [(x1, x2, y_true), ...] 형태의 리스트"""
    for epoch in range(epochs):
        total_squared_error = 0.0
        for x1, x2, y_true in data:
            squared_error = neuron.train_step(x1, x2, y_true, learning_rate)
            total_squared_error += squared_error
        if (epoch + 1) % 500 == 0 or epoch == 0:
            mse = total_squared_error / len(data)
            print(f"Epoch [{epoch+1:4d}/{epochs}] | MSE: {mse:.6f} | "
                  f"w1={neuron.w1:.4f} w2={neuron.w2:.4f} b={neuron.b:.4f}")


AND_DATA = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
]

XOR_DATA = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]


if __name__ == "__main__":
    print("=== AND 문제 (linearly separable) ===")
    neuron_and = NeuronWithActivation(seed=42)
    train(neuron_and, AND_DATA, epochs=3000, learning_rate=1.0)

    print("\nAND 최종 예측:")
    for x1, x2, y_true in AND_DATA:
        a, _ = neuron_and.forward(x1, x2)
        print(f"  ({x1}, {x2}) -> 예측={a:.4f} (정답={y_true})")

    print("\n=== XOR 문제 (not linearly separable) ===")
    neuron_xor = NeuronWithActivation(seed=42)
    train(neuron_xor, XOR_DATA, epochs=3000, learning_rate=1.0)

    print("\nXOR 최종 예측:")
    for x1, x2, y_true in XOR_DATA:
        a, _ = neuron_xor.forward(x1, x2)
        print(f"  ({x1}, {x2}) -> 예측={a:.4f} (정답={y_true})")