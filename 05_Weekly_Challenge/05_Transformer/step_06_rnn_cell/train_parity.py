"""
RNN cell 학습 실험.

비교 대상:
    1. RNNCell (hidden state로 이전 시점 정보를 전달)
    2. MemorylessBaseline (현재 입력 x_t만 보고 즉시 y_t를 예측, hidden state 없음
       -- 5단계 배치 MLP를 시퀀스 각 시점에 독립적으로 적용한 것과 동일한 구조)

가설: RNNCell은 학습이 되지만, MemorylessBaseline은 원리적으로 학습이 안 된다
(같은 입력 x_t=1이 어떤 때는 정답 1, 어떤 때는 정답 0이라, 입력만으로는 구분 불가능).
"""


import numpy as np
from step_05_vectorization.value_numpy import NumpyValue
from rnn_cell import RNNCell
from parity_task import generate_parity_sequence


class MemorylessBaseline:
    """hidden state 없이, 매 시점 x_t만 보고 즉시 y_t를 예측하는 최소 모델"""

    def __init__(self, input_size: int = 1, output_size: int = 1, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.5
        self.W = NumpyValue(rng.uniform(-scale, scale, size=(input_size, output_size)))
        self.b = NumpyValue(np.zeros((1, output_size)))

    def parameters(self) -> list:
        return [self.W, self.b]

    def forward(self, sequence: list) -> list:
        outputs = []
        for x_t in sequence:
            x_val = NumpyValue(x_t)
            y_t = (x_val @ self.W + self.b).sigmoid()
            outputs.append(y_t)
        return outputs

    def compute_loss(self, outputs: list, targets: list) -> NumpyValue:
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


def generate_training_sequences(num_sequences: int, length: int, base_seed: int) -> list:
    return [generate_parity_sequence(length, seed=base_seed + i) for i in range(num_sequences)]


def evaluate_accuracy(model, sequences: list) -> float:
    correct, total = 0, 0
    for inputs, targets in sequences:
        outputs = model.forward(inputs)
        for y_t, target_t in zip(outputs, targets):
            predicted = 1.0 if y_t.data[0, 0] >= 0.5 else 0.0
            if predicted == target_t[0, 0]:
                correct += 1
            total += 1
    return correct / total


def train_model(model, sequences: list, epochs: int, learning_rate: float):
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in sequences:
            total_loss += model.train_step(inputs, targets, learning_rate)
        if (epoch + 1) % 200 == 0 or epoch == 0:
            acc = evaluate_accuracy(model, sequences)
            avg_loss = total_loss / len(sequences)
            print(f"Epoch [{epoch+1:4d}/{epochs}] | Loss: {avg_loss:.6f} | Accuracy: {acc*100:.1f}%")


if __name__ == "__main__":
    sequences = generate_training_sequences(num_sequences=8, length=6, base_seed=100)

    print("=== RNNCell (hidden state 있음) ===")
    rnn = RNNCell(input_size=1, hidden_size=4, output_size=1, seed=0)
    train_model(rnn, sequences, epochs=1000, learning_rate=0.5)
    print(f"최종 정확도: {evaluate_accuracy(rnn, sequences)*100:.1f}%\n")

    print("=== MemorylessBaseline (hidden state 없음) ===")
    baseline = MemorylessBaseline(input_size=1, output_size=1, seed=0)
    train_model(baseline, sequences, epochs=1000, learning_rate=0.5)
    print(f"최종 정확도: {evaluate_accuracy(baseline, sequences)*100:.1f}%")