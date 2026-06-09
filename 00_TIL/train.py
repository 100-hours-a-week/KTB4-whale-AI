import numpy as np

class Trainer:
    """
        모델 훈련
        - 모델 훈련
        - 훈련 결과 확인
    """
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def train(self, inputs, outputs, epochs):
        PRINT_INTERVAL = epochs // 10

        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(inputs)):
                perceptron_loss = self._train_perceptron(i, inputs, outputs)
                total_loss += perceptron_loss

            if (epoch + 1) % PRINT_INTERVAL == 0:
                print(f"Epoch {epoch+1:6d} | Loss: {total_loss / len(inputs):.6f}")

    def _train_perceptron(self, index, inputs, outputs):
        x = inputs[index:index+1]
        y = outputs[index:index+1].reshape(1, 1)

        # 1. Forward
        output = self.model.forward(x)
        # 2. Loss (MSE)
        loss = np.mean((output - y) ** 2) # 손실(실제값 - 예측값) 계산
        # 3. Backward
        self.model.backward(x, y)

        return loss