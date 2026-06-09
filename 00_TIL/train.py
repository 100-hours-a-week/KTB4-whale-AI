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
        loss = self.loss_function(output, y)
        # 3. Backward
        self.model.backward(x, y)

        return loss
    
    def loss_function(self, output, y, type = 'mse'):
        if type == 'bce':
            return self._bce(output, y)
        elif type == 'mae':
            return self._mae(output, y)
        else:
            return self._mse(output, y)

    # 회귀 문제일 때, loss 함수
    def _mse(self, output, y):
        return np.mean((output - y) ** 2)
    
    def _mae(self, output, y):
        """Mean Absolute Error (MAE)"""
        return np.mean(np.abs(output - y))

    # 분류 문제일 때, loss 함수
    def _bce(self, output, y):
        """Binary Cross Entropy (BCE)"""
        # 이진 분류용
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
    
    def _hinge(self, output, y):
        """
            Hinge
            - y: 정답 라벨 (-1 or 1)
            - output: 모델의 raw 출력(0, 1 - y * output)
        """
        return np.mean(np.maximum(0, 1 - y * output))
    
    # TODO: 제곱 힌지 로스 함수도 있던데, 무슨 차이인지 확인 필요