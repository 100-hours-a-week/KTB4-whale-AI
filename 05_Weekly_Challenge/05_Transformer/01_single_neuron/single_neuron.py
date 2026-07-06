import random

class SingleNeuron:
    """
        y = w * x + b

        forward, loss, backward, update를
        명시적으로 분리된 메서드로 구현
    """
    def __init__(self, seed: int = 42):
        random_generator = random.Random(seed)
        self.w = random_generator.uniform(-1.0, 1.0) # w 초기화
        self.b = random_generator.uniform(-1.0, 1.0) # b 초기화

    def forward(self, x: float) -> float:
        return self.w * x + self.b
    
    def compute_loss(self, y_pred: float, y_true: float) -> float:
        """
            SE(Squared Error, 제곱 오차)를 사용해 
            loss 도출 (샘플 1개에 대한 값)
        """
        return (y_pred - y_true) ** 2
    
    def backward(self, x: float, y_pred: float, y_true: float) -> tuple[float, float]:
        """
            수동으로 미분 gradient 계산
        """
        error = y_pred - y_true
        # y_pred = w*x + b
        # L = (y_pred - y_true) ** 2

        # 1. dL_dw
        # dL_dw = dL/dy_pred * dy_pred/dw
        # dy_pred/dw = x
        # dL/dy_pred = 2*(y_pred - y_true)
        # 따라서, dL_dw = 2*(y_pred - y_true)*x
        dL_dw = 2 * error * x

        # 2. dL_db
        # dL_db = dL/dy_pred * dy_pred/db
        # dy_pred/db = 1
        # dL/dy_pred = 2*(y_pred - y_true)
        # 따라서, dL_db = 2*(y_pred - y_true)*1
        dL_db = 2 * error * 1

        return dL_dw, dL_db
    
    def update(self, dL_dw: float, dL_db: float, learning_rate: float):
        """gradient descent(경사 하강법)으로 w, b 갱신"""
        self.w -= learning_rate * dL_dw
        self.b -= learning_rate * dL_db

    def train_step(self, x: float, y_true: float, learning_rate: float) -> float:
        """
            forward -> loss -> backward -> update 4단계 한 번에 수행
        """
        y_pred = self.forward(x)
        squared_error = self.compute_loss(y_pred, y_true)
        dL_dw, dL_db = self.backward(x, y_pred, y_true)
        self.update(dL_dw, dL_db, learning_rate)
        return squared_error

def generate_data(n: int, true_w: float, true_b: float, seed: int = 0) -> list[tuple[float, float]]:
    random_generator = random.Random(seed)
    data = []
    for _ in range(n):
        x = random_generator.uniform(-10, 10)
        y = true_w * x + true_b
        data.append((x, y))
    return data

def train(neuron: SingleNeuron, data: list[tuple[float, float]], epochs: int, learning_rate: float):
    for epoch in range(epochs):
        loss = 0.0
        for x, y_true in data:
            squared_error = neuron.train_step(x, y_true, learning_rate)
            loss += squared_error
        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = loss / len(data)
            print(f"Epoch [{epoch+1:3d}/{epochs}] | Loss: {avg_loss:.6f} | w={neuron.w:.4f}, b={neuron.b:.4f}")

if __name__ == "__main__":
    TRUE_W, TRUE_B = 2.0, 1.0

    print(f"목표: y = {TRUE_W}x + {TRUE_B}\n")

    neuron = SingleNeuron(seed=42)
    print(f"초기값: w={neuron.w:.4f}, b={neuron.b:.4f}\n")

    data = generate_data(n=20, true_w=TRUE_W, true_b=TRUE_B, seed=0)
    train(neuron, data, epochs=100, learning_rate=0.001)

    print(f"\n최종값: w={neuron.w:.4f}, b={neuron.b:.4f}")
    print(f"목표값: w={TRUE_W:.4f}, b={TRUE_B:.4f}")