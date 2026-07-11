import numpy as np
from preprocess import Preprocessor
from train import Trainer
from evaluate import Evaluator

class SimpleMLP:
    def __init__(self, input_size=2, hidden_size=2, output_size=1, learning_rate=0.5):
        # 1. Preprocessor
        self.preprocessor = Preprocessor()

        init = self.preprocessor.initialize_weights(input_size, hidden_size, output_size)

        self.W1 = init['W1']
        self.b1 = init['b1']
        self.W2 = init['W2']
        self.b2 = init['b2']
        self.learning_rate = learning_rate

        # 2. Trainer
        self.trainer = Trainer(self, learning_rate)

        # 3. Evaluator
        self.evaluator = Evaluator(self)

    # 1. Forward
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1 # (1, hidden_size)
        self.a1 = self.preprocessor.activate_function(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2 # (1, output_size)
        self.a2 = self.preprocessor.activate_function(self.z2) # 출력
        return self.a2
    
    # 2. Backward
    def backward(self, X, y):
        # Backward (Chain Rule)
        # 출력층 gradient
        dL_da2 = 2 * (self.a2 - y) / y.shape[0] # 출력층에서 손실(Loss)을 a2(예측값)에 대해 미분 → 오차 크기 계산
        dL_dz2 = dL_da2 * self.preprocessor.derivative_function(self.z2) # dL/da2 × sigmoid 미분
        dW2 = np.dot(self.a1.T, dL_dz2) # 출력층 가중치 W2에 대한 gradient 계산
        db2 = np.sum(dL_dz2, axis=0, keepdims=True) # 출력층 편향 b2에 대한 gradient 계산

        # 은닉층 gradient
        dL_da1 = np.dot(dL_dz2, self.W2.T) # 은닉층으로 오차를 전달 (Backpropagation 핵심)
        dL_dz1 = dL_da1 * self.preprocessor.derivative_function(self.z1) # 은닉층에서 gradient 계산 (Chain Rule)
        dW1 = np.dot(X.T, dL_dz1) # 입력 → 은닉층 가중치 W1에 대한 gradient
        db1 = np.sum(dL_dz1, axis=0, keepdims=True) # 은닉층 편향 b1에 대한 gradient

        # Weight, Bias Update (Optimizer)
        self.W1 -= self.learning_rate * dW1   # W1 업데이트
        self.b1 -= self.learning_rate * db1   # b1 업데이트
        self.W2 -= self.learning_rate * dW2   # W2 업데이트
        self.b2 -= self.learning_rate * db2   # b2 업데이트

    # 위임(Delegation)
    def train(self, X, y, epochs=20000):
        self.trainer.train(X, y, epochs)

    def evaluate(self, X, y):
        return self.evaluator.evaluate(X, y)

# =================== Work Process (작업 과정) ===================
if __name__ == "__main__":
    """
        데이터 정의 (XOR 연산 추론하기)
        - 입력/출력 데이터 입수
        - 가중치와 편향 크기 설정
        - 학습률, 에포크 설정
        - 활성화 함수 설정
    """
    # XOR 데이터
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 입력 데이터 입수
    y = np.array([0, 1, 1, 0]) # 출력 데이터 입수

    # 모델 생성
    model = SimpleMLP()

    # 학습
    print("=== 학습 시작 ===")
    model.train(X, y)

    # 평가
    print("\n=== 예측 결과 ===")
    evaludations, accuracy = model.evaluate(X, y)