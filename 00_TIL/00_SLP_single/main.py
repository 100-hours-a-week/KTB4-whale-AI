# AND 연산
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 정의
# 1.1. 입력/출력 데이터 정의
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])

# 1.2. 가중치와 편향 초기화
# (NumPy의 난수 생성 함수 사용, for Symmetry Breaking)
weights = np.random.rand(2) # [0, 1) 사이 균등 분포(Uniform Distribution)에서 난수 선택
bias = np.random.rand(1)

# 1.3. Set Learning Rate, Epoch
learning_rate = 0.1 # 각 학습 단계에서 오차(에러)로부터 변경할 가중치와 편향의 크기를 결정하는 하이퍼파라미터
epochs = 5 # 전체 데이터셋을 한 번 학습하는 단위

# 1.4. 활성화 함수 정의
"""
    스텝 함수 외 다양한 활성화 함수 존재
    sigmoid, tanh, ReLU, Leaky ReLU, GeLU, SiLU 등
"""
def step_function(x):
    return 1 if x >= 0 else 0

# 1.4.1. 활성화 함수
"""
    인공신경망에서 뉴런의 출력을 결정하는 "비선형" 함수
"""
def activate_function(x):
    return step_function(x)

# 2. 모델 훈련
# 2.1. 모델 훈련
for epoch in range(epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1} / {epochs}")
    print(f"{'='*60}")

    for i in range(len(inputs)):
        # 1. Forward
        total_input = np.dot(inputs[i], weights) + bias # 총 입력(선형 함수) 계산
        prediction = activate_function(total_input) # 예측값 계산

        # Loss Signal
        loss = outputs[i] - prediction # 손실(실제값 - 예측값) 계산
        # loss > 0: 예측이 부족, weight, bias 증가
        # loss < 0: 예측이 과함, weight, bias 감소
        print(f"\n[Sample {i+1}] 입력: {inputs[i]} | 정답: {outputs[i]}")
        print(f"  현재 weights = {weights.round(4)}, bias = {bias.round(4)}")
        print(f"  total_input = {total_input.round(4)}")
        print(f"  prediction  = {prediction}")
        print(f"  loss        = {loss:.4f}")

        # Weight(with Bias) Update (Backward + Optimizer)
        # Optimizer: 고정된 learning_rate를 사용한 확률적 경사하강법(SGD) 스타일
        weights += learning_rate * loss * inputs[i]
        bias += learning_rate * loss
        print(f"→ weights += {learning_rate} * {loss} * {inputs[i]}")
        print(f"→ bias += {learning_rate} * {loss}")
        print(f"→ 업데이트 후 weights = {weights.round(4)}, bias = {bias.round(4)}")

# 2.2. 훈련 결과 확인
print("\n ==== 학습 종료 ====")
print("학습된 가중치:", weights)
print("학습된 편향:", bias)

# 3. 모델 테스트
# 3.1. 예측 데이터 정의 (AND 연산은 학습 데이터와 동일하므로 inputs 데이터 사용)

# 3.2. 예측 함수 정의
def predict(input_data):
    total_input = np.dot(input_data, weights) + bias
    print(total_input)
    return step_function(total_input)

# 3.3. 예측 결과 확인
print("\n ==== 예측 결과 ====")
for input_data in inputs:
    print(f"입력: {input_data}, 예측 출력: {predict(input_data)}")