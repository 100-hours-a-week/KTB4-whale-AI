from types import SimpleNamespace
from train_perceptron import train_perceptron

def train(inputs, outputs, epochs, weights, bias, learning_rate, activate_f):
    """
        모델 훈련
        - 모델 훈련
        - 훈련 결과 확인
    """
    # 2. 모델 훈련
    # 2.1. 모델 훈련
    current_weights = weights.copy()
    current_bias = bias.copy()

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} / {epochs}")
        print(f"{'='*60}")

        for index in range(len(inputs)):
            perceptron = train_perceptron(
                index=index,
                inputs=inputs,
                outputs=outputs,
                weights=current_weights,
                bias=current_bias,
                learning_rate=learning_rate,
                activate_f=activate_f
            )

            current_weights = perceptron.weights
            current_bias = perceptron.bias

    # 2.2. 훈련 결과 확인
    print("\n ==== 학습 종료 ====")
    print("학습된 가중치:", current_weights)
    print("학습된 편향:", current_bias)

    return SimpleNamespace(
        weights=current_weights,
        bias=current_bias
    )