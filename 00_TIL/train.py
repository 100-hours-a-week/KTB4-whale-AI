from types import SimpleNamespace
from train_perceptron import train_perceptron

def train(inputs, outputs, epochs, W1, b1, W2, b2, learning_rate, activate_f):
    """
        모델 훈련
        - 모델 훈련
        - 훈련 결과 확인
    """
    # 2. 모델 훈련
    # 2.1. 모델 훈련
    current_W1, current_b1 = W1.copy(), b1.copy()
    current_W2, current_b2 = W2.copy(), b2.copy()

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} / {epochs}")
        print(f"{'='*60}")

        for index in range(len(inputs)):
            perceptron = train_perceptron(
                index=index,
                inputs=inputs,
                outputs=outputs,
                W1=current_W1,
                b1=current_b1,
                W2=current_W2,
                b2=current_b2,
                learning_rate=learning_rate,
                activate_f=activate_f
            )

            current_W1 = perceptron.W1
            current_b1 = perceptron.b1
            current_W2 = perceptron.W2
            current_b2 = perceptron.b2

    # 2.2. 훈련 결과 확인
    print("\n ==== 학습 종료 ====")
    print("W1:", current_W1)
    print("W2:", current_W2)

    return SimpleNamespace(
        W1=current_W1,
        b1=current_b1,
        W2=current_W2,
        b2=current_b2
    )