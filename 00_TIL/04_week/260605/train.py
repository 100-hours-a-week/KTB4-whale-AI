from train_batch import train_batch 
from types import SimpleNamespace

def train(inputs, outputs, epochs, weights, bias, learning_rate, activate_f):
    """
        모델 훈련
        - 모델 훈련
        - 훈련 결과 확인
    """
    # 2. 모델 훈련
    # 2.1. 모델 훈련
    worked_batch = None

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} / {epochs}")
        print(f"{'='*60}")

        worked_batch = train_batch(inputs, outputs, weights, bias, learning_rate, activate_f)

    assert worked_batch is not None # "이 변수는 None이 아니다"라고 알려줌

    # 2.2. 훈련 결과 확인
    print("\n ==== 학습 종료 ====")
    print("학습된 가중치:", worked_batch.weights)
    print("학습된 편향:", worked_batch.bias)

    return SimpleNamespace(
        weights=worked_batch.weights,
        bias=worked_batch.bias
    )