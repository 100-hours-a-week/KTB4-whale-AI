import numpy as np
from preprocess import _sigmoid_function

def evaluate(inputs, outputs, W1, b1, W2, b2):
    """
        모델 평가하기
        - 예측 결과 출력
        - Accurancy 계산 및 출력
    """
    correct = 0
    total = len(inputs)
    predictions = []

    print("\n" + "="*60)
    print("모델 평가 결과")
    print("="*60)
    print(f"{'입력':<12} | {'예측':<6} | {'정답':<6} | {'결과':<6}")
    print("-"*60)

    for i, input_data in enumerate(inputs):
        pred_value = _predict_single(input_data, W1, b1, W2, b2)
        true_value = outputs[i]
        is_correct = (pred_value == true_value)

        if is_correct: correct += 1

        predictions.append(pred_value)

        # 예측 결과 출력
        result = "✅ Correct" if is_correct else "❌ Wrong"
        print(f"{str(input_data):<12} | {pred_value:<6} | {true_value:<6} | {result}")


    # Accurancy 계산 및 출력
    accuracy = correct / total
    print("-"*60)
    print(f"정확도 (Accuracy): {correct}/{total} = {accuracy:.2%}")
    print("="*60)

def _predict_single(input_data, W1, b1, W2, b2):
    """단일 입력에 대한 MLP 순전파"""
    x = input_data.reshape(1, -1)           # (1, 2)

    # Forward
    z1 = np.dot(x, W1) + b1
    a1 = _sigmoid_function(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = _sigmoid_function(z2)

    # Sigmoid 출력 → 0.5 기준으로 이진 분류
    return 1 if a2[0, 0] >= 0.5 else 0