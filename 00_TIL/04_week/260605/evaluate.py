import numpy as np

def evaluate(inputs, outputs, weights, bias):
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
        pred_value = _predict_single(input_data, weights, bias)
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

def _predict_single(input_data, weights, bias):
    total_input = np.dot(input_data, weights) + bias
    print(total_input)
    return _step_function(total_input)

def _step_function(x):
    return 1 if x >= 0 else 0