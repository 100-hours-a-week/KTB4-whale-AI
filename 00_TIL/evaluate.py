import numpy as np

class Evaluator:
    """
        모델 평가하기
        - 예측 결과 출력
        - Accurancy 계산 및 출력
    """
    def __init__(self, model):
        self.model = model

    def evaluate(self, inputs, outputs):
        correct = 0
        total = len(inputs)
        evaluations = []

        print("\n" + "="*60)
        print("모델 평가 결과")
        print("="*60)
        print(f"{'입력':<12} | {'예측':<6} | {'정답':<6} | {'결과':<6}")
        print("-"*60)

        for i, input_data in enumerate(inputs):
            y = input_data.reshape(-1, 1) # shape 결과: (2,) -> (1, 2)
            output = self.model.forward(y)
            pred_value = 1 if output[0, 0] >= 0.5 else 0 # 평가할 때는 step function
            true_value = outputs[i]

            evaluations.append(pred_value)

            # 예측 결과 출력
            result = "✅ Correct" if pred_value == true_value else "❌ Wrong"
            print(f"{str(input_data):<12} | {pred_value:<6} | {true_value:<6} | {result}")
            if pred_value == true_value: correct += 1


        # Accurancy 계산 및 출력
        accuracy = correct / total
        print("-"*60)
        print(f"정확도 (Accuracy): {correct}/{total} = {accuracy:.2%}")
        print("="*60)

        return evaluations, accuracy