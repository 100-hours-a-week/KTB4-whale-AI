"""
1단계(SingleNeuron, y_pred = w*x + b)가 비선형(non-linear) 데이터를
학습하지 못한다는 것을, 선형 데이터와 대조하여 실증적으로 확인한다.

목적:
    2단계(activation function 도입)로 넘어가야 하는 필요성을
    "설명"이 아니라 "실행 결과"로 증명한다.

방법:
    - 선형 데이터: y = 2x + 1 (SingleNeuron의 가설 공간과 정확히 일치)
    - 비선형 데이터: y = x^2   (SingleNeuron의 가설 공간 밖)
    두 데이터에 동일한 학습 절차(같은 epoch, learning_rate)를 적용하고,
    최종 MSE가 어떻게 다른지 비교한다.
"""

import random
from single_neuron import SingleNeuron, train


def generate_linear_data(n: int, true_w: float, true_b: float, seed: int = 0) -> list[tuple[float, float]]:
    """y = true_w*x + true_b (선형 관계, SingleNeuron의 가설 공간과 일치)"""
    random_generator = random.Random(seed)
    data = []
    for _ in range(n):
        x = random_generator.uniform(-10, 10)
        y = true_w * x + true_b
        data.append((x, y))
    return data


def generate_nonlinear_data(n: int, seed: int = 0) -> list[tuple[float, float]]:
    """y = x^2 (비선형 관계, 어떤 w, b를 골라도 직선으로는 정확히 표현 불가능)"""
    random_generator = random.Random(seed)
    data = []
    for _ in range(n):
        x = random_generator.uniform(-10, 10)
        y = x ** 2
        data.append((x, y))
    return data


def compute_mse(neuron: SingleNeuron, data: list[tuple[float, float]]) -> float:
    total_squared_error = 0.0
    for x, y_true in data:
        y_pred = neuron.forward(x)
        total_squared_error += (y_pred - y_true) ** 2
    return total_squared_error / len(data)


def test_linear_data_converges_near_zero_mse():
    """1단계 가설 공간과 일치하는 선형 데이터는 MSE가 거의 0까지 줄어들어야 한다"""
    data = generate_linear_data(n=20, true_w=2.0, true_b=1.0, seed=0)
    neuron = SingleNeuron(seed=42)

    train(neuron, data, epochs=200, learning_rate=0.001)
    final_mse = compute_mse(neuron, data)

    assert final_mse < 0.1, f"선형 데이터인데도 MSE가 충분히 안 줄어듦: {final_mse}"
    print(f"PASS: 선형 데이터(y=2x+1) 최종 MSE={final_mse:.6f} (0에 근접 -> 1단계로 충분히 학습 가능)")
    return final_mse


def test_nonlinear_data_cannot_converge_near_zero():
    """
    1단계 가설 공간 밖의 비선형 데이터(y=x^2)는,
    동일한 epoch을 줘도 MSE가 0 근처로 줄어들지 않아야 한다.
    (직선 하나로는 포물선을 정확히 근사할 수 없으므로, 남는 오차가 구조적으로 존재)
    """
    data = generate_nonlinear_data(n=20, seed=0)
    neuron = SingleNeuron(seed=42)

    train(neuron, data, epochs=200, learning_rate=0.0001)  # x^2 스케일이 커서 lr을 낮춤
    final_mse = compute_mse(neuron, data)

    # 선형 데이터의 기준(0.1)과 비교해, 비선형 데이터는 훨씬 큰 MSE에서 머물러야 한다
    assert final_mse > 10.0, (
        f"비선형 데이터인데도 MSE가 너무 작음 (1단계로 풀렸다는 뜻이라 모순): {final_mse}"
    )
    print(f"PASS: 비선형 데이터(y=x^2) 최종 MSE={final_mse:.4f} "
          f"(0 근처로 줄지 않음 -> SingleNeuron으로는 구조적으로 학습 불가능)")
    return final_mse


def test_increasing_epochs_does_not_rescue_nonlinear_case():
    """
    비선형 데이터에서 epoch을 10배로 늘려도(200 -> 2000),
    MSE가 유의미하게 더 줄어들지 않아야 한다.
    (선형 데이터라면 epoch을 늘리면 MSE가 계속 줄어드는 것과 대조)
    """
    data = generate_nonlinear_data(n=20, seed=0)

    neuron_200 = SingleNeuron(seed=42)
    train(neuron_200, data, epochs=200, learning_rate=0.0001)
    mse_200 = compute_mse(neuron_200, data)

    neuron_2000 = SingleNeuron(seed=42)
    train(neuron_2000, data, epochs=2000, learning_rate=0.0001)
    mse_2000 = compute_mse(neuron_2000, data)

    # epoch을 10배 늘렸어도 MSE 감소폭이 크지 않아야 함 (이미 수렴 가능한 한계에 도달)
    improvement_ratio = (mse_200 - mse_2000) / mse_200
    print(f"epoch=200일 때 MSE={mse_200:.4f}, epoch=2000일 때 MSE={mse_2000:.4f}, "
          f"개선율={improvement_ratio*100:.2f}%")

    assert improvement_ratio < 0.3, (
        f"epoch을 늘렸더니 MSE가 크게 개선됨 (구조적 한계가 아니라 단순 미수렴일 가능성): "
        f"개선율={improvement_ratio*100:.2f}%"
    )
    print("PASS: epoch을 10배 늘려도 MSE가 크게 개선되지 않음 "
          "-> 학습 부족이 아니라 모델 구조(직선) 자체의 한계")


if __name__ == "__main__":
    print("=== 1. 선형 데이터 (기준선) ===")
    linear_mse = test_linear_data_converges_near_zero_mse()

    print("\n=== 2. 비선형 데이터 (y=x^2) ===")
    nonlinear_mse = test_nonlinear_data_cannot_converge_near_zero()

    print("\n=== 3. epoch을 늘려도 비선형 데이터는 개선되지 않음 ===")
    test_increasing_epochs_does_not_rescue_nonlinear_case()

    print(f"\n요약: 선형 데이터 MSE={linear_mse:.6f} vs 비선형 데이터 MSE={nonlinear_mse:.4f}")
    print("모든 테스트 통과: SingleNeuron(1차식)은 비선형 데이터를 구조적으로 학습할 수 없음을 확인.")