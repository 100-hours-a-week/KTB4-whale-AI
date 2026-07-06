"""
sigmoid, sigmoid_derivative 검증.

검증 항목:
    1. sigmoid(0) = 0.5 (정의상 항상 성립하는 특수값)
    2. sigmoid의 출력이 항상 (0, 1) 범위 안에 있는가
    3. sigmoid_derivative가 수치 미분(numerical differentiation)과 일치하는가
       (극한 정의를 h를 아주 작게 근사하여 직접 비교)
"""

from activation import sigmoid, sigmoid_derivative


def numerical_derivative(func, z: float, h: float = 1e-6) -> float:
    """극한 정의를 h를 아주 작은 값으로 근사"""
    return (func(z + h) - func(z - h)) / (2 * h)  # 중심 차분(central difference) - 더 정확한 근사


def test_sigmoid_at_zero_equals_half():
    result = sigmoid(0.0)
    assert abs(result - 0.5) < 1e-9, f"sigmoid(0)은 정확히 0.5여야 함: {result}"
    print(f"PASS: sigmoid(0) = {result}")


def test_sigmoid_output_range():
    """
    수학적으로 sigmoid(z)는 어떤 z에서도 정확히 (0, 1) 열린 구간 안에 있다.
    다만 z가 매우 크면(예: 100) float 64비트 정밀도로는 1.0으로 반올림되는데,
    이는 sigmoid 자체의 결함이 아니라 부동소수점(floating point) 표현의 한계다.
    그래서 극단값(-100, 100)은 [0, 1] 폐구간으로, 중간값은 (0, 1) 열린 구간으로 검증한다.
    """
    extreme_values = [-100, 100]
    for z in extreme_values:
        s = sigmoid(z)
        assert 0 <= s <= 1, f"sigmoid 출력이 [0,1] 범위를 벗어남: sigmoid({z})={s}"

    moderate_values = [-10, -1, 0, 1, 10]
    for z in moderate_values:
        s = sigmoid(z)
        assert 0 < s < 1, f"sigmoid 출력이 (0,1) 범위를 벗어남: sigmoid({z})={s}"

    print("PASS: 중간 범위 입력은 (0,1) 열린 구간, 극단 입력은 float 정밀도 한계로 [0,1] 경계값 포함 확인")


def test_sigmoid_derivative_matches_numerical():
    for z in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        analytical = sigmoid_derivative(z)
        numerical = numerical_derivative(sigmoid, z)
        diff = abs(analytical - numerical)
        assert diff < 1e-4, f"z={z}에서 미분 불일치: analytical={analytical}, numerical={numerical}"
        print(f"PASS: z={z:5.1f} -> analytical={analytical:.6f}, numerical={numerical:.6f}")


if __name__ == "__main__":
    test_sigmoid_at_zero_equals_half()
    test_sigmoid_output_range()
    test_sigmoid_derivative_matches_numerical()
    print("\n모든 테스트 통과.")