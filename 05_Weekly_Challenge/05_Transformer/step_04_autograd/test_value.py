"""
Value 클래스 검증.

검증 항목:
    1. 단순 곱셈(a*b)의 gradient가 손으로 유도한 값과 일치하는가
    2. 여러 연산이 섞인 식(a*b + a*c)에서, a로 미분한 결과가
       손으로 유도한 chain rule 결과와 일치하는가 (a가 두 경로에 동시에 관여하는 경우)
    3. sigmoid의 gradient가 수치 미분(numerical differentiation)과 일치하는가
    4. 뺄셈(a - b)의 gradient가 손으로 유도한 값과 일치하는가
    5. 거듭제곱(a ** n)의 gradient가 power rule(거듭제곱 법칙)과 일치하는가
    6. sigmoid, 뺄셈, 거듭제곱을 조합한 squared error가 1단계에서
       손으로 유도한 공식(dL_dw = 2*error*x)과 일치하는가
"""

import math
from value import Value


def test_simple_multiplication_gradient():
    """
    f = a * b
    df/da = b, df/db = a  (곱셈 미분 공식, 손으로 유도)
    """
    a = Value(3.0)
    b = Value(4.0)
    f = a * b

    f.backward()

    assert f.data == 12.0, f"forward 계산 오류: {f.data}"
    assert a.grad == 4.0, f"df/da 오류: 기대값 4.0(=b), 실제값 {a.grad}"
    assert b.grad == 3.0, f"df/db 오류: 기대값 3.0(=a), 실제값 {b.grad}"
    print(f"PASS: test_simple_multiplication_gradient (f={f.data}, a.grad={a.grad}, b.grad={b.grad})")


def test_shared_variable_across_two_paths():
    """
    f = a*b + a*c   (a가 두 개의 곱셈 경로에 동시에 사용됨)

    손으로 유도:
        df/da = b + c   (두 경로의 gradient가 합산되어야 함 — multivariable chain rule)
        df/db = a
        df/dc = a
    """
    a = Value(2.0)
    b = Value(3.0)
    c = Value(5.0)

    f = a * b + a * c
    f.backward()

    expected_f = 2.0 * 3.0 + 2.0 * 5.0  # 6 + 10 = 16
    expected_da = 3.0 + 5.0             # b + c = 8
    expected_db = 2.0                   # a
    expected_dc = 2.0                   # a

    assert f.data == expected_f, f"forward 오류: {f.data} vs {expected_f}"
    assert a.grad == expected_da, f"df/da 오류: {a.grad} vs {expected_da} (두 경로 합산 확인)"
    assert b.grad == expected_db, f"df/db 오류: {b.grad} vs {expected_db}"
    assert c.grad == expected_dc, f"df/dc 오류: {c.grad} vs {expected_dc}"
    print(f"PASS: test_shared_variable_across_two_paths "
          f"(f={f.data}, a.grad={a.grad}, b.grad={b.grad}, c.grad={c.grad})")


def test_sigmoid_gradient():
    z = Value(0.5)
    a = z.sigmoid()
    a.backward()

    expected_a = 1 / (1 + math.exp(-0.5))
    expected_grad = expected_a * (1 - expected_a)  # sigmoid'(z) = sigmoid(z)*(1-sigmoid(z))

    assert abs(a.data - expected_a) < 1e-9, f"sigmoid 계산 오류: {a.data} vs {expected_a}"
    assert abs(z.grad - expected_grad) < 1e-9, f"sigmoid 미분 오류: {z.grad} vs {expected_grad}"
    print(f"PASS: test_sigmoid_gradient (a={a.data:.6f}, z.grad={z.grad:.6f})")


def test_subtraction_gradient():
    """
    f = a - b
    df/da = 1, df/db = -1
    """
    a = Value(5.0)
    b = Value(3.0)
    f = a - b
    f.backward()

    assert f.data == 2.0, f"뺄셈 계산 오류: {f.data}"
    assert a.grad == 1.0, f"df/da 오류: {a.grad}"
    assert b.grad == -1.0, f"df/db 오류: {b.grad}"
    print(f"PASS: test_subtraction_gradient (f={f.data}, a.grad={a.grad}, b.grad={b.grad})")


def test_power_gradient():
    """
    f = a ** 2
    df/da = 2*a  (power rule)
    """
    a = Value(3.0)
    f = a ** 2
    f.backward()

    assert f.data == 9.0, f"거듭제곱 계산 오류: {f.data}"
    assert a.grad == 6.0, f"df/da 오류: 기대값 6.0(=2*3), 실제값 {a.grad}"
    print(f"PASS: test_power_gradient (f={f.data}, a.grad={a.grad})")


def test_squared_error_matches_manual_derivation():
    """
    지금까지 반복 사용한 squared error 형태를 Value로 구성했을 때,
    1단계에서 손으로 유도한 dL_dw = 2*error*x 공식과 일치하는지 확인.

    L = (w*x + b - y_true)^2
    """
    w = Value(2.0)
    b = Value(1.0)
    x_val, y_true_val = 3.0, 10.0

    y_pred = w * x_val + b
    L = (y_pred - y_true_val) ** 2
    L.backward()

    error = y_pred.data - y_true_val  # 7.0 - 10.0 = -3.0
    expected_dL_dw = 2 * error * x_val  # 1단계에서 유도한 공식
    expected_dL_db = 2 * error

    assert abs(w.grad - expected_dL_dw) < 1e-9, f"dL_dw 불일치: {w.grad} vs {expected_dL_dw}"
    assert abs(b.grad - expected_dL_db) < 1e-9, f"dL_db 불일치: {b.grad} vs {expected_dL_db}"
    print(f"PASS: test_squared_error_matches_manual_derivation "
          f"(w.grad={w.grad} vs 1단계 공식={expected_dL_dw}, "
          f"b.grad={b.grad} vs 1단계 공식={expected_dL_db})")


if __name__ == "__main__":
    test_simple_multiplication_gradient()
    test_shared_variable_across_two_paths()
    test_sigmoid_gradient()
    test_subtraction_gradient()
    test_power_gradient()
    test_squared_error_matches_manual_derivation()
    print("\n모든 테스트 통과.")