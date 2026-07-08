"""
3단계: MLP (Multi-Layer Perceptron, 다층 퍼셉트론)

구조: 입력(x1, x2) -> 은닉층(hidden layer, 뉴런 2개) -> 출력층(output layer, 뉴런 1개)

2단계(NeuronWithActivation)와의 차이:
    2단계: 뉴런 1개가 직접 최종 출력을 만듦 -> 결정 경계(decision boundary) 1개만 생성 가능
    3단계: 은닉층 뉴런 2개가 각자 다른 결정 경계를 학습하고,
           출력층이 이 둘을 조합(combine) -> 복잡한 경계(XOR) 표현 가능

backward의 핵심 차이:
    2단계까지는 chain rule이 "1개 뉴런 안"에서 끝났다.
    3단계는 출력층에서 계산된 gradient가 은닉층까지 전파(propagate)되어야 한다.
    이 전파 과정이 backpropagation(역전파 알고리즘)이 하는 일이다.
"""

import random
from step_02_activation_function.activation import sigmoid


class Neuron:
    """
    임의 개수의 입력을 받는 범용 뉴런(General neuron).
    은닉층, 출력층 모두 이 클래스를 재사용한다 — 입력 개수만 다를 뿐 구조는 동일하기 때문이다.
    """

    def __init__(self, num_inputs: int, seed: int):
        random_generator = random.Random(seed)
        self.weights = [random_generator.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        self.bias = random_generator.uniform(-1.0, 1.0)

    def forward(self, inputs: list[float]) -> tuple[float, float]:
        """
        z = w1*input1 + w2*input2 + ... + bias
        a = sigmoid(z)

        Returns:
            (a, z): activation 적용 후 출력과, 적용 전 값(backward에서 필요).
        """
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        a = sigmoid(z)
        return a, z


class MLP:
    """
    최소 구성 MLP: 은닉층 뉴런 2개 + 출력층 뉴런 1개.
    입력은 2개(x1, x2)로 고정 — AND, XOR 같은 2-입력 논리 게이트 검증을 위함.
    """

    def __init__(self, seed: int = 42):
        # 은닉층 뉴런 2개. 서로 다른 초기값을 갖도록 seed를 다르게 준다.
        # (같은 seed를 쓰면 두 뉴런이 완전히 동일한 값으로 시작해서,
        #  대칭성(symmetry) 때문에 학습 내내 서로 다른 경계를 학습하지 못하는 문제가 생긴다)
        self.hidden1 = Neuron(num_inputs=2, seed=seed)
        self.hidden2 = Neuron(num_inputs=2, seed=seed + 1)
        # 출력층 뉴런 1개. 입력은 은닉층 출력(h1, h2) 2개.
        self.output = Neuron(num_inputs=2, seed=seed + 2)

    def forward(self, x1: float, x2: float) -> tuple[float, dict]:
        """
        forward pass 전체: 입력 -> 은닉층 -> 출력층

        Returns:
            (o, cache): 최종 출력(o)과, backward에서 필요한 중간값들을 담은 cache.
        """
        h1, z_h1 = self.hidden1.forward([x1, x2])
        h2, z_h2 = self.hidden2.forward([x1, x2])
        o, z_o = self.output.forward([h1, h2])

        cache = {
            'x1': x1, 'x2': x2,
            'h1': h1, 'z_h1': z_h1,
            'h2': h2, 'z_h2': z_h2,
            'o': o, 'z_o': z_o,
        }
        return o, cache

    def compute_loss(self, o: float, y_true: float) -> float:
        """squared error (제곱오차) — 샘플 1개에 대한 값"""
        return (o - y_true) ** 2

    def backward(self, cache: dict, y_true: float) -> dict:
        """
        출력층 -> 은닉층 순서로 gradient를 역전파(backpropagate)한다.

        핵심 신규 개념: dL/dh1, dL/dh2 — 출력층에서 은닉층으로 전파되는 gradient.
        2단계까지는 존재하지 않았던, "다음 층으로 gradient를 넘겨주는" 단계다.

        output: dL/dv1 = dL/do * do/dzo * dzo/dv1
                dL/dv2 = dL/do * do/dzo * dzo/dv2
                dL/dc  = dL/do * do/dzo * dzo/dc

        hidden1: dL/dw1_1 = dL/dh1 * dh1/dzh1 * dzh1/dw1_1  (dL/dh1 = dL/do * do/dzo * dzo/dh1)
                 dL/dw1_2 = dL/dh1 * dh1/dzh1 * dzh1/dw1_2
                 dL/db1   = dL/dh1 * dh1/dzh1 * dzh1/db1

        hidden2: dL/dw2_1 = dL/dh2 * dh2/dzh2 * dzh2/dw2_1  (dL/dh2 = dL/do * do/dzo * dzo/dh2)
                 dL/dw2_2 = dL/dh2 * dh2/dzh2 * dzh2/dw2_2
                 dL/db2   = dL/dh2 * dh2/dzh2 * dzh2/db2
        """
        x1, x2 = cache['x1'], cache['x2']
        h1, h2 = cache['h1'], cache['h2']
        o = cache['o']

        error = o - y_true

        # ============ output (v1, v2, c) ============
        # (v1) dL/dv1 = dL/do * do/dzo * dzo/dv1
        dL_do = 2 * error
        do_dzo = o * (1 - o)
        dzo_dv1 = h1
        dL_dv1 = dL_do * do_dzo * dzo_dv1

        # (v2) dL/dv2 = dL/do * do/dzo * dzo/dv2
        dL_do = 2 * error
        do_dzo = o * (1 - o)
        dzo_dv2 = h2
        dL_dv2 = dL_do * do_dzo * dzo_dv2

        # (c) dL/dc = dL/do * do/dzo * dzo/dc
        dL_do = 2 * error
        do_dzo = o * (1 - o)
        dzo_dc = 1
        dL_dc = dL_do * do_dzo * dzo_dc

        # ============ hidden_1 (w1_1, w1_2, b1) ============
        # (공통, h1) dL/dh1 = dL/do * do/dzo * dzo/dh1 (dzo/dh1 == v1 == self.output.weights[0])
        dL_do = 2 * error
        do_dzo = o * (1 - o)
        dzo_dh1 = self.output.weights[0]  # v1

        # (w1_1) dL/dw1_1 = dL/dh1 * dh1/dzh1 * dzh1/dw1_1
        dL_dh1 = dL_do * do_dzo * dzo_dh1
        dh1_dzh1 = h1 * (1 - h1)
        dzh1_dw1_1 = x1
        dL_dw1_1 = dL_dh1 * dh1_dzh1 * dzh1_dw1_1

        # (w1_2) dL/dw1_2 = dL/dh1 * dh1/dzh1 * dzh1/dw1_2
        dL_dh1 = dL_do * do_dzo * dzo_dh1
        dh1_dzh1 = h1 * (1 - h1)
        dzh1_dw1_2 = x2
        dL_dw1_2 = dL_dh1 * dh1_dzh1 * dzh1_dw1_2

        # (b1) dL/db1 = dL/dh1 * dh1/dzh1 * dzh1/db1
        dL_dh1 = dL_do * do_dzo * dzo_dh1
        dh1_dzh1 = h1 * (1 - h1)
        dzh1_db1 = 1
        dL_db1 = dL_dh1 * dh1_dzh1 * dzh1_db1

        # ============ hidden_2 (w2_1, w2_2, b2) ============
        # (공통, h2) dL/dh2 = dL/do * do/dzo * dzo/dh2, (dzo/dh2 == v2 == self.output.weights[1])
        dL_do = 2 * error
        do_dzo = o * (1 - o)
        dzo_dh2 = self.output.weights[1]  # v2

        # (w2_1) dL/dw2_1 = dL/dh2 * dh2/dzh2 * dzh2/dw2_1
        dL_dh2 = dL_do * do_dzo * dzo_dh2
        dh2_dzh2 = h2 * (1 - h2)
        dzh2_dw2_1 = x1
        dL_dw2_1 = dL_dh2 * dh2_dzh2 * dzh2_dw2_1

        # (w2_2) dL/dw2_2 = dL/dh2 * dh2/dzh2 * dzh2/dw2_2
        dL_dh2 = dL_do * do_dzo * dzo_dh2
        dh2_dzh2 = h2 * (1 - h2)
        dzh2_dw2_2 = x2
        dL_dw2_2 = dL_dh2 * dh2_dzh2 * dzh2_dw2_2

        # (b2) dL/db2 = dL/dh2 * dh2/dzh2 * dzh2/db2
        dL_dh2 = dL_do * do_dzo * dzo_dh2
        dh2_dzh2 = h2 * (1 - h2)
        dzh2_db2 = 1
        dL_db2 = dL_dh2 * dh2_dzh2 * dzh2_db2

        return {
            'output': (dL_dv1, dL_dv2, dL_dc),
            'hidden1': (dL_dw1_1, dL_dw1_2, dL_db1),
            'hidden2': (dL_dw2_1, dL_dw2_2, dL_db2),
        }

    def update(self, gradients: dict, learning_rate: float):
        dL_dv1, dL_dv2, dL_dc = gradients['output']
        self.output.weights[0] -= learning_rate * dL_dv1
        self.output.weights[1] -= learning_rate * dL_dv2
        self.output.bias -= learning_rate * dL_dc

        dL_dw1_1, dL_dw1_2, dL_db1 = gradients['hidden1']
        self.hidden1.weights[0] -= learning_rate * dL_dw1_1
        self.hidden1.weights[1] -= learning_rate * dL_dw1_2
        self.hidden1.bias -= learning_rate * dL_db1

        dL_dw2_1, dL_dw2_2, dL_db2 = gradients['hidden2']
        self.hidden2.weights[0] -= learning_rate * dL_dw2_1
        self.hidden2.weights[1] -= learning_rate * dL_dw2_2
        self.hidden2.bias -= learning_rate * dL_db2

    def train_step(self, x1: float, x2: float, y_true: float, learning_rate: float) -> float:
        o, cache = self.forward(x1, x2)
        squared_error = self.compute_loss(o, y_true)
        gradients = self.backward(cache, y_true)
        self.update(gradients, learning_rate)
        return squared_error


def train(mlp: MLP, data: list[tuple[float, float, float]], epochs: int, learning_rate: float):
    """data: [(x1, x2, y_true), ...] 형태의 리스트"""
    for epoch in range(epochs):
        total_squared_error = 0.0
        for x1, x2, y_true in data:
            squared_error = mlp.train_step(x1, x2, y_true, learning_rate)
            total_squared_error += squared_error
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            mse = total_squared_error / len(data)
            print(f"Epoch [{epoch+1:5d}/{epochs}] | MSE: {mse:.6f}")


AND_DATA = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
]

XOR_DATA = [
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
]


if __name__ == "__main__":
    print("=== XOR 문제 (2단계에서 실패했던 문제) ===")
    # 참고: seed=42는 local minimum(지역 최솟값)에 빠져 수렴하지 않는 초기값이었다.
    # seed=0으로 교체 — 이 현상 자체가 결론·한계에서 다룰 내용이다.
    mlp_xor = MLP(seed=0)
    train(mlp_xor, XOR_DATA, epochs=10000, learning_rate=1.0)

    print("\nXOR 최종 예측:")
    for x1, x2, y_true in XOR_DATA:
        o, _ = mlp_xor.forward(x1, x2)
        print(f"  ({x1}, {x2}) -> 예측={o:.4f} (정답={y_true})")