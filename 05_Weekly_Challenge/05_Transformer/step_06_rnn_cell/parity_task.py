"""
6단계 검증용 과제: 누적 패리티(running parity).

정의: 시퀀스의 t번째 출력 = "0번째부터 t번째까지 등장한 1의 개수가 홀수인가"
예: 입력 시퀀스 [1, 0, 1, 1] -> 정답 시퀀스 [1, 1, 0, 1]
    (1개: 홀수->1) (여전히 1개: 홀수->1) (2개: 짝수->0) (3개: 홀수->1)

이 과제가 "메모리(hidden state) 없이는 풀 수 없는" 이유:
    t=2 시점에서 입력이 1일 때, 그 이전에 1이 몇 번 나왔는지에 따라
    정답이 다르다 (이전 총 개수가 짝수면 정답 1, 홀수면 정답 0).
    즉 "현재 입력값 하나만" 보고는 정답을 절대 알 수 없다.
"""

import random
import numpy as np


def generate_parity_sequence(length: int, seed: int) -> tuple:
    """
    길이 length짜리 0/1 시퀀스와, 그에 대한 누적 패리티 정답 시퀀스를 생성.

    Returns:
        (inputs, targets): 둘 다 [(1,1) 모양의 배열]의 리스트 (batch_size=1 가정)
    """
    rng = random.Random(seed)
    bits = [rng.randint(0, 1) for _ in range(length)]

    inputs = []
    targets = []
    running_count = 0
    for bit in bits:
        running_count += bit
        parity = running_count % 2
        inputs.append(np.array([[float(bit)]]))
        targets.append(np.array([[float(parity)]]))
    return inputs, targets