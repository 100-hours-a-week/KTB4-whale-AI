# 260609 회고록

- (현상) `ValueError: shapes (2,1) and (2,2) not aligned: 1 (dim 1) != 2 (dim 0)`

1. forward는 왜 (1, 2) 형태를 기대하는가?
   - SimpleMLP.forward 내부 계산

     ```python
     def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1 # (1, hidden_size)
        self.a1 = self.preprocessor.activate_function(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2 # (1, output_size)
        self.a2 = self.preprocessor.activate_function(self.z2) # 출력
        return self.a2
     ```

   - W1의 shape은 `(2, hidden_size)`이므로, "행렬 곱셈 규칙"에 따라 X는 `(1, 2)`여야 함.
     - W1의 shape은 preprocessor.initialize_weights 메서드에서 확인 가능

2. reshape의 역할과 내부 파라미터의 의미
   - reshape는 배열의 형태를 바꾸는 함수
   - 예시

     ```python
     input = np.array([0, 0]) # shape
     input.reshape(1, -1)
     input.reshape(-1, 1)
     ```

     - `-1`는 나머지 차원을 자동으로 계산함을 의미

3. 행렬(Matrix)란?
   - 선형대수에서 행렬은 숫자들의 직사각형 배열
     - 0차원 배열: 스칼라(Scaler)
     - 1차원 배열: 벡터(Vector)
     - 2차원 배열: 행렬(Matrix)

4. 차원(Dimension)이란?
   - 배열이 몇 개의 축(방향)을 갖고 있는지를 의미
   - 예시
     |예시|차원|`shape(data)` 결과|설명|
     |-|-|-|-|
     |5|0차원|()|스칼라|
     |[5, 6]|1차원|(2,)|벡터|
     |[[5, 6]]|2차원|(1,2)|행렬(1행 2열)|
     |[[5], [6]]|2차원|(2, 1)|행렬(2행 1열)|
