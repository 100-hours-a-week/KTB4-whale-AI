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

     - (원인) W1의 shape은 `(2, hidden_size)`이므로, "행렬 곱셈 규칙"에 따라 X는 `(1, 2)`여야 함.
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

- (현상) python 동작 터미널 명령어를 실행했을 때, pyc 파일이 생기는 이유
  1. .pyc 파일이 무엇인가?
     - 해당 파일은 "파이썬 바이트코드" 파일
     - .py 파일 -> 사람이 읽을 수 있는 소스 코드
     - .pyc 파일 -> 파일썬이 이해할 수 있도록 컴파일된 중간 코드
     - (소결) 파이썬은 순수한 인터프리터 언어가 아니라, 컴파일 + 실행 과정을 거치는 하이브리드 형태
  2. `python main.py` 명령어를 실행했을 때 실제로 일어나는 일 (전체 흐름)
     1. 파이썬 파서로 main.py에서 AST 생성
     2. 컴파일러로 AST에서 바이트코드 생성
     3. 파이썬 가상 머신(PVM)으로 실제 실행
  3. AST란 무엇인가?
     - Abstract Syntax Tree의 약자로
     - py 파일을 컴파일러가 이해할 수 있도록 트리 구조로 변환된 결과
  4. .pyc 파일을 만드는 근본적인 이유
     - 파이썬 성능 최적화를 위해
     - py 파일을 실행할 때마다 파싱부터 컴파일 과정을 반복하면 느림
     - 한 번 컴파일한 바이트코드를 .pyc 파일로 저장해두고, 다음 실행할 때 컴파일 과정을 생략하고 바로 실행 가능
     - 특히 import 문으로 다른 모듈을 불러올 때 효과 극대화
  5. `__pycache__` 폴더가 생기는 이유
     - 소스 코드와 컴파일 결과를 분리해서 관리하기 위해 파이썬 3.3 버전부터 자동화
  6. (확장) 의도적으로 Git에 커밋하면 안되는 이유
     - 환경 의존적: pyc 파일은 파이썬 버전, 운영체제에 따라 다르게 생성됨
     - 용량 낭비: 바이트코드 파일이 불필요하게 커밋됨
     - 충돌 위험: 팀원마다 파이썬 버전이 다르면 pyc 파일이 계속 충돌 발생
     - 의미 없음: py 파일만으로 언제든 재생성 가능
