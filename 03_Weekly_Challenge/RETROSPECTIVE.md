## NumPy

### 데이터 타입

#### (NumPy의 설계 의도) dtype 지정 시 주의사항

- `np.uint8`은 0~255까지만 저장할 수 있는 타입(1바이트)

1. (방법 1) 생성 시점저에 값을 넣을 때
   - NumPy는 이 값이 정말 uint8에 들어갈 수 있는지 엄격히 검사
   - 300은 uint8 범위를 벗어나기 떄문에 OverflowError로 데이터 손실을 미리 막아줌

2. (방법 2) 나중에 변환할 떄
   - 기본 동작이 `casting='unsafe'` (안전하지 않은 허용)
   - 범위를 벗어나는 값은 자동으로 wrap around(Modulo 256) 처리
   - 300 % 256 = 44가 된다.

### 인덱스

#### 요소들의 인덱스 출력하기

- `np.where(조건)[0]`
  - `np.where(조건)`는 항상 튜플로 인덱스 조회
  - `[0]`: 인덱스만 사용하고 싶을 때, where에서 실제 배열 꺼냄.

### 연산

#### 브로드캐스팅 규칙

- 배열의 모양을 마지막 축부터 맞춘다.
- 각 축에서 두 크기가 같거나 한쪽이 1이면 연산 가능하다.
- 모자란 축은 길이 1로 있다고 간주해 값이 복제된다.
  - 이 때 결과 shape은 각 축별로 더 큰 값을 선택한다.
- 예시
  - `(2, 3) + (1, 3) -> (2, 3)`
  - `(2, 1) + (1, 3) -> (2, 3)`
    - 행이 맞지 않지만 1이라 확장 가능
  - `(2, 3) + Scalar -> (2, 3)`
    - 스칼라는 모든 위치로 브로드캐스트 가능
  - `(2, 3) + (3, 2) -> X`
    - 마지막 축도 다르고
    - 서로 1도 아니고
    - 같지도 않음
    - 오류 반환

## Pandas

### 데이터프레임

#### 하드코딩에서 확장성있는 코드 수정

```python
# 3번 퀘스트
# 각 학생의 총점을 계산하는 새로운 열을 추가한 후
# 총점이 250점 이상인 학생들만 포함된 데이터프레임을 생성하기
data = {'이름': ['홍길동', '김철수', '박영희', '이순신'],
        '국어': [85, 90, 88, 92],
        '영어': [78, 85, 89, 87],
        '수학': [92, 88, 84, 90]}
df = pd.DataFrame(data)

# 방법 1. 하드코딩
# df['총점'] = df['국어'] + df['영어'] + df['수학']

# 방법 2. 이름 열을 제외한 컬럼 선택
# score_columns = df.columns[df.columns != '이름']
# df['총점'] = df[score_columns].sum(axis=1)

# 방법 3. 숫자형 열만 있는 컬럼 선택
df['총점'] = df.select_dtypes(include='number').sum(axis=1)

high_total = df[df['총점'] >= 250]
print(high_total)
```

- 확장성 있는 총점 데이터 추가 방식 생각
- 방법 3이 베스트
  - 방법 1은 하드코딩 되어있어, 새로운 과목이 들어올 때 대응하기 어려움
  - 방법 2는 과목만 대응 가능, 새로운 컬럼(ID, 반, 평균 등)이 들어올 때 대응하기 어려움
  - 방법 3는 성적만이 숫자로 들어온다는 가정 (한계: 과목 외 숫자값이 들어올 경우 대응하기 어려움)

#### groupby에서 agg 함수 사용 방법

```python
grouped_1 = df.groupby('부서')
aggregated = grouped_1.agg('sum')
```

```python
grouped_1 = df.groupby('부서')
aggregated = grouped_1['급여'].agg('sum')
```

```python
grouped_1 = df.groupby('부서')
aggregated = grouped_1['급여'].agg(lambda x: sum(x) + 2)
```

```python
grouped_1 = df.groupby('부서')
aggregated = grouped_1.agg({
  '근속연수': 'max',
  '급여': ['sum', 'mean', 'max', 'min']
})
```

```python
import pandas as pd
grouped_1 = df.groupby('부서')
aggregated = grouped_1.agg(
  급여_mean=pd.NamedAgg(column="급여", aggfunc="mean")
  근속연수_max=pd.NamedAgg(column="근속연수", aggfunc="max")
)
```

#### lambda 사용 시 타입 에러 발생

```python
# DataFrame 생성 및 filtering 기능 활용
# =================== DataFrame 생성 ===================
df = pd.DataFrame(data)
print("✅ 회사 직원 정보 DataFrame 생성")
print(df)

# =================== filtering 기능 활용 ===================
# 1. 그룹화 후 필터링
grouped_1 = df.groupby('부서') # 분할: 부서 기준으로 그룹 분할
filtered = grouped_1.filter(lambda x: x['급여'].mean() > 5000) # 적용 및 결합 # type: ignore
print("\n✅ [1. 그룹화 후 필터링] groupby로 부서별 급여 평균 중 5000 초과 직원 산출")
print(filtered)
```

- 에러: `ParamSpec "P@filter"에 대한 인수가 없습니다.`
- `# type: ignore`를 추가하여 타입 에러 일시적 해결
- 근본적인 원인 해결 필요
- AI 왈: pandas + Pylance 조합의 현재 한계
  - 자주 발생하는 타입 체커 버그
  - 💡 실무 규칙:
    - `groupby.filter()`는 편리하지만 타입 이슈가 자주 발생
    - 복잡한 그룹 필터링은 `transform()` + mask 조합을 기본으로 사용

## Matplotlib

### (Seaborn) 금융데이터 3번 문제

```python
# 3번 퀘스트
# df 데이터프레임에서
# 주간(7일) 단위로 종가(Close) 평균을 리샘플링한 후,
# 이를 바탕으로 주간 변동성(표준편차)을 계산하기
date_rng = pd.date_range(start='2024-01-01', periods=30, freq='D')
close_prices = np.random.uniform(100, 200, size=len(date_rng))  # 100~200 사이의 랜덤 종가 생성

df = pd.DataFrame({
    "datetime": date_rng,
    "close": close_prices
})

# 1. datetime를 인덱스로 설정
df.set_index('datetime', inplace=True)

# (문제) ❌ rolling(window=7).std()로 표준편차를 구하면 안되는 이유: 인덱스 미스 조정 문제 발생
# # 2. 1일(D) -> 7일(W)로 다운샘플링
# df_week = df.resample('W').mean()
# print(df_week.head().reset_index())

# # 3. 주간 변동성(표준편차) 계산
# df_week['volatility'] = df['close'].rolling(window=7).std()
# print(df_week.head(10).reset_index())

# (AI 추천 방식) 💡 다운샘플링 과정에서 같이 결과를 가져오므로, 인덱스 미스 조정 문제가 발생하지 않음.
df_week = df.resample('W').agg({
    'close': ['mean', 'std']
})
df_week.columns = ['close_mean', 'volatility']
print(df_week)
```

1. `df_week = df.resample('W').mean()`
   - df_week는 7일로 다운샘플링 되었으므로, 인덱스는 주간 날짜 5개로 만들어짐.
   - 예시: 2024-01-07, 2024-01-14, 2024-01-21, 2024-01-28, 2024-02-04
     - 2024-01-01 ~ 2024-01-07 -> 2024-01-07로 다운샘플링
     - ... (계속 7일 진행)
     - 2024-01-29 ~ 2024-01-30 -> 2024-02-04로 다운샘플링
     - (소결) 마지막 주는 2일 밖에 없지만, 계산 결과를 7일 뒤 날짜로 주간 날짜를 인덱스로 하여 값을 적용

2. `df['close'].rolling(window=7).std()`
   - 해당 과정은 df 인덱스(일간 날짜) 기준으로 단순 이동 평균(rolling) 진행된다.
   - 따라서, 결과물도 일간 날짜 30개로 만들어진다.

3. `df_week['volatility']에 할당`
   - df_week에 일간 날짜 30개를 할당하려 할 때, pandas는 두 인덱스를 비교해서 인덱스가 같으면 그 값을 가져온다.
   - 현재 인덱스가 주간 날짜, 일간 날짜이므로, 같은 날짜일 때 그 값을 가져온다.
   - df_week의 마지막 주는 단순 이동 평균 데이터프레임에서 생길 수 없으므로, NaN이 할당된다.

4. 결론
   - resample('W').mean()에서는 인덱스(마지막 주)가 2024-02-04이고, 값이 생기지만
   - rolling(단순 이동 평균)에서는 인덱스가 2024-01-30이므로, df_week에 할당할 때 값이 NaN이 할당된다.
   - 따라서 사용하면 안 된다.

### (SciPy) 통계 계산한 분석 결과를 활자로 어떻게 표현할 것인지 정리

```python
# 3번 퀘스트
# 분산 분석(ANOVA, Analysis of Variance)을 수행하여
# 여러 그룹의 평균이 서로 다른지 검정하기
np.random.seed(42)
group_1 = np.random.normal(loc=50, scale=10, size=30)  # 평균 50, 표준편차 10
group_2 = np.random.normal(loc=55, scale=10, size=30)  # 평균 55, 표준편차 10
group_3 = np.random.normal(loc=60, scale=10, size=30)  # 평균 60, 표준편차 10

f_stat, p_value = stats.f_oneway(group_1, group_2, group_3)
print(f"f_stat: {f_stat}") # f_stat는 약 12.2095
print(f"p_value: {p_value}") # p_value는 약 0.0000212(0.002%)
alpha = 0.05
if p_value < alpha:
    print("유의미한 차이가 있습니다. (p = {:.7f})".format(p_value))
else:
    print("유의미한 차이가 없습니다.")
```

1. (f_stat) 그룹 내외 분산 크기 파악 비교하기
   - f_stat는 그룹 간 차이가 그룹 내 변동보다 어느 정도 큰지를 의미
   - f_stat가 12.2095이므로, 그룹 간 차이가 그룹 내 변동보다 12배 크다는 뜻
   - (결론) 그룹 간 차이가 그룹 내 변동보다 훨씬 크다는 의미
2. (p_value) 서로 다른 세 그룹의 평균이 다른지 아닌지 판단하기
   - p-value는 관측된 차이가 우연일 확률이므로, 이 문제에서 0.0000212(0.002%)
   - alpha는 유의미하다고 인정하는 최대 확률이고, 일반적으로 5% (= 최대 5%까지 유의미하다는 의미)
   - (결론) 세 그룹은 서로 다르다.(유의미) (0.0000212 < 0.05이므로, 세 그룹은 서로 같다는 귀무가설 기각)
     - 세 그룹의 평균이 모두 같다는 가설은 거의 확실히 틀렸다는 의미
