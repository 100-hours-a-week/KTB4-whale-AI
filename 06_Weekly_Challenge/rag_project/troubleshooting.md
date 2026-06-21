# Step A 트러블슈팅 회고록

> RAG 파이프라인 구축 과정(Step A: Indexing Phase)에서 발생한 트러블슈팅을 시계열 순서로 기록한다.
> 전체 항목 작성이 완료된 후, 별도 단계에서 성격별로 분류할 예정이다.

---

## #1. `pytest` 실행 시 `ModuleNotFoundError: No module named 'model'`

### 문제 상황

`document_loader.py`의 테스트 코드(`tests/test_document_loader.py`)를 작성하고 실행했을 때, 다음과 같은 에러가 발생했다.

```
ModuleNotFoundError: No module named 'model'
```

테스트 파일 안에서는 `from model.document_loader import load_document` 형태로 import를 시도하고 있었다.

### 원인 분석

프로젝트 구조는 다음과 같았다.

```
.
├── src
│   └── model
│       └── document_loader.py
└── tests
    └── test_document_loader.py
```

`document_loader.py`는 `src/model/` 디렉토리 안에 있었지만, Python의 `import` 시스템은 기본적으로 **현재 작업 디렉토리(working directory)와, `sys.path`에 명시적으로 등록된 경로**만 탐색한다. `src/model/` 디렉토리는 어디에도 등록되어 있지 않았기 때문에, Python은 `model`이라는 이름의 모듈(패키지)을 찾을 수 없었다.

즉, 근본 원인은 **"`src` 디렉토리가 import 가능한 경로로 등록되지 않았다"**는 것이었다.

### 재현 방법

1. `src/model/` 아래에 모듈 파일을 만든다.
2. 프로젝트 루트가 아닌 다른 경로 설정 없이, `tests/` 디렉토리의 테스트 파일에서 `from model.모듈명 import 함수명` 형태로 import한다.
3. `python -m pytest tests/` 실행 시 동일한 에러가 재현된다.

### 해결 과정

테스트 파일 상단에 `sys.path`를 직접 조작하는 코드를 추가했다.

```python
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from model.document_loader import load_document  # noqa: E402
```

`Path(__file__).resolve().parent.parent`로 테스트 파일 기준 프로젝트 루트를 계산하고, 그 아래 `src` 디렉토리를 `sys.path`에 추가했다. 이후 `from model.document_loader import load_document`가 정상적으로 동작했다.

(이 방식은 추후 `tests/conftest.py` 도입, 최종적으로 `pyproject.toml` 기반 패키지 설치로 대체되었다 — 트러블슈팅 #4~#7 참고)

### 배운 점

- Python의 `import`는 마법처럼 프로젝트 전체를 다 찾아주는 게 아니라, **명시적으로 등록된 경로만 탐색**한다는 것을 직접 에러를 통해 체감했다.
- `src/` 레이아웃(소스 코드를 별도 디렉토리에 두는 구조)을 쓸 경우, **그 경로를 Python에게 알려주는 작업이 별도로 필요하다**는 것을 알게 되었다. 이 문제는 이후 더 표준적인 해결책(패키지 설치)으로 이어지는 시작점이 되었다.

---

## #2. `sentence-transformers`의 `encode()` 반환 타입 불일치 경고 (`Tensor` vs `ndarray`)

### 문제 상황

`embedder.py`에서 `TextEmbedder.encode()` 메서드를 작성했을 때, 에디터(Pylance)에서 다음과 같은 타입 에러가 표시되었다.

```
형식 "Tensor"을 형식 "ndarray[_AnyShape, dtype[Any]]"에 반환하도록 할당할 수 없습니다.
"Tensor"은 "ndarray[_AnyShape, dtype[Any]]"에 할당할 수 없습니다.
```

실제로 코드를 실행하면 `shape: (16, 384)`가 정확히 출력되어, 런타임에서는 전혀 문제가 없었다.

### 원인 분석

`encode()` 메서드는 다음과 같이 반환 타입을 `np.ndarray`로 명시하고 있었다.

```python
def encode(self, texts: list[str]) -> np.ndarray:
    ...
    embeddings = self.model.encode(texts)
    return embeddings
```

그런데 `sentence-transformers`의 `model.encode()`는 내부 설정에 따라 `numpy.ndarray` 또는 `torch.Tensor`를 반환할 수 있도록 타입이 선언되어 있었다. 기본 동작은 `numpy.ndarray`를 반환하지만, 정적 타입 검사기(Pylance)는 **코드를 실행하지 않고 선언된 타입 정의만** 보기 때문에, "이론적으로 `Tensor`가 반환될 수도 있는데, 그걸 `np.ndarray`라고 약속한 함수에서 그대로 반환하고 있다"는 점을 경고한 것이었다.

즉, 근본 원인은 **"함수의 타입 힌트(반드시 `ndarray`)와, 호출하는 라이브러리 함수의 실제 반환 가능 타입(`ndarray` 또는 `Tensor`) 사이의 불일치"**였다.

### 재현 방법

1. `sentence-transformers`의 `SentenceTransformer.encode()`를 별도 파라미터 없이 호출한다.
2. 그 반환값을 `-> np.ndarray`로 타입을 명시한 함수에서 그대로 반환한다.
3. 정적 타입 검사기(Pylance, mypy 등)로 분석하면 동일한 경고가 재현된다.

### 해결 과정

`encode()` 호출 시 `convert_to_numpy=True` 파라미터를 명시적으로 전달했다.

```python
embeddings = self.model.encode(texts, convert_to_numpy=True)
return embeddings
```

이 파라미터는 `sentence-transformers`에게 "결과를 반드시 numpy 배열로 변환해서 반환하라"고 명시적으로 지시한다. 이렇게 하면 함수의 실제 동작과 타입 힌트(`-> np.ndarray`)가 항상 일치하게 되어, 정적 타입 검사기의 경고가 해소되었다.

### 배운 점

- 타입 에러가 **"코드가 실제로 잘못 동작한다"는 뜻이 아니라, "에디터의 정적 분석기가 타입 선언과 실제 가능한 값 사이의 불일치를 경고하는 것"**일 수 있다는 점을 구분하게 되었다. 런타임 결과(실제 실행 출력)와 정적 분석 결과(에디터 경고)는 서로 다른 검증 단계이며, 둘 다 확인하는 습관이 중요하다.
- 외부 라이브러리를 사용할 때, 함수가 **여러 타입을 반환할 수 있도록 유연하게 설계된 경우**, 내가 원하는 타입을 명시적으로 고정하는 파라미터(`convert_to_numpy=True`)가 있는지 확인하는 것이 안전하다는 것을 배웠다.

---

## #3. `self.vectors`의 `Optional` 타입으로 인한 속성/인덱싱 접근 에러

### 문제 상황

`vector_store.py`의 `InMemoryVectorStore` 클래스에서, 그리고 이를 사용하는 테스트 코드(`test_vector_store.py`)에서 각각 다른 형태의 타입 에러가 발생했다.

`vector_store.py`의 `__main__` 블록에서:

```
"shape"은(는) "None"의 알려진 특성이 아님
```

`test_vector_store.py`에서:

```
'None' 유형의 개체는 아래 첨자를 사용할 수 없습니다.
```

### 원인 분석

`InMemoryVectorStore` 클래스는 다음과 같이 `self.vectors`를 선언하고 있었다.

```python
def __init__(self):
    self.chunks: list[str] = []
    self.vectors: np.ndarray | None = None  # 아직 추가된 데이터가 없으면 None
```

`self.vectors`의 타입이 **"`np.ndarray`이거나 `None`일 수 있다"**고 선언되어 있었다. `add()`를 호출하기 전까지는 실제로 `None`이지만, `add()`를 호출한 뒤에는 항상 `np.ndarray`가 된다는 사실은 **코드를 사람이 읽으면 알 수 있지만, 정적 타입 검사기는 코드의 실행 순서(런타임 흐름)를 추적하지 않고 선언된 타입만 본다.**

그 결과, `store.vectors.shape`(속성 접근)이나 `store.vectors[0]`(인덱싱)처럼 `None`에는 존재하지 않는 동작을 시도하는 모든 코드에서, **"혹시 `None`인 상태로 이 줄에 도달하면 어떻게 할 것인가"**라는 경고가 발생했다. 속성 접근과 인덱싱은 코드 형태(syntax)는 다르지만, **근본 원인은 완전히 동일** — `Optional` 타입(`np.ndarray | None`)에 대한 정적 분석기의 보수적인 경고였다.

### 재현 방법

1. 클래스 속성을 `타입 | None`으로 선언하고 초기값을 `None`으로 둔다.
2. 별도의 메서드(`add()`)에서 그 속성을 실제 값으로 채운다.
3. `add()` 호출 이후의 코드에서 그 속성에 `.shape` 같은 속성 접근이나 `[0]` 같은 인덱싱을 시도한다.
4. 정적 타입 검사기는 "호출 순서상 `add()`가 먼저 실행된다"는 사실을 추적하지 못하므로, 두 경우 모두 동일한 종류의 경고가 재현된다.

### 해결 과정

`None`이 아님을 코드에서 명시적으로 확인하는 `assert` 문을 접근 직전에 추가했다.

```python
store = InMemoryVectorStore()
store.add(chunks, vectors)

assert store.vectors is not None  # 타입 검사기에게 "여기서부터는 None이 아니다"를 알려줌

print(store.vectors.shape)
```

테스트 코드에서도 인덱싱하는 모든 위치 앞에 동일한 `assert store.vectors is not None`을 추가했다.

`assert x is not None` 패턴은 정적 타입 검사기가 **"이 지점 이후로는 `x`가 `None`일 가능성을 제외하고 추론해도 된다"**는 신호로 인식하도록 설계되어 있어, 이후 코드에서는 경고가 발생하지 않는다.

### 배운 점

- 표면적으로 다른 에러 메시지(속성 접근 에러 vs 인덱싱 에러)라도, **타입 선언(`Optional`)과 실행 흐름 사이의 간극이라는 동일한 근본 원인**에서 나올 수 있다는 것을 배웠다. 에러 메시지의 문구보다 "왜 이 타입이 이렇게 선언되어 있었는가"를 먼저 보는 습관이 디버깅 속도를 높인다.
- `assert x is not None`은 단순히 "타입 검사기를 통과시키기 위한 우회"가 아니라, **실제로 그 가정이 깨졌을 때 `AssertionError`로 즉시 알려주는 정직한 안전장치**라는 점에서, 타입 안정성과 런타임 안정성을 동시에 높이는 방법이라는 것을 이해했다.
