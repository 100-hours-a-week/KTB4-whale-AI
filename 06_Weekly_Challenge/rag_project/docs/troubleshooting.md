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

---

## #4. 컴퓨터 재시작 후 VSCode Pylance가 `model.chunker` import를 인식하지 못함

> 이 트러블슈팅부터 #7까지는 **하나의 근본 문제("import 경로를 어떻게 안정적으로 해결할 것인가")에 대해, 점점 더 넓은 범위의 해결책을 순서대로 시도해 나간 연속된 과정**이다.

### 문제 상황

이전까지(#1) `sys.path.append`로 import 문제를 해결하고 정상적으로 작업을 진행하고 있었다. 그런데 컴퓨터를 재시작하고 VSCode를 다시 열었을 때, 동일한 코드에서 다음과 같은 에디터 경고가 새로 나타났다.

```
가져오기 "model.chunker"을(를) 확인할 수 없습니다.
```

실제로 `python -m pytest`로 실행하면 정상적으로 통과했다 — 에디터(Pylance)에서만 보이는 경고였다.

### 원인 분석

`sys.path.append`는 **코드가 실제로 실행되는 순간(런타임)**에만 효과가 있는 방법이다. 그런데 Pylance는 코드를 실행하지 않고 **정적으로(파일을 읽기만 하고)** 분석하기 때문에, "이 줄이 실행되면 `sys.path`에 `src`가 추가된다"는 동적인 사실을 추적하지 못한다.

어제까지는 VSCode의 Python Interpreter 설정이나 Pylance의 캐시된 분석 정보가 우연히 맞아서 경고가 뜨지 않았을 가능성이 있으나, 재시작 과정에서 이 상태가 초기화되며 Pylance가 다시 `model` 패키지의 위치를 알 수 없는 상태로 돌아간 것으로 추정된다.

근본 원인: **`sys.path.append`는 런타임 해결책이고, Pylance는 정적 분석기이므로, 애초에 이 방법으로는 에디터 경고를 근본적으로 해결할 수 없었다.**

### 재현 방법

1. `sys.path.append`로만 import 경로를 해결한 프로젝트를 구성한다.
2. VSCode로 해당 프로젝트를 열고 Python Interpreter를 가상환경으로 설정한다.
3. VSCode를 완전히 재시작(`Reload Window` 또는 컴퓨터 재시작)한다.
4. `sys.path.append`에 의존하는 import 문이 있는 파일을 열면 동일한 경고가 재현될 수 있다 (Pylance의 분석 캐시 상태에 따라 재현 여부가 달라질 수 있음).

### 해결 과정

가장 작은 범위의 해결책(`sys.path.append`)이 실패했으므로, 그보다 한 단계 넓은 범위의 해결책부터 순서대로 시도했다.

먼저 `PYTHONPATH` 환경변수를 터미널에서 직접 설정했다.

```bash
export PYTHONPATH="$PWD/src"
```

- (개선 내용) `sys.path.append`는 **코드 자신이, 코드가 실행되는 순간**에 경로를 추가하는 방식이었던 반면, `PYTHONPATH`는 **터미널(셸)이 Python을 실행하기 직전에** 미리 `sys.path`에 경로를 넣어두는 방식이다. 이렇게 설정하면 코드 안의 `sys.path.append` 줄을 지워도 동일하게 동작했다 — 즉, 코드를 더 깨끗하게 만들 수 있다는 장점이 확인되었다.
- (한계점) 같은 터미널 세션에서는 정상 동작했지만, **새 터미널 창을 열어서 같은 명령어를 실행하면 `ModuleNotFoundError`가 다시 발생**했다. `echo $PYTHONPATH`로 확인한 결과, 새 터미널에는 이 환경변수가 비어 있었다 — `export`로 설정한 환경변수는 **그 셸 세션에만 종속**되고, 터미널을 새로 열면 사라진다는 것이 직접 실험으로 확인되었다.
- (한계점) 이 방법 역시 **Pylance의 import 경고 자체는 해결하지 못했다** — `PYTHONPATH`도 결국 "코드 실행 시점"의 경로 설정이라, 코드를 실행하지 않는 정적 분석기는 여전히 이를 추적하지 못했다.

### 배운 점

- **"런타임에서 잘 동작한다"는 것과 "에디터에서 경고 없이 보인다"는 것은 서로 다른 검증 축**이라는 것을 명확히 이해하게 되었다. `pytest` 실행 결과만 보고 "문제없다"고 판단하면, 에디터 차원의 문제를 놓칠 수 있다.
- `sys.path.append`와 `PYTHONPATH`는 **결과적으로 같은 일(경로를 `sys.path`에 추가)을, 누가 언제 하는지만 다르게** 수행하는 방법이라는 것을 직접 비교하며 이해했다. 코드 안에 적던 것을 터미널 설정으로 옮긴 것뿐이라, "반복해야 하는 단위"가 파일에서 터미널 세션으로 바뀌었을 뿐 근본적인 한계(매번 다시 설정해야 함)는 해결되지 않았다.
- 이 한계를 직접 실험(새 터미널에서 재현)으로 확인한 것이 다음 해결책(반복을 자동화하는 방법)으로 넘어갈 명확한 근거가 되었다.

---

## #5. Import 경로 문제 — `conftest.py`로 반복 설정 제거 시도

> 이 항목은 #4에서 이어지는 동일한 근본 문제("import 경로를 어떻게 안정적으로 해결할 것인가")에 대한 다음 단계의 해결책 시도이다.

### 문제 상황

`PYTHONPATH`(#4)는 코드의 `sys.path.append`를 제거할 수 있게 해주었지만, **터미널을 새로 열 때마다 사람이 직접 다시 설정해야 한다**는 한계가 있었다. 이 "반복 설정" 문제를 해결하기 위한 다음 시도가 필요했다.

### 원인 분석

`PYTHONPATH`의 한계는 "환경변수가 특정 셸 세션에만 종속된다"는 점에서 비롯되었다. 이를 해결하려면, **사람이 매번 손으로 설정하지 않아도, 특정 도구(이 경우 `pytest`)가 실행될 때 자동으로 경로 설정이 적용되는 방법**이 필요했다.

`pytest`는 테스트를 실행하기 전에 `conftest.py`라는 이름의 특수 파일을 찾으면 자동으로 먼저 읽어들이는 내부 규칙을 가지고 있다 — 이는 pytest 공식 문서(https://docs.pytest.org/en/stable/reference/fixtures.html)에 명시된 동작이다. 이 파일에 경로 설정 로직을 한 번만 적어두면, `tests/` 디렉토리 안의 모든 테스트 파일이 자동으로 그 설정을 공유한다.

### 재현 방법

1. `tests/` 폴더 안에 여러 개의 테스트 파일이 있고, 각 파일이 개별적으로 `sys.path.append`를 호출하고 있는 상태를 만든다.
2. 이 중복된 설정 코드를 모든 파일에서 제거한다.
3. `tests/conftest.py`라는 이름의 파일을 만들지 않은 상태로 테스트를 실행하면 `ModuleNotFoundError`가 재현된다.

### 해결 과정

`tests/conftest.py` 파일을 생성하고, 기존에 각 테스트 파일에 중복되어 있던 경로 설정 로직을 이 파일로 옮겼다.

- (개선 내용) 터미널을 새로 열 때마다 사람이 직접 설정할 필요가 없어졌다 — `pytest`를 실행하는 것 자체가 `conftest.py`를 자동으로 읽어들이기 때문에, **설정이 "한 번 작성하면 계속 유지되는" 형태**로 바뀌었다. 실제로 4개 테스트 파일에서 중복 코드를 제거한 뒤에도 19개 테스트가 모두 정상적으로 통과했다.
  - (추가 수정) 4개 테스트 파일 각각에서 아래 두 줄을 제거했다.

    ```python
    import sys
    from pathlib import Path

    SRC_DIR = Path(__file__).resolve().parent.parent / "src"
    sys.path.append(str(SRC_DIR))
    ```

- (한계점) 이 방법은 `pytest`를 통해 실행할 때만 적용된다. `python src/model/chunker.py`처럼 파일을 직접 실행하는 경우나, VSCode의 Pylance(정적 분석기)에는 영향을 주지 않았다 — 실제로 conftest.py를 적용한 후에도 VSCode의 import 경고는 그대로 남아 있었다.

### 배운 점

- `conftest.py`는 "코드 안에 반복해서 적어야 하는 설정을, 도구가 인식하는 표준 위치 하나로 모으는" 방법이라는 것을 이해했다. `PYTHONPATH`가 "사람이 매번 반복"해야 했던 것을, `conftest.py`는 "pytest가 자동으로 한 번에 처리"하도록 바꾼 것이다.
- 다만 이 방법도 결국 내부적으로 `sys.path.append`를 사용하고 있다는 점에서, **"실행 시점에 경로를 추가하는" 근본적인 접근 방식 자체는 #1, #4와 동일**하다. 적용 범위가 넓어졌을 뿐, "정적 분석기에는 적용되지 않는다"는 한계는 여전히 해결되지 않았다는 것을 확인했다.
- 공식 문서(pytest 공식 레퍼런스)를 직접 찾아서 "왜 이 파일명이 특별한지"를 검증한 것이, 단순히 동작을 따라 하는 것보다 더 확실한 근거를 갖게 해주었다.

---

## #6. Import 경로 문제 — `.vscode/settings.json`으로 Pylance 전용 경고 해결 시도

> 이 항목은 #4, #5에서 이어지는 동일한 근본 문제에 대한 다음 단계의 해결책 시도이다.

### 문제 상황

`conftest.py`(#5)는 pytest 실행 시의 반복 설정 문제는 해결했지만, **VSCode(Pylance)의 import 경고는 여전히 그대로 남아 있었다.** 코드가 정상 동작하는 것과 무관하게, 에디터에서 빨간 줄이 계속 표시되는 상태였다.

### 원인 분석

`conftest.py`도 결국 내부적으로 `sys.path.append`를 사용하는 방식이다. 이는 **코드가 실행되는 시점**에만 효과가 있는 런타임 해결책이고, Pylance는 코드를 실행하지 않고 정적으로 분석하기 때문에 이 효과를 추적할 수 없다는 점은 #4에서 확인한 것과 동일했다.

따라서 근본 원인은 #4와 같다 — **Pylance는 런타임 경로 조작을 추적하지 못하므로, Pylance 자신에게 직접 경로를 알려주는 별도의 설정이 필요하다.**

### 재현 방법

1. `sys.path.append` 또는 `conftest.py`로만 import 경로를 해결한 프로젝트를 구성한다.
2. VSCode로 해당 프로젝트를 열고, import 문이 있는 파일을 확인한다.
3. `python.analysis.extraPaths` 같은 Pylance 전용 설정이 없는 상태에서는 동일한 import 경고가 재현된다.

### 해결 과정

프로젝트 루트에 `.vscode/settings.json` 파일을 생성했다.

- (개선 내용) Pylance가 `./src`를 import 가능한 경로로 직접 인식하게 되어, 에디터의 import 경고가 사라졌다. VSCode의 `Python: Select Interpreter`로 가상환경을 재선택하는 작업과 함께 적용한 뒤 정상적으로 해결되었다.
  - (추가 수정) `.vscode/settings.json` 파일을 새로 만들고 아래 내용을 작성했다.

    ```json
    {
      "python.analysis.extraPaths": ["./src"],
      "python.autoComplete.extraPaths": ["./src"]
    }
    ```

- (한계점) 이 설정은 **VSCode(Pylance)에만 적용**되는 에디터 전용 설정이다. 다른 에디터(PyCharm 등)나 다른 컴퓨터에서 이 프로젝트를 열면 동일한 설정 파일이 없으므로 효과가 없다. 또한 이 방법은 #1, #4, #5와 별개로 **새로운 설정 파일을 또 하나 추가**한 것이라, "새 프로젝트를 만들 때마다 반복해야 하는가?"라는 근본적인 의문이 남았다.

### 배운 점

- "코드 실행 시점에 적용되는 해결책(런타임)"과 "정적 분석기에 적용되는 해결책(에디터)"은 **완전히 분리된 별개의 문제**라는 것을 다시 한번 명확히 이해했다. 하나를 해결해도 다른 하나는 자동으로 해결되지 않는다.
- 이 방법으로 문제는 해결되었지만, **"매번 다른 프로젝트를 만들 때마다 이 설정을 또 추가해야 하는가"**라는 의문이 들었다. 이는 "임시방편(workaround)으로 증상을 해결하는 것"과 "근본적으로 문제 자체가 발생하지 않도록 만드는 것"의 차이를 고민하게 된 계기가 되었고, 다음 단계(`pyproject.toml`을 통한 정식 패키지화)로 넘어가는 직접적인 동기가 되었다.

---

## #7. Import 경로 문제 — `pyproject.toml` + `pip install -e .`로 최종 해결

> 이 항목은 #1, #4, #5, #6에서 이어져 온 동일한 근본 문제("import 경로를 어떻게 안정적으로 해결할 것인가")에 대한 최종 해결책이다.

### 문제 상황

지금까지 시도한 네 가지 방법(`sys.path.append`, `PYTHONPATH`, `conftest.py`, `.vscode/settings.json`)은 모두 각자의 범위(파일, 터미널 세션, pytest, 에디터) 안에서만 작동했고, **하나의 설정으로 모든 도구(pytest 실행, 직접 실행, 에디터)에 동시에 적용되는 방법은 없었다.**

### 원인 분석

지금까지의 모든 방법은 본질적으로 **"Python에게 `src` 디렉토리의 위치를 그때그때 알려주는" 방식**이었다 — 코드 안에서, 터미널에서, pytest 설정 파일에서, 또는 에디터 설정에서. 즉 "이 프로젝트가 무엇인지"를 Python 생태계 자체에는 한 번도 등록한 적이 없었다.

Python에는 이를 위한 표준 메커니즘이 있다 — 프로젝트를 **정식 패키지로 설치**하면, `sys.path` 조작이나 별도의 설정 파일 없이도 Python 인터프리터, pytest, Pylance를 포함한 모든 도구가 그 패키지를 동일하게 인식한다. 이때 `pip install -e .`(editable install)를 사용하면, 패키지를 복사해서 설치하는 대신 **프로젝트 폴더 자체를 참조**하도록 설치되어, 소스 코드를 수정할 때마다 재설치할 필요가 없다.

### 재현 방법

1. `src/model/` 레이아웃의 프로젝트에서, `pyproject.toml` 없이 `sys.path.append`/`conftest.py`/`.vscode/settings.json` 중 일부만 적용한 상태를 만든다.
2. 위 설정들이 적용되지 않는 도구(예: 다른 에디터, 또는 동일 에디터의 새 워크스페이스)에서 같은 프로젝트를 연다.
3. 동일한 import 경고 또는 에러가 그 도구에서는 재현된다 — 각 해결책이 적용 범위 밖에서는 항상 무력하다는 것이 재현된다.

### 해결 과정

프로젝트 루트에 `pyproject.toml` 파일을 생성했다.

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-project"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]
```

`[tool.setuptools.packages.find]`의 `where = ["src"]`가 "패키지들은 `src` 폴더 아래에 있다"는 것을 명시하는 부분이며, 이는 현재 프로젝트의 src layout 구조와 정확히 일치한다.

이후 가상환경이 활성화된 상태에서 다음 명령어로 패키지를 설치했다.

```bash
pip install -e .
```

- (개선 내용) 별도의 경로 설정 없이, `pytest` 실행과 VSCode(Pylance) 양쪽에서 동시에 import 문제가 해결되었다. VSCode를 재시작(`Reload Window`)한 뒤 import 경고가 완전히 사라졌고, `python -m pytest tests/ -v`로도 19개 테스트가 모두 정상 통과했다.
  - (추가 수정) 더 이상 필요 없어진 `tests/conftest.py`, `.vscode/settings.json` 파일을 삭제했다.
- (주의 사항) 패키지 설치(`pip install -e .`)라는 한 단계가 프로젝트 설정 과정에 추가되므로, 이 프로젝트를 처음 받는 사람(또는 다른 컴퓨터)은 반드시 이 명령어를 먼저 실행해야 한다는 점을 알고 있어야 한다 — 다만 이는 Python 패키지 배포의 표준적인 절차이므로, 특별한 약점이라기보다는 "표준을 따르기 위한 최소한의 절차"로 볼 수 있다.

### 배운 점

- 지금까지의 네 가지 방법(#1, #4, #5, #6)은 모두 "Python에게 경로를 그때그때 알려주는" 동일한 계열의 해결책이었고, 각자 적용 범위(파일/세션/도구/에디터)만 달랐다는 것을 이 마지막 단계에 이르러서야 전체적으로 조망할 수 있었다. **문제를 해결하는 범위를 점차 넓혀가다 보면, 결국 "그때그때 알려주기"가 아니라 "원천적으로 등록하기"라는 다른 범주의 해결책이 필요해진다**는 것을 직접 체험했다.
- `src/` 레이아웃과 `pyproject.toml` 기반의 editable install은 NumPy, Pandas를 포함한 다수의 Python 라이브러리가 실제로 채택하는 표준 프로젝트 구조라는 것을 알게 되었다. 처음에 겪었던 단순한 import 에러가, 결국 "Python 패키지를 어떻게 구조화하고 배포하는가"라는 더 큰 주제로 이어졌다는 점에서, 작은 문제 하나가 더 넓은 생태계 이해로 확장될 수 있다는 것을 경험했다.

---

## #8. Fixed-size Chunking으로 인한 검색 품질 저하 — 핵심 정보가 상위 k개에서 누락됨

> 이 항목은 #1~#7과 달리 코드 에러가 아니라, **Retrieval 결과의 품질(quality) 문제**를 다룬다.

### 문제 상황

`"What is the default API port for NimbusFlow?"`라는 질문으로 Retrieval을 실행했을 때, 실제 정답("포트 8842")이 담긴 chunk가 top-3 검색 결과에 전혀 포함되지 않았다. 대신 설치 경로, 버전 정보처럼 질문과 표면적으로만 연관된 chunk들이 상위에 노출되었다.

### 원인 분석

디버깅 스크립트로 전체 16개 chunk의 유사도 점수를 직접 확인한 결과, "8842"가 포함된 chunk(인덱스 6)는 실제로 **16개 중 3위(score=0.6546)**를 기록했다. 그런데 1위(chunk 3, score=0.7273)와 2위(chunk 1, score=0.7245)가 근소한 차이로 더 높은 점수를 받아, k=3 검색에서 정작 가장 관련성이 높아야 할 chunk가 경계에 걸려 누락되는 경우가 발생했다.

내용을 직접 살펴보면, "8842"가 포함된 chunk는 다음과 같이 구성되어 있었다.

```
...to alternate between local execution and cloud execution depending on a metric
called the Drift Score. If the Drift Score exceeds 0.73, NimbusFlow automatically
switches execution to the cloud worker pool.

## 4. API Usage

NimbusFlow exposes a REST API on port `8842` by default.

### 4.1 Authentication
...
```

`chunk_size=300, chunk_overlap=50`으로 고정 길이 분할(fixed-size chunking)을 하다 보니, **"## 4. API Usage" 섹션의 핵심 문장(포트 8842)이, 전혀 다른 주제(Drift Score, 인증 토큰)와 한 chunk 안에 함께 섞여 들어갔다.** 이로 인해 해당 chunk의 임베딩 벡터가 "API 포트"라는 단일 주제를 강하게 대표하지 못하고, 여러 주제가 섞인 모호한 벡터가 되어 유사도 점수가 희석된 것으로 판단된다.

반면 1, 2위로 검색된 chunk들은 "NimbusFlow", "default"라는 단어를 반복적으로 포함하고 있어, 질문에 포함된 "default"라는 표면적 단어와 더 강하게 일치한 것으로 보인다.

근본 원인: **Fixed-size chunking은 문서의 의미 단위(섹션, 문단)를 고려하지 않고 글자 수로만 자르기 때문에, 서로 무관한 내용이 한 chunk에 뒤섞이고, 그 결과 핵심 정보를 담은 chunk의 임베딩이 약화될 수 있다.**

### 재현 방법

1. 여러 개의 독립된 주제(섹션)가 포함된 문서를 준비한다.
2. 섹션 경계와 무관하게 고정된 글자 수(`chunk_size`)로 chunking한다.
3. 어떤 섹션의 핵심 문장이 chunk의 일부에만, 그것도 다른 주제와 함께 섞여 들어가도록 경계가 형성되는 경우가 생긴다.
4. 그 섹션에 대한 질문으로 검색하면, 핵심 정보가 담긴 chunk가 표면적 단어 일치도가 높은 다른 chunk에 밀려 상위 k위 안에 들지 못하는 현상이 재현될 수 있다.

### 해결 과정

(이 항목은 원인 진단까지 완료한 상태이며, 실제 개선은 Step B(Query Phase) 완성 이후 별도로 진행하기로 결정했다. 실제 구현 및 검증 결과는 트러블슈팅 #12에서 이어서 다룬다.)

- (주의 사항) 이 문제는 코드 버그가 아니라 **Chunking 전략의 설계 한계**이므로, Retrieval 알고리즘(`cosine_similarity`, `retrieve_top_k`) 자체를 수정해서 해결할 수 있는 문제가 아니다. 근본적인 해결은 Chunking 단계로 거슬러 올라가야 한다.
- (개선 방향 후보)
  - k값을 늘려 누락 위험을 줄이는 방법 (단, distractor chunk가 함께 포함될 위험과 트레이드오프 존재)
  - Section-based chunking으로 전환하여, 섹션 경계를 chunk 경계와 일치시키는 방법
  - `chunk_size`를 줄여 한 chunk에 여러 주제가 섞일 가능성을 낮추는 방법

### 배운 점

- Retrieval 코드가 모든 테스트(8개)를 통과했다는 사실과, 실제 도메인 질문에서 정답을 찾아내는 것은 **서로 다른 차원의 검증**이라는 것을 직접 체감했다. 단위 테스트는 "알고리즘이 설계대로 동작하는가"를 보장하지만, "그 설계가 실제로 좋은 결과를 만드는가"는 별도로 점검해야 한다.
- Chunking 전략(표 28에서 미리 검토했던 fixed-size vs section-based의 trade-off)이 단순한 설계 선택이 아니라, **실제 검색 품질에 직접적인 영향을 미치는 핵심 변수**라는 것을 추상적인 이론이 아니라 실제 데이터로 확인했다.
- 디버깅 스크립트(`debug_retrieval.py`)를 별도로 작성해 "어떤 chunk가 몇 위인지"를 직접 눈으로 확인한 것이, 추측에 의존하지 않고 근본 원인을 정확히 짚는 데 핵심적이었다 — 막연히 "임베딩이 이상한가보다"라고 넘기지 않고, 실제 순위와 점수를 출력해 검증한 것이 정확한 진단으로 이어졌다.

---

## #9. Generation용 LLM 사이즈 선택 — 로컬 환경 제약과 "근본에서 확장" 원칙의 적용

> 이 항목은 에러 트러블슈팅이 아니라, **설계 결정(design decision) 과정**을 기록한 것이다.

### 문제 상황

Generation 단계에서 사용할 Gemma 4의 구체적인 사이즈를 결정해야 했다. Gemma 4는 E2B, E4B, 26B(A4B, MoE), 31B(Dense) 네 가지 사이즈로 제공되며, 실행 환경은 로컬 Mac M3(통합 메모리 18GB)였다.

### 검토 과정

먼저 4가지 사이즈의 특성을 비교했다.

| 모델           | 파라미터 규모                                            | 아키텍처                           | 권장 환경                                |
| -------------- | -------------------------------------------------------- | ---------------------------------- | ---------------------------------------- |
| E2B            | 약 2.3B effective, Q4 양자화 시 2GB RAM                  | Dense + Per-Layer Embeddings (PLE) | Raspberry Pi, 스마트폰급 edge 기기       |
| E4B            | 약 4.5B effective, Q4_K_M 기준 약 2.5GB 디스크/4~5GB RAM | Dense + PLE                        | 중급 노트북/태블릿                       |
| 26B (A4B, MoE) | 총 26B 중 4B만 활성화                                    | Mixture of Experts                 | 8GB RAM에서 Q4 양자화로 노트북 동작 가능 |
| 31B (Dense)    | 31B 전체 파라미터 활성화                                 | Dense                              | 단일 80GB H100 GPU급 환경                |

18GB 통합 메모리 환경에서는, 31B는 명백히 과도했고, 26B(MoE)도 활성 파라미터는 4B뿐이지만 라우팅을 위해 26B 전체를 메모리에 올려야 한다는 점에서 부담이 있었다. 따라서 **실질적인 후보는 E2B와 E4B 두 가지로 압축**되었다.

(참고: 이 검토에 앞서, 프로젝트 초반에 언급되었던 "Colab GPU L4 사용 중"이라는 정보를 그대로 사용해 한 차례 검토했던 적이 있었으나, 실제 작업이 이미 `.py` 파일과 `pytest` 기반의 로컬 Mac M3 환경으로 진행되고 있다는 점을 다시 확인하고, 올바른 환경(Mac M3, 18GB) 기준으로 재검토했다.)

### 최종 결정

E2B와 E4B 둘 다 메모리 제약상으로는 문제가 없었으므로, 둘 사이의 선택은 순수히 **학습 방법론**의 문제였다. 이 프로젝트에서는 지금까지 다음과 같은 사례들에서 **"근본(기본)에서 확장" 원칙**을 일관되게 적용해 왔다.

- Import 경로 해결: `sys.path.append`(가장 단순) → `PYTHONPATH` → `conftest.py` → `.vscode/settings.json` → `pyproject.toml`(가장 표준적) 순으로, 가장 작은 해결책부터 시도하고 한계가 보일 때만 확장
- Chunking 전략: Fixed-size(가장 단순) 방식을 먼저 학습용으로 선택하고, Section-based는 추후 개선 후보로 남겨둠

같은 원칙을 적용하여, **가장 작은 사이즈인 E2B를 먼저 선택**하기로 결정했다.

- (개선 내용) 메모리 여유가 충분해 E4B를 선택할 수도 있었지만, 학습 목적상 "가장 작은 것에서 한계를 직접 확인한 뒤 확장한다"는 원칙을 동일하게 적용했다.
  - (추가 개선) 만약 E2B의 답변 품질이 부족하거나 메모리 문제가 발생하면, 그 시점에 E4B로 확장하고 이를 별도의 트러블슈팅 항목으로 기록하기로 결정했다.
- (주의 사항) 이번 결정은 코드 변경이 아니라 **계획 단계의 결정**이므로, 실제로 E2B를 사용해본 뒤에 추가적인 문제가 발생할 가능성이 있으며, 그 경우는 별도 항목(#10 이후)으로 이어질 예정이다.

### 배운 점

- 학습 원칙(근본에서 확장)은 **특정 기술 영역(예: import 경로, chunking)에만 한정되는 규칙이 아니라, 프로젝트의 모든 기술적 의사결정(이번에는 LLM 사이즈 선택)에 동일하게 적용할 수 있는 상위 원칙**이라는 것을 확인했다. 메모리 여유가 충분하다는 사실이, 더 큰 모델을 선택할 이유가 되는 것은 아니었다.
- 의사결정의 전제(이번 경우 실행 환경)는 프로젝트가 진행되며 바뀔 수 있으므로, 중요한 결정을 내리기 전에 "지금 이 전제가 여전히 유효한가"를 다시 확인하는 절차가 필요하다는 것을 배웠다.

---

## #10. Base 모델과 Instruction-tuned 모델을 구분하지 않아 발생한 무한 반복 생성

### 문제 상황

`google/gemma-4-E2B`(이하 base 모델)로 Generation을 실행했을 때, 정답("Project Driftwood")은 정확히 생성했지만, 그 이후 멈추지 않고 `Question: ... Answer: ...` 형태의 새로운 질문-답변 쌍을 계속 만들어내며 `max_new_tokens`을 모두 채울 때까지 반복 생성했다.

### 원인 분석

처음에는 EOS 토큰 미지정이 원인이라 판단해 `eos_token_id`, `pad_token_id`를 추가했으나 해결되지 않았다.

진짜 원인은 모델 자체가 "답변 후 멈춰야 한다"는 행동을 학습하지 않았다는 것이었다.

- `google/gemma-4-E2B`는 instruction-tuned(`-it`)가 아닌 **base(사전학습) 모델**이다. Base 모델은 다음 토큰을 통계적으로 이어 쓰도록만 학습되어 있을 뿐, "질문에 답하고 멈춘다"는 행동은 학습한 적이 없다.
- 반면 instruction-tuned 모델은 **"지시문-응답 쌍"으로 base 모델을 추가 학습(SFT + RLHF)시킨 결과물**이며, 이 추가 학습 데이터에 "응답 후 종료"라는 패턴이 포함되어 있다.
  (https://toloka.ai/blog/base-llm-vs-instruction-tuned-llm, https://arxiv.org/html/2406.14972v1)

근본 원인: 트러블슈팅 #9에서 "근본에서 확장" 원칙으로 **사이즈(E2B/E4B/26B/31B)** 축은 가장 작은 것을 선택했지만, **학습 방식(base/instruction-tuned)**이라는 별도의 축을 고려하지 않았다. `-it`는 사이즈를 키우는 것이 아니라, 같은 E2B 위에 "응답 종료" 행동을 추가 학습시킨 것이므로, `-it`로 전환해도 "근본에서 확장" 원칙과 충돌하지 않는다.

- 추가로, base 모델(`google/gemma-4-E4B`) 의 Hugging Face 페이지조차 예제 코드에서 `MODEL_ID = "google/gemma-4-E4B-it"`로 `-it` 모델을 직접 지정하고 있었다 (https://huggingface.co/google/gemma-4-E4B).
- 또한 Gemma 4는 `tokenizer_config.json`에 chat template이 없어 `AutoTokenizer.apply_chat_template()`이 기본 설정으로는 동작하지 않는다는 점도 확인했다 (https://github.com/huggingface/transformers/issues/45205).

### 재현 방법

1. instruction-tuned(`-it`)가 아닌 base 모델을 `AutoModelForCausalLM.from_pretrained()`로 로딩한다.
2. `"Question: {question}\n\nAnswer:"` 형태로 끝나는 prompt를 구성한다.
3. `do_sample=False`(greedy decoding)로 `model.generate()`를 호출하면, 답변 생성 후에도 같은 Q&A 패턴이 반복 생성되는 현상이 재현된다.

### 해결 과정

- (개선 내용) 모델을 `google/gemma-4-E2B`(base)에서 `google/gemma-4-E2B-it`(instruction-tuned)로 교체했다. Instruction-tuned 모델은 "사용자 질문에 답하고 멈춘다"는 지시를 따르도록 별도로 학습되어 있어, 반복 생성 문제가 해결되었다.
  - (추가 수정) `AutoTokenizer`로 직접 prompt를 토큰화하던 방식을, Gemma 4의 공식 chat template을 적용하는 `AutoProcessor.apply_chat_template()` 방식으로 변경했다. 이를 통해 모델이 학습된 형식(turn 구조)에 맞는 입력을 받게 되었다.
  - (추가 수정) 더 이상 필요하지 않은 `eos_token_id`, `pad_token_id` 수동 지정 코드를 제거했다 — chat template이 자동으로 올바른 종료 마커를 포함시켜주기 때문이다.
- (주의 사항) instruction-tuned 모델로 전환하면서, base 모델과는 별개의 가중치 파일을 새로 다운로드해야 했다(약 10GB, 트러블슈팅 #11에서 이어짐).

### 배운 점

- LLM을 고를 때, 모델 이름에 붙은 접미사(`-it`, `-base` 등)가 단순한 버전 표시가 아니라 **모델의 근본적인 학습 방식(지시 따르기 여부)을 가리키는 중요한 정보**라는 것을 직접 겪고 나서 이해했다.
- "정답을 정확히 생성했다"는 사실만으로 Generation 단계가 잘 동작한다고 판단하면 안 된다는 것을 배웠다 — 정답 생성과 "언제 멈춰야 하는지를 아는 것"은 서로 다른 능력이며, 후자가 부족하면 RAG 시스템의 실제 응답 품질(불필요한 텍스트가 섞여 나오는 문제)에 직접적인 영향을 준다.
- 증상(반복 생성)을 보고 가장 먼저 떠올린 원인(EOS 토큰 미지정)이 틀렸을 수 있다는 것, 그리고 한 가지 수정으로 해결되지 않으면 더 근본적인 가정(모델 자체의 종류)까지 의심해봐야 한다는 디버깅 태도를 다시 확인했다.
- 메커니즘을 검증하는 과정에서 확인한 위 논문(arxiv.org/html/2406.14972v1)은, "RAG에서 흔히 instruct 모델을 쓰는 관행과 달리, 실험 환경에서는 base 모델이 instruct 모델보다 평균 20% 더 높은 정확도를 보였다"는 상반된 결과도 함께 제시하고 있었다. 이는 instruction-tuned 모델이 항상 더 우수하다고 단정할 수 없다는 것을 보여준다 — 다만 그 논문이 다루는 문제(정답의 정확도)와 우리가 겪은 문제(응답을 멈추는 기본 동작)는 서로 다른 차원이므로, 이번 트러블슈팅에서 `-it`를 선택한 결정 자체와는 충돌하지 않는다는 것도 함께 확인해두었다.

---

## #11. 공식 경로(AutoProcessor) 채택을 위한 의존성 확장 — 비용 대비 판단 과정

> 이 항목은 에러 트러블슈팅이 아니라, **"근본에서 확장(최소 의존성)" 원칙과 "공식 문서 우선" 원칙이 충돌했을 때, 실제 비용을 따져 판단을 내린 과정**을 기록한다.

### 문제 상황

`AutoProcessor.apply_chat_template()` 방식(트러블슈팅 #10에서 채택)으로 전환한 뒤, `AutoProcessor.from_pretrained()` 실행 시 `Gemma4Processor requires the PIL library but it was not found` 에러가 발생했다. RAG 파이프라인은 텍스트만 다루는데, 이미지 처리용 라이브러리(`Pillow`)가 요구된 것이다.

### 원인 분석

`AutoProcessor`는 텍스트와 이미지를 모두 처리할 수 있는 멀티모달 인터페이스이며, Gemma 4가 멀티모달 모델이기 때문에 `Gemma4Processor` 클래스 자체가 이미지 처리 의존성을 항상 요구하도록 구성되어 있었다. 텍스트만 사용하더라도, 클래스 자체의 요구사항에서 벗어날 수 없었다.

이 시점에서 "근본에서 확장"(최소 의존성: `AutoTokenizer` + `chat_template.jinja` 수동 로딩)과 "공식 문서 우선"(공식 예제와 동일한 `AutoProcessor` 유지) 두 원칙이 충돌했다. `AutoTokenizer` 방식은 GitHub 이슈의 사용자 코멘트에서 제안된 비공식 우회책이라는 점이 확인되어, 단순히 "더 가벼운 쪽"을 택하기보다 **실제 비용을 비교**하는 방식으로 판단을 진행했다.

### 재현 방법

1. 멀티모달 모델의 `AutoProcessor`를, 텍스트만 사용하는 상황에서 그대로 사용한다.
2. 이미지 처리 의존성(`Pillow`, 이어서 `torchvision`)이 설치되어 있지 않으면, 의존성이 하나씩 드러나는 형태로 `ImportError`/`ModuleNotFoundError`가 연쇄적으로 발생한다.

### 해결 과정

- (개선 내용) `Pillow`(4.7MB) 설치 후 재실행하자, 추가로 `torchvision`을 요구하는 에러가 발생했다. 이 시점에 "의존성이 어디까지 이어질지 모른다"는 우려가 있었으나, `torchvision`(1.9MB)을 설치한 뒤에는 추가 의존성 요구 없이 모델이 정상 로딩되었다.
  - (추가 수정) 누적 추가 용량은 `Pillow` + `torchvision` = 약 6.6MB로, 이미 다운로드한 모델 가중치(10GB)에 비해 무시할 수준임을 확인했다.
- (주의 사항) 이번에는 의존성 체인이 2단계에서 끝났지만, 멀티모달 인터페이스를 텍스트 전용 작업에 사용할 경우 의존성이 몇 단계까지 이어질지는 사전에 알 수 없다는 점은 일반적인 위험으로 남아 있다.

### 배운 점

- "근본에서 확장" 원칙은 절대적인 규칙이 아니라, **그 원칙을 적용함으로써 얻는 실질적 이득과, 그로 인해 포기해야 하는 안정성(공식 경로) 사이의 비용을 비교해서 적용해야 한다**는 것을 깨달았다. 이번 경우, 최소 의존성으로 얻는 이득(6.6MB 절약)은 공식 경로를 포기하면서 감수해야 하는 위험(비공식 우회, 향후 호환성 문제)에 비해 훨씬 작았다.
- 의존성 문제를 만났을 때, 첫 에러 하나만 보고 "이 경로는 너무 무겁다"고 단정하기보다, **실제로 몇 단계까지 이어지는지 한두 번은 직접 따라가 보고 판단하는 것**이 추측에만 의존한 판단보다 더 근거 있는 결정으로 이어진다는 것을 확인했다.
- 다만 이 "한두 번까지 따라가 본다"는 기준 자체도 명확한 규칙은 아니므로, 실제로 의존성이 계속 이어지는 상황이라면 어느 시점에 우회 경로로 전환할지 미리 정해두는 것이 필요하다는 것을 인지했다.

---

## #12. Section-based Chunking 구현 및 검증 — 섹션 간 혼동은 해결, 섹션 내부 항목 혼동은 새로 발견

> 이 항목은 트러블슈팅 #8(Fixed-size chunking의 검색 품질 저하)에서 보류했던 개선 작업을, Step B(Query Phase) 완성 후 이어서 진행한 결과를 다룬다.

### 문제 상황

#8에서 보류했던 개선 방향 후보 중 **Section-based chunking**을 구현하고, 실제로 검색 품질이 개선되는지 검증해야 했다.

### 원인 분석

#8에서 진단한 근본 원인(서로 다른 주제가 한 chunk에 섞여 임베딩이 희석됨)을 해결하려면, chunk 경계를 문서의 의미 단위(섹션)와 일치시켜야 한다는 방향이 이미 정해져 있었다.

### 재현 방법

(이 항목은 새로운 버그의 재현이 아니라, #8에서 발견한 문제가 Section-based chunking 적용 후에도 일부 질문 유형에서는 유사한 패턴으로 다시 나타나는지 확인하는 작업이다 — 아래 해결 과정에서 함께 다룬다.)

### 해결 과정

`chunk_by_section()` 함수를 새로 구현했다 — 마크다운 `##` 헤더 기준으로 섹션을 분리하고, 섹션이 `chunk_size`를 초과하면 그 섹션 내부에서만 `chunk_fixed_size()`를 재사용해 추가 분할하는 하이브리드 방식이다. 섹션 경계는 어떤 경우에도 넘지 않는다.

- (개선 내용) `tests/test_chunker.py`에 `TestChunkBySection` 클래스를 추가해 섹션 경계 보존, 내부 재분할, 예외 처리, 실제 데이터 회귀 테스트(`"8842"`와 `"Drift Score"`가 분리되었는지)를 검증했다 (11개 테스트 모두 통과).
  - (추가 수정) `debugs/debug_retrieval.py`를 Fixed-size와 Section-based 두 전략을 동일 조건(같은 질문, 같은 embedding 모델)으로 비교하도록 확장했다.
- (개선 내용) #8의 원래 문제였던 `"What is the default API port for NimbusFlow?"` 질문에서, "8842" chunk의 순위가 **3위(score=0.6546) → 1위(score=0.7740)**로 개선되었다. 같은 정보가 overlap으로 인해 2개 chunk(인덱스 6, 7)에 중복되어 있던 것도 1개 chunk로 정리되었다.
- (개선 내용) 같은 검증을 다른 질문(`"내부 코드네임은?"`)에도 적용한 결과, 해당 케이스에서도 순위와 점수가 함께 개선됨을 확인했다 (3위 0.6792 → 1위 0.7593).
- (한계점) 추가로 두 질문(`checkpoint_interval_sec`, `NF-227` 에러코드)을 검증한 결과, **두 전략 모두 1위는 정답을 유지했지만, Section-based의 점수 자체는 오히려 하락**했다 (각각 0.5044→0.4543, 0.6529→0.5082). 이 두 질문의 정답은 Configuration, Error Codes 섹션처럼 **여러 항목이 나열된 "목록형 섹션"**에 속해 있었다 — 섹션 전체를 하나의 chunk로 묶다 보니, 찾고자 하는 키워드 하나의 비중이 섹션 내 다른 항목들에 의해 상대적으로 희석된 것으로 보인다. 이는 #8에서 발견했던 "섹션 간 혼동" 문제와 같은 원리의 문제가, "섹션 내부의 개별 항목 간 혼동"이라는 더 작은 단위에서 다시 나타난 것이다. (추후 다른 트러블슈팅으로 개선 시도 예정)

### 배운 점

- 하나의 개선책(Section-based chunking)이 모든 질문 유형에 똑같이 효과적이지는 않다는 것을 여러 질문으로 검증하며 확인했다. "섹션 경계로 인한 주제 혼동"이라는 문제는 해결되었지만, "섹션 내부에 나열된 여러 항목 간의 혼동"이라는 같은 원리의 문제가 더 작은 스케일에서 반복될 수 있다는 것을 배웠다 — 청킹 전략의 개선은 특정 단위(섹션)에서의 해결일 뿐, 그 하위 단위에서 동일한 패턴의 문제가 다시 나타날 수 있다는 일반화된 통찰을 얻었다.
- 하나의 대표 사례(API 포트 질문)만으로 개선을 확정하지 않고, 성격이 다른 여러 질문(단일 정보형 vs 목록형 섹션)으로 검증한 것이 이 한계를 발견하는 데 핵심적이었다 — 표 5(문서 도메인 특수성)에서 강조했던 "검증 가능한 anchor를 충분히 마련해 둔다"는 설계가 이번에도 새로운 발견으로 이어졌다.

---

## #13. `src/main.py`의 경로 계산 실수 — 디렉토리 깊이가 다른 파일에 기존 코드를 그대로 복사

### 문제 상황

FastAPI 서버(`uvicorn src.main:app --reload`)를 처음 실행했을 때, 서버 시작 단계(lifespan)에서 다음 에러가 발생하며 애플리케이션이 시작되지 못했다.

```
FileNotFoundError: 파일을 찾을 수 없습니다: /Users/.../06_Weekly_Challenge/data/nimbusflow_manual.md
```

실제 데이터 파일은 `rag_project/data/nimbusflow_manual.md`에 정확히 존재했지만, 에러 메시지의 경로에는 `rag_project`가 한 단계 빠져 있었다.

### 원인 분석

`main.py`에서 데이터 파일 경로를 계산하는 코드는 다음과 같았다.

```python
data_path = Path(__file__).resolve().parent.parent.parent / "data" / "nimbusflow_manual.md"
```

이 `.parent.parent.parent` 패턴은 `chunker.py`, `embedder.py` 등 **`src/model/` 디렉토리(프로젝트 루트로부터 2단계 깊이)**에 있는 파일들의 `__main__` 블록에서 가져온 것이었다. 그런데 `main.py`는 **`src/` 바로 아래(프로젝트 루트로부터 1단계 깊이)**에 위치한 파일이라, 같은 횟수만큼 `.parent`를 적용하면 프로젝트 루트를 한 단계 지나쳐 그 상위 디렉토리를 가리키게 되었다.

근본 원인: 디렉토리 깊이가 다른 파일에 기존에 잘 동작했던 경로 계산 코드를 그대로 복사하면서, 새 파일의 위치 기준으로 `.parent` 횟수를 다시 계산하지 않았다.

### 재현 방법

1. 서로 다른 디렉토리 깊이에 있는 두 파일(예: `src/model/x.py`와 `src/main.py`)을 준비한다.
2. 한 파일에서 프로젝트 루트를 찾기 위해 사용한 `Path(__file__).resolve().parent...` 패턴을, `.parent` 개수를 조정하지 않고 다른 파일에 그대로 복사한다.
3. 복사한 파일을 실행하면, 의도한 디렉토리보다 한 단계 위(또는 아래) 디렉토리를 가리키게 되어 `FileNotFoundError`가 재현된다.

### 해결 과정

- (개선 내용) `main.py`의 위치(`src/` 바로 아래)에 맞춰 `.parent.parent.parent`를 `.parent.parent`로 수정했다. `main.py` 기준으로 `.parent`(→ `src/`) → `.parent`(→ 프로젝트 루트)이므로, 두 번만 적용해야 정확하다.
- (주의 사항) 이런 경로 계산 코드는 파일을 옮기거나 복사할 때마다 재검증이 필요한 부분이라, 추후 파일이 늘어나면 매번 `.parent` 횟수를 세는 대신 더 안정적인 방법(예: 프로젝트 루트를 가리키는 상수를 한 곳에서 정의해 재사용하는 방법)을 고려할 수 있다.

### 배운 점

- "다른 파일에서 잘 동작했던 코드"를 그대로 복사하는 것은, 그 코드가 **암묵적으로 가정하고 있던 조건**(이번 경우 "파일이 프로젝트 루트로부터 몇 단계 떨어져 있는가")까지 함께 복사한다는 것을 의미한다는 것을 다시 확인했다. 코드를 재사용할 때는 그 코드가 어떤 전제 위에서 동작했는지를 먼저 점검해야 한다.
- 에러 메시지에 출력된 실제 경로(`.../06_Weekly_Challenge/data/...`)를 자세히 읽고, 어느 디렉토리가 빠졌는지를 비교한 것이 원인을 빠르게 좁히는 데 효과적이었다 — 추상적으로 "경로가 잘못된 것 같다"고 추측하기보다, 에러가 보여주는 정확한 경로와 실제 의도한 경로를 나란히 놓고 비교하는 습관이 유효했다.

---

## #14. FastAPI 엔드포인트 테스트 생략 결정 — Generation 모듈과 동일한 논리

> 이 항목은 에러 트러블슈팅이 아니라, **테스트 작성 여부에 대한 설계 결정**을 기록한다.

### 문제 상황

`src/main.py`(`/query`, `/health` 엔드포인트)에 대해 `TestClient`를 사용한 자동화 테스트를 작성할지 결정해야 했다.

### 원인 분석

`/query`는 내부적으로 `TextGenerator`(Gemma 4 E2B-it)를 호출하므로, 이를 테스트하려면 테스트 실행마다 `lifespan` 이벤트가 동작하여 약 10GB 모델을 매번 로딩해야 한다. 또한 LLM이 생성하는 답변의 정확한 문구는 검증할 수 없고, "200 OK가 오는가", "JSON 스키마가 맞는가" 정도만 검증 가능하다 — 이는 Generation 단계에서 `test_generator.py` 작성을 생략하기로 했던 결정과 같은 논리이다.

`/health`는 모델 호출이 없어 가볍지만, `lifespan`이 서버 시작 시 전체 인덱싱과 모델 로딩을 수행하므로, `/health`만 테스트하려 해도 결국 무거운 시작 과정을 거쳐야 한다.

### 재현 방법

1. FastAPI 앱에서 `lifespan` 이벤트로 무거운 리소스(LLM, embedding 모델)를 로딩하도록 구성한다.
2. `TestClient`로 해당 앱의 엔드포인트를 호출하는 테스트를 작성한다.
3. 테스트를 실행하면, 매 테스트 세션마다 `lifespan`이 실행되어 모델 로딩 비용이 그대로 발생하는 것을 확인할 수 있다 — 엔드포인트 자체가 가벼워도 회피할 수 없다.

### 해결 과정

- (개선 내용) `/query`, `/health` 모두에 대해 `TestClient` 기반 자동화 테스트를 작성하지 않기로 결정했다. 대신 `curl`을 사용한 수동 검증으로 대체했다.
  - (추가 수정) `/query`는 실제 질문("내부 코드네임은?")으로 호출해 정답("Project Driftwood")과 근거 chunk, 점수가 정확히 반환되는지 확인했다.
  - (추가 수정) `/health`는 호출 결과 `{"status": "ok", "indexed_chunks": 20}`이 반환되어, Section-based chunking 결과(20개 chunk, 트러블슈팅 #12)와 일치함을 확인했다.
- (주의 사항) 이 결정은 "테스트가 불필요하다"는 뜻이 아니라, "모델 로딩 비용 대비 자동화 테스트가 주는 추가 검증 가치가 낮다"는 비용 대비 판단이다. 만약 추후 엔드포인트 로직이 복잡해지거나(예: 입력 검증 분기 추가), 모델 의존성 없이 테스트 가능한 부분이 늘어나면 재검토할 수 있다.

### 배운 점

- 테스트 작성 여부를 "테스트가 가능한가"만으로 판단하지 않고, "테스트 비용 대비 얻는 검증 가치가 충분한가"로 판단하는 기준을 Generation 모듈(트러블슈팅, `test_generator.py` 생략 결정)에 이어 FastAPI 계층에도 동일하게 적용했다. 같은 원리가 다른 계층(모듈 → API)에도 일관되게 적용될 수 있다는 것을 확인했다.

## #15. `scope="class"` fixture를 instance method로 정의할 때 발생하는 `PytestRemovedIn10Warning`

### 1. 문제 상황

`tests/test_evaluate_faithfulness.py`에서 `scope="class"` fixture를 다음과 같이 일반 인스턴스 메서드 형태로 작성했다.

```python
@pytest.fixture(scope="class")
def sample_data(self):
    ...
```

테스트를 실행하자 다음과 같은 경고가 발생했다:

```bash
textPytestRemovedIn10Warning: Class-scoped fixture defined as instance method is deprecated.
Instance attributes set in this fixture will NOT be visible to test methods,
as each test gets a new instance while the fixture runs only once per class.
Use @classmethod decorator and set attributes on cls instead.
```

처음에는 경고 메시지만 보고 대략적으로 @classmethod를 붙이면 될 것이라고 생각했으나, 왜 이런 경고가 발생하는지에 대한 근본적인 이해가 부족하다는 것을 느꼈다.

### 2. 원인 분석

2.1 경고 메시지 해석
경고 메시지의 핵심 내용은 다음과 같았다:

Class-scoped fixture를 instance method로 정의하는 것은 더 이상 권장되지 않는다(deprecated).
fixture 내부에서 self에 값을 저장해도, 실제 테스트 메서드에서는 해당 값을 볼 수 없다.
그 이유는 fixture는 클래스당 한 번만 실행되지만, 각 테스트는 새로운 인스턴스를 받기 때문이다.

이 문장을 통해 단순한 문법 문제가 아니라, pytest의 fixture 생명주기와 관련된 구조적인 문제라는 것을 인지했다.
2.2 scope="class"의 의미와 self 사용 시 발생하는 문제
scope="class"는 해당 클래스 내 모든 테스트가 fixture를 한 번만 실행하고 공유하겠다는 의미를 담고 있다.
여기서 핵심적인 의문이 생겼다:
“fixture는 클래스당 한 번만 실행되는데, pytest는 왜 각 테스트마다 새로운 인스턴스를 생성하는가?
그렇다면 fixture 안에서 self.xxx = 값으로 저장한 것은, 실제 테스트 실행 시점의 self와 다른 객체가 되는 것 아닌가?”
이 질문을 해결하기 위해 self와 cls의 차이를 정리했다.

일반 메서드에서 self는 메서드를 호출한 인스턴스를 가리킨다.
@classmethod에서 cls는 클래스 자체를 가리킨다.

scope="class" fixture의 목적은 클래스 전체에서 데이터를 공유하는 것이었는데, self를 사용하면 fixture 실행 시점과 테스트 실행 시점의 인스턴스가 달라서 값이 공유되지 않는 구조였다. 이것이 경고가 발생하는 근본 원인이었다.
2.3 @classmethod를 선택한 이유
위 문제를 해결하기 위해 @classmethod 데코레이터를 적용했다.
@classmethod를 사용하면 메서드의 첫 번째 인자가 인스턴스가 아닌 클래스 자체로 전달되므로, cls.xxx = 값 형태로 클래스 레벨에 속성을 저장할 수 있다. 이는 scope="class"의 목적과 정확히 일치했다.
또한 pytest 공식 문서에서도 scope="class" fixture를 정의할 때는 @classmethod 사용을 권장하고 있었다.

### 3. 해결 과정

최종적으로 fixture를 아래와 같이 수정했다:

```python
Python@pytest.fixture(scope="class")
@classmethod
def sample_data(cls):
    ...
```

수정 후 PytestRemovedIn10Warning은 사라졌으며, 모든 테스트가 정상적으로 통과했다.

### 4. 배운 점

이번 경험을 통해 단순히 경고 메시지를 보고 대응하는 것이 아니라, 경고가 발생하는 근본 원리를 파고드는 것의 중요성을 다시 확인했다.
특히 다음과 같은 개념을 더 명확하게 정리할 수 있었다:

scope="class" fixture는 클래스당 한 번 실행되지만, 각 테스트는 별도의 인스턴스를 받기 때문에 self로는 값을 공유할 수 없다.
@classmethod를 통해 cls를 사용하면 클래스 레벨에 값을 저장할 수 있으며, 이는 scope="class"의 목적과 잘 맞는다.
데코레이터(@)는 함수를 감싸는 고차함수의 한 형태이며, classmethod는 메서드의 첫 번째 인자를 변경하는 역할을 한다.

추가로 깊게 공부하고 싶은 주제
이번 이슈를 해결하면서 자연스럽게 생긴 추가 질문들은 다음과 같다. 당장 Level 0 진행에 필수적인 내용은 아니었기 때문에, 나중에 필요할 때 별도로 학습하기로 했다.
| 주제 | 주요 포인트 | 학습 우선순위 |
| --- | --- | --- |
| 고차함수와 클로저 | 데코레이터의 내부 동작 원리 | 중 |
| Python Descriptor 프로토콜 | `__get__`, `__set__`, Property | 하 |
| pytest 고급 사용법 | fixture 실행 순서, Dependency Injection | 중 |
| Java AOP와 Python 데코레이터 | 개념 비교 | 하 |

## #16. Context Precision (키워드 기반 최소 버전)의 한계

### 1. 문제 상황

Level 0에서 Context Precision을 가장 단순한 키워드 겹침 방식으로 구현했다.

- 질문과 context를 단어 단위로 분리한 후, 겹침 비율이 threshold=0.3 이상이면 관련 있다고 판단
- LLM을 전혀 사용하지 않고 순수한 문자열 처리만으로 평가

테스트는 모두 통과했지만, 실제 실행 결과에서 명확한 한계가 드러났다.
**실행 예시 결과:**

```text
[Context Precision Score] 0.5
[Threshold] 0.3

Chunk 0: overlap=0.50 → 관련 있음   (정상)
Chunk 3: overlap=0.38 → 관련 있음   ← 문제
```

**문제 사례:**

- Chunk 3 ("The default checkpoint interval is 90 seconds.")
  - default라는 단어 하나 때문에 overlap=0.38이 나와서 관련 있음으로 판단됨
  - 하지만 실제로는 질문("What is the default API port for NimbusFlow?")과 거의 관련이 없는 내용

### 2. 원인 분석

이 현상은 **키워드 겹침 기반 방식의 본질적인 한계**에서 비롯된다.

### 3. 주요 한계점

**주요 한계점 표**
| 한계 | 설명 | 이번 사례에서의 영향 |
| --- | --- | --- |
| **의미 이해 불가능** | 단어의 의미나 맥락을 전혀 고려하지 않음 | default라는 단어만 겹쳐도 관련 있다고 판단 |
| **중요 키워드와 일반 단어 구분 불가** | 질문에서 핵심적인 단어(API, port)와 일반적인 단어(default, the)를 동일하게 취급 | - |
| **동의어·유의어 처리 불가** | port와 포트, default와 기본값 등을 같은 의미로 인식하지 못함 | - |
| **문맥(Context) 무시** | 단어 순서나 문장 전체의 의미를 고려하지 않음 | - |
| **우연한 겹침에 취약** | 질문과 무관한 문서에서도 우연히 단어가 겹칠 경우 오판 가능 | Chunk 3이 대표적인 예시 |

이 방식은 **"단어가 겹치면 관련이 있을 것이다"**라는 매우 단순한 가정에 기반하고 있기 때문에, 의미적 관련성과 실제 관련성을 구분하지 못하는 구조적 한계를 가지고 있다.

### 4. 배운 점

키워드 기반 최소 평가 단계를 통해 다음과 같은 점을 명확히 이해할 수 있었다:

- **가장 단순한 방법**은 구현이 쉽고 빠르지만, **의미를 이해하지 못한다는 치명적인 한계**를 가진다.
- Context Precision을 제대로 평가하기 위해서는 단순한 문자열 겹침을 넘어선 **의미적 판단**이 필요하다는 것을 체감했다.
- 이후에는 이 한계를 보완할 수 있는 방향(Embedding 기반 또는 LLM 기반)으로 확장해야 한다는 방향성이 명확해졌다.

**개선 방향**

- Embedding을 활용한 의미적 유사도 기반 판단
- LLM을 활용한 관련성 판단 (LLM-as-a-Judge)
- 질문에서 중요한 키워드를 가중치 있게 처리하는 방식 (예: BM25)

## #17. Answer Relevancy (키워드 기반 최소 버전)의 한계

### 문제 상황

Level 0에서 **Answer Relevancy**를 가장 단순한 키워드 겹침 기반으로 구현했다.

- 생성된 답변과 질문을 단어 단위로 분리한 후, 겹침 비율을 계산
- threshold=0.3 이상이면 관련 있다고 판단
- 답변 **전체를 하나의 단위**로 평가

테스트는 모두 통과했지만, 이 방식은 Context Precision과 동일한 본질적인 한계를 가지고 있다.

### 원인 분석

### 주요 한계점

**주요 한계점 표**
| 한계 | 설명 | 구체적인 영향 |
| --- | --- | --- |
| **의미 이해 불가능** | 단어의 의미나 맥락을 전혀 고려하지 않음 | 의미적으로 관련이 없더라도 단어가 겹치면 점수가 높게 나올 수 있음 |
| **동의어·유의어 처리 불가** | port와 포트, default와 기본값 등을 구분하지 못함 | - |
| **중요 키워드와 일반 단어 구분 불가** | 질문에서 핵심적인 단어와 일반적인 단어를 동일하게 취급 | the, is, for 같은 단어가 점수에 영향을 줌 |
| **답변 길이에 취약** | 답변이 길수록 우연히 단어가 겹칠 확률이 증가 | 긴 답변일수록 점수가 과대평가될 가능성 존재 |
| **문맥 무시** | 단어 순서나 문장 구조를 고려하지 않음 | - |

### 대표적인 문제 사례

Context Precision에서 이미 확인했던 현상과 유사하게, **의미 없는 단어(default, the 등) 하나만 겹쳐도 점수가 올라가는 문제**가 그대로 존재한다.

이 방식은 결국 **"단어가 겹치면 관련이 있을 것이다"**라는 매우 단순한 가정에 기반하고 있기 때문에, 의미적 관련성과 실제 관련성을 구분하지 못하는 구조적 한계를 가지고 있다.

### 배운 점

키워드 기반 Answer Relevancy를 구현하면서 다음과 같은 점을 확인했다:

- **Faithfulness**와 **Answer Relevancy**는 모두 생성 품질을 평가하는 지표지만, 평가 관점이 다르다.
  - Faithfulness: 답변이 context에 충실한가?
  - Answer Relevancy: 답변이 질문과 관련 있는가?
- 키워드 기반 방식은 구현이 매우 간단하지만, **의미를 이해하지 못한다는 치명적인 한계**를 가지고 있다.
- Answer Relevancy를 제대로 평가하기 위해서는 단순한 키워드 겹침을 넘어선 **의미적 판단**이 필요하다는 것을 다시 체감했다.

**개선 방향**

- Embedding을 활용한 의미적 유사도 기반 판단
- LLM을 활용한 관련성 판단 (LLM-as-a-Judge)
- 질문에서 중요한 키워드를 가중치 있게 처리하는 방식 (예: TF-IDF, BM25)

## #18. Context Recall (키워드 기반 최소 버전)의 한계

### 문제 상황

Level 0에서 **Context Recall**을 가장 단순한 키워드 기반으로 구현했다.

- Ground Truth를 **문장 형태의 리스트**로 정의
- 검색된 context를 하나로 합쳐서 키워드 기반으로 검색
- 각 Ground Truth 문장이 포함되어 있는지 판단하여 점수 계산

테스트는 모두 통과했지만, 이 방식은 기존 지표들(Context Precision, Answer Relevancy)과 유사한 본질적인 한계를 가지고 있으며, 추가로 Ground Truth 정의와 관련된 한계도 존재한다.

### 원인 분석

### 주요 한계점

**주요 한계점 표**
| 한계 | 설명 | 구체적인 영향 |
| --- | --- | --- |
| **의미 이해 불가능** | 단어의 의미나 맥락을 전혀 고려하지 않음 | 표현이 다르면 같은 내용이라도 놓칠 수 있음 |
| **동의어·유의어 처리 불가** | port와 포트, engine과 엔진 등을 구분하지 못함 | - |
| **Ground Truth 정의의 어려움** | 정답에 필요한 정보를 어떻게 정의하느냐에 따라 점수가 크게 달라짐 | 평가자의 주관이 크게 개입될 수 있음 |
| **검색된 context 합침으로 인한 정보 손실** | 여러 chunk에 정보가 분산되어 있을 때 놓칠 가능성 | 실제로는 검색됐는데 포함되지 않은 것으로 판단될 수 있음 |
| **키워드 기반의 근본적 한계** | 단순 단어 겹침만으로 판단 | 의미적 관련성을 반영하지 못함 |

### 대표적인 문제 사례

- Ground Truth를 "NimbusFlow는 데이터 파이프라인 엔진이다"라고 정의했을 때, context에 "lightweight data pipeline orchestration engine"이라고 표현되어 있으면 키워드 겹침이 부족해 미포함으로 판단될 수 있음.
- 검색된 context가 여러 개로 나뉘어 있을 때, 정보를 하나로 합치면서 문맥이 흐려지거나 중요한 정보가 희석될 수 있음.

이 방식은 결국 **"키워드가 겹치면 포함된 것이다"**라는 단순한 가정에 기반하고 있기 때문에, 의미적 이해와 Ground Truth 정의의 품질에 크게 의존하는 구조적 한계를 가지고 있다.

### 배운 점

Context Recall을 구현하면서 다음과 같은 점을 확인했다:

- Context Recall은 다른 세 가지 지표와 달리 **Ground Truth(정답)**가 반드시 필요하다는 점에서 근본적으로 다르다.
- 키워드 기반 방식은 구현이 간단하지만, **의미를 이해하지 못한다는 치명적인 한계**를 그대로 가지고 있다.
- Ground Truth를 어떻게 정의하느냐가 평가 결과에 큰 영향을 미치기 때문에, 실제 사용 시 Ground Truth의 품질 관리가 중요하다는 것을 깨달았다.

**개선 방향**

- Embedding을 활용한 의미적 유사도 기반 판단 (키워드 겹침 → 의미 유사도)
- LLM을 활용한 관련성 판단 (LLM-as-a-Judge)
- 각 chunk를 개별적으로 평가한 후 결과를 종합하는 방식 (context를 하나로 합치는 대신)
- Ground Truth를 자동으로 생성하거나, 더 체계적으로 관리하는 방식
