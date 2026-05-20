# 커뮤니티 사이트 백엔드 서버 구현

## 260518 - HTTP REST API 설계 및 구현

### 변수명 명명 규칙

- camelCase, snake_case 중 어떤 것을 사용할까? snake_case
  - [함수, 변수명 가이드](https://peps.python.org/pep-0008/#function-and-variable-names)

### response_model

- 단순히 소스 코드 작성 시, 타입 추론하기 위한 내용으로 이해함
- 데코레이터에 쓰는 방식으로 FastAPI 공식 추천 방식
- 실제 실무에서 사용하는 Best Practice
- response_model이 있으면 타입 명시하지 않아도 타입 추론 가능

### global

- 선언한 변수가 global 변수인지 확인하기 위함
- 전역 변수를 지역에서 읽는 경우는 global 없이 가능
- 전역 변수를 지역에서 수정할 경우는 global 사용 필요

### raise

- Python에서 에러(예외)를 의도적으로 발생시키는 키워드
- 에러 관리에서 사용함.

## 260519 - AI 모델 서빙

### 파이썬 문법 뜯어보기

```python
post_text = "\n\n".join(
  [f"제목: {p.title}\n내용:{p.content}" for p in posts]
)
```

- `[... for p in posts]`
  - 리스트 컴프리헨션 (list comprehension)
  - posts 리스트에 있는 객체를 하나씩 p라는 변수로 꺼냄.
  - f-string으로 새로운 리스트 생성
- `f"제목: {p.title}\n내용:{p.content}"`
  - f-string (포맷 문자열)
  - `{p.title}`: 해당 게시글의 제목
  - `\n`: 줄바꿈
  - `{p.content}`: 해당 게시글의 내용
- `"\n\n".join(...)`
  - join() 메서드
  - 각 문자열을 두 줄씩 띄워서 하나로 합침
- (소결) 게시글을 AI에게 보내기 좋은 형태의 텍스트로 변환

### item['title'] VS item.title의 차이

- `item['title']`
  - 데이터 타입: dict
- `itme.title`
  - 데이터 타입: PostRead (Pydnatic 모델 객체)

### `if not 리스트` 조건문의 반영 범위

- 해당 리스트가 빈 리스트 유무 분기 처리 방식
- 동일한 조건문
  - `if len(list) == 0`
  - `if len(list) < 1`
  - `if not len(list)`
- len() 방식과 비교
  - `not list`
    - 의미: 리스트가 비었는가
    - 속도: 가장 빠름
    - 가독성: Pythonic(추천)
  - `len(list) == 0`
    - 의미: 리스트 길이가 0인가
    - 속도: 조금 느림
    - 가독성: 명시적
- (한계) `if not`이라고 해서 None인지 유무까지 검사하지 않음
- (소결) 동작은 100% 동일하지만, Pythonic 추천
  - (간결함) 더 짧고 깔끔함
  - (범용성) 리스트 외 빈 문자열, 빈 딕셔너리, 0, False 등 "빈/거짓"인 모든 곳에 같은 방식으로 체크 가능

### Chat() 함수의 응답값 타입 오류

- (현상) response.message.content의 타입이 `str` | `None`으로 None일 수도 있다는 타입 오류 발생
- (원인 1) ollama 라이브러리가 내부적으로 `content: Optional[str]`으로 타입 힌트를 줌
- (원인 2) 반면, 내가 만든 SummaryRead 모델은 str로 None을 허용하지 않기 때문
- (해결) 반환되는 content의 값이 None일 경우, "생성 불가" 내용 반환 필요
  - `content: str = "요약 생성 불가"`: 클래스 타입에 기본값 인자 추가
  - `SummaryRead(summary=summarized) if summarized is not None else SummaryRead()`: Comprehension 사용
- (한계) 기획에 따라 HTTP Error가 필요할 수 있으므로, 서비스 기획 상 적절한 해결 방안인지 먼저 논의 필요

### ID counter의 초기화 문제

- (현상) 댓글, 게시글이 있다가 모두 지워졌을 때, ID counter를 초기화하는 로직 존재
- (원인) 데이터가 지워졌다면 ID counter도 함께 초기화가 필요할 것으로 예상
- (예상 문제) 초기화 불필요
  - (가정) 댓글이 10개 있는데 모두 지워진 상태, ID counter도 초기화
  - (문제 1) 추후 10번 댓글 복구, 삭제된 댓글 다시 보관하는 기능(soft delete)를 추가할 경우 ID 충돌 발생
  - (문제 2) 클라이언트(프론트엔드)가 10번 댓글을 캐싱하거나 공유 링크로 가지고 있을 가능성, 이 때 재사용하게 되면 다른 댓글 출력됨
- (실무) 실제 DB(PostgreSQL, MySQL, SQLite 등)에서는 삭제 시, 반드시 auto-increment를 유지함. (항상 증가)
  - (판단) 리스트가 비었을 때 counter를 초기화하는 것은 임시적 편의일 뿐, **장기적으로는 기술 부채**

## 260520 - 데이터베이스 적용하기
