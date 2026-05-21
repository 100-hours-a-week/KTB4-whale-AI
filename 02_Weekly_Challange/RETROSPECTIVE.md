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

### DB는 왜 연결해야 될까

- (현상) 지금은 메모리와 JSON 방식으로 데이터가 저장되어 있음
- (문제) 서버를 껐다가 다시 켜면 모든 데이터가 사라진다.
- (해결) DB를 연결하면 데이터가 파일(또는 서버)에 영구적으로 저장된다.
- (장점) 추후 사용자 수, 게시글 수가 많아져도 안정적으로 관리할 수 있다.

- 강사님 추천 방식
  - MySQL, SQLModel(ORM), MongoDB
    - MySQL: RDBMS, 표 형태로 데이터를 구조화해서 저장
      - (장점) 안정적이고, 규모가 커져도 잘 버팀
      - (단점) 초보자가 설치하고 설정하는 과정에서 러닝 커브
    - SQLModel: FastAPI 공식 ORM, FastAPI 개발자가 만든 라이브러리
      - (장점) FastAPI 공식 라이브러리 Pydantic 모델을 거의 그대로 DB 테이블로 사용, 낮은 러닝 커브
      - (장점) 내부적으로 MySQL(SQLite, PostgreSQL) 사용 가능, 높은 호환성
      - (장점) 현재 프로젝트에서 가장 최적화
      - (단점) 실무에서 깊이
    - MongoDB: NoSQL 문서형 데이터베이스
      - 데이터를 JSON 형태 통째로 저장 (관계형 DB처럼 표로 나누지 않음)
      - (장점) 유연하고, 빠르게 개발 가능
      - (단점) 관계가 명확한 프로젝트에서는 오히려 불편할 수 있음

- 현재까지 내가 이해한 내용
  - 컴퓨터는 CPU, Memory, SSD, I/O Device로 구성된다.
  - 기존에 변수를 할당하는 방식은 Memory에 저장하므로, 프로그램이 종료되면(서버가 꺼지면, 컴퓨터가 꺼지면) RAM 내용은 사라진다.
  - 문제를 해결하기 위해 DBMS(소프트웨어)는 SSD 파일에 데이터를 영구 저장하여, 프로그램이 종료되어도 데이터가 사라지지 않는다.
    - DBMS는 SSD의 파일을 체계적으로 읽고 쓰는 프로그램(엔진)이다.
    - 이 엔진이 RAM과 SSD 사이에서 데이터를 오가며 동작한다.
  - DBMS로 DB를 설계, 생성, 조작(관리)하는 방법은 크게 3가지가 있다.
    - RDB(직접 SQL)
    - ORM(추상화)
    - NoSQL(다른 패러다임)
  - 진입 장벽이 낮은 ORM(SQLModel)부터, MySQL + SQLAlchemy, (Raw SQL) MySQL, (NoSQL) MongoDB 순으로 학습 예정
    - 러닝 커브가 낮다면, 그만큼 구체적인 내용이 생략되어 있음을 의미한다.
    - 따라서, 순서대로 학습하면서 어떤 부분이 생략되어 있는지, 차이점을 이해하는 것을 목표로 한다.

  - (한계) 하드웨어에서 시스템 버스를 통해 CPU에 접근하므로, Memory에서 접근하는 것보다 속도가 느리다.
    - 그래서 **DB 설계, 캐싱, 인덱스** 등이 매우 중요하다.

- DB의 실제 구성
  1. 나의 코드 실행
     - FastAPI가 요청 받음
     - DB 드라이버가 호출됨
  2. DB 드라이버는 DBMS 엔진으로 명령 전달 (TCP 소켓으로)
     - DBMS 내부 - 메모리(RAM) 중심
       - Query Parser: 문장을 파싱(해석)
       - Query Optimizer: 가장 빠른 실행 계획 수립 (인덱스 등)
       - Buffer Pool: RAM 캐시에서 해당 데이터가 이미 있는지 먼저 확인
         - 있으면, SSD 안 건드리고 바로 처리
         - 없으면, SSD에서 파일을 읽어서 RAM으로 올림
       - Transaction Manager: ACID 보장 (트랜잭션 시작)
  3. Storage Engine (실제 저장 담당)
     - Write-Ahead Log (WAL) 파일에 먼저 기록 (장애 시 복구용, 가장 먼저 SSD에 씀)
     - 실제 테이블 파일(.ibd, .wt 등)에 데이터 기록
  4. 성공하면 결과 반환
     - 파이썬 코드로 돌아옴
     - .id 같은 값이 채원진 객체를 받음.
  - 용어 정리
    - DB 드라이버: DB마다 사용하는 다른 프로토콜(통신 규칙)을 프로그래밍 언어가 이해할 수 있는 형태로 번역하는 기능
      - 라이브러리로 설치해서 사용
    - 커넥션(Connection): 프로그램과 DB 서버 사이에 맺은 통신 채널
      - 한 번 커넥션을 맺으면 그 채널을 통해 여러 번 쿼리를 주고 받을 수 있다.
      - 커넥션을 만들 때는 TCP 소켓을 통해 실제 네트워크 연결(또는 로컬 소켓)이 발생
      - (실무) 커넥션을 만드는 비용은 매우 비싸므로, Connection Pool로 미리 여러 커넥션을 만들어 놓고 재사용한다.
      - SQLModel, SQLAlchemy, pymongo 모두 내부적으로 연결 풀을 관리한다.

## 260521 - 리팩토링

### 왜 리팩토링이 필요한가?

- 안 나누면, 스파게티 코드가 발생할 수 있다.

### 어떻게 나누는 게 좋을까?

- 소프트웨어 공학의 중요한 원칙에 의거
  - Sparation of Concerns (관심사 분리)
  - 무엇에 관심에 있는지에 따라 분리한다.

- MVC vs Simple Moduler 구조 비교
  - MVC: 강사 추천
    - 전통적인 웹 프레임워크(Django, Spring, ASP.NET)에서 가장 많이 쓰이는 고전적인 패턴
    - FastAPI에서 REST API 위주이기 때문에 View 부분이 없음. (JSON만 반환)
    - 확장성: Controller가 점점 커질 위험이 있음.
  - Simple Moduler: AI 추천
    - MVC로 강하게 구분되는 반면, 4~5 폴더로 유연하게 분리
    - (FastAPI와의 적합성) FastAPI 공식 문서와 커뮤니티에서 가장 많이 사용하는 방식
    - (확장성) 기능별로 파일을 더 세밀하게 나눌 수 있음.
    - crud 폴더로 DB 로직 중복을 쉽게 방지
  - 결과: 더 유연하고 유지보수가 편리하므로, Simple Moduler 적용
  - 추후: MVC 패턴 적용 예정

- MVC 패턴
  - Route(Views)
    - (관심사) 요청을 어디로 보낼까?
    - (이해) URL + HTTP 메서드만 다루고 싶다.
  - Controller:
    - (관심사) 요청을 어떻게 처리할까?
    - (이해) HTTP 처리 + 비즈니스 로직 + DB 호출
  - Model
    - (관심사) 데이터는 어떻게 저장/조회할까?
    - (이해) DB 스키마 + 쿼리 + 데이터 유효성
