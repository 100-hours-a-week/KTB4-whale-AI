# HTTP

- HyperText Transfer Protocol
- 일반적인 텍스트를 뛰어넘는 "구조화된 텍스트"를 전송하기 위한 통신 규약
  - "구조화된 텍스트" 대표적인 사례: html, css, js, png, jpeg 등
- 네트워크 위에서 데이터를 전송받기 위한 규약, 규칙 중 하나

## HTTP는 왜 사용하는가?

- 웹 클라이언트와 웹 서버가 정의된 규칙에 따라 하이퍼텍스트 문서를 전송하기 위해
  - 규칙 없이 통신해도 되지만, 다양한 컴퓨터끼리 소통할 때마다 새로운 통신 규칙을 사용해야 하는 불편함
  - 웹 상에서 정보를 공유하기 위해 표준 방법의 필요성 대두
  - 현재는 하이퍼텍스트 문서를 전송할 때 정의된 규칙 HTTP를 기본으로 개발할 때 사용

## HTTP는 무엇을 전송하는가?

- HTTP Message를 전송한다.
  - HTTP Message는 Start Line(시작줄) + Header + (빈 줄) + Body로 구성된다.

1. Start Line: `GET / HTTP/1.1`

2. HTTP Header: 통신에 필요한 **메타데이터**로 구성, **서버 또는 미들웨어**에서 요청 처리 전에 필요한 정보를 제공하는 역할
   - Request/Response Header, General Header, Representation(Entity) Header로 구성
     - Request/Response: 요청/응답을 보낼 때 전송되는 헤더, 요청의 성격과 목적을 설명하고 부가적인 정보 제공
     - General: 요청, 응답 모두에서 사용되는 일반적인 헤더, 메시지 전체에 대한 정보 제공, 캐시, 연결 유지 등 관련 설정 포함
     - Representation: 응답 메시지의 본문 데이터 관련 헤더, 데이터의 형식, 언어, 압축 등을 나타냄.

   - 예시
     - Request 기준
       - Request: Host, User-Agent, Accept, Accept-Language, Accept-Encoding
       - General: Connection, Upgrade-Insecure-Requests
       - Representation: Content-Type, Content-Length, Content-Encoding, Expires
         - 클라이언트가 올바르게 해석하고 처리할 수 있도록 도움.

     - Response 기준
       - Response: Access-Control, Etag, Server, Set-Cookie, Vary, X-Frame-Options
       - General: Connection, Date, Keep-Alive, Transfer-Encoding
       - Representation: Content-Type, Content-Encoding, Last-Modified

3. HTTP Body: HTML 문서, JSON 데이터 등
   - 데이터, 문서 등 실제 전송할 내용

## HTTP는 어떻게 전송하는가? (FastAPI 환경 기준)

- HTTP는 클라이언트가 **URL**을 통해 특정 웹 주소로 요청을 보내는 방식이다.
  - URL: Uniform Resource Locator의 약자, 리소스의 위치와 종류를 나타내는 **고유**한 주소
  - 사용 이유: 웹 상의 리소스를 식별하고 접근하기 위해
  - URL 구성: `Scheme(스킴)://Domain:Port/Resource Path + Parameter`
    - Parameter 자리에는 Query String, Path Variable 사용 가능

- HTTP는 **Request, Response 방식**으로 전송한다.

1. Request
   - GET, POST, PUT, PATCH, DELETE 등의 메서드를 명시하여 요청
   - FastAPI에서 Request Body 검증하는 방법
     - Pydantic Model 사용
       - Pydantic Model은 DTO 역할을 한다.
         - DTO(Data Tranfer Object): Request Body의 데이터 구조와 타입을 정의
         - 요청으로 전달된 데이터를 해당 모델로 변환하는 과정에서 자동으로 유효성 검사 수행
         - Request Body의 구조와 규칙을 모델로 선언하면, FastAPI는 이를 기반으로 자동 검증 수행 (**라우터 함수 실행 전**에)
           - 코드 예시:

             ```python
             class PostCreate(BaseModel):
             title: str
             content: str

             @router.post("/posts")
             def create_post(payload: PostCreate):
             ```

             - 이렇게 선언하면 FastAPI는 라우터 함수 실행되기 전에 자동으로 아래 내용 검증
               - Request Body 존재 여부
               - 필수 필드(title, content) 누락 여부
               - 각 필드의 타입 일치 여부
             - 검증 실패할 경우, 라우터 함수 실행되지 않고, **422 Unprocessable Entity 응답을 반환**함.

2. Response
   - 요청 받은 내용을 처리하고 Status Code를 HTTP Header에 담아 클라이언트에게 응답을 보냄
   - 100 ~ 500번대 Status Code 구성
     - 100번대: 정보 메시지
     - 200번대: 성공
     - 300번대: 리다이렉션
     - 400번대: 클라이언트 오류
     - 500번대: 서버 오류

- 파이썬에서는
  1. (개발 환경 용도) FastAPI의 Uvicorn으로 사용
  2. (운영 환경 용도) Uvicorn에 Gunicorn 추가 (리눅스/MacOS 환경 O, Window 환경 X)
  - 설치: `pip install gunicorn`
  - 실행: `gunicorn [옵션] 모듈: 앱`

- 개발 및 실행 환경 세팅
  - `pyproject.toml`: 빌드, 패키징, 의존성, 도구 설정을 통합하고 표준화하여 개발 워크플로우를 개선할 수 있다.
  - `requirements.txt`: 파이썬 프로젝트에 필요한 외부 라이브러리와 그 버전을 기록해두는 텍스트 파일
  - Poetry: (개선된 버전, 기존에는 pyproject.toml + requirements.txt) 파이썬 프로젝트용 패키지, 의존성, 빌드 관리 도구

## 관련 추가 학습 개념

1. REST API
   - REST: 자원을 식별하고, 자원의 상태 정보를 구조화한 형식으로 서버와 클라이언트 간 정보를 교환하는 방식
   - API: 응용 프로그램 인터페이스, 운영체제와 응용 프로그램 사이의 통신에 사용되는 메시지 형식
