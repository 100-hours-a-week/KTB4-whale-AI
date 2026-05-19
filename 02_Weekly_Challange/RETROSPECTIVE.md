# 커뮤니티 사이트 백엔드 서버 구현

## 변수명 명명 규칙

- camelCase, snake_case 중 어떤 것을 사용할까? snake_case
  - [함수, 변수명 가이드](https://peps.python.org/pep-0008/#function-and-variable-names)

## response_model

- 단순히 소스 코드 작성 시, 타입 추론하기 위한 내용으로 이해함
- 데코레이터에 쓰는 방식으로 FastAPI 공식 추천 방식
- 실제 실무에서 사용하는 Best Practice
- response_model이 있으면 타입 명시하지 않아도 타입 추론 가능

## global

- 선언한 변수가 global 변수인지 확인하기 위함
- 전역 변수를 지역에서 읽는 경우는 global 없이 가능
- 전역 변수를 지역에서 수정할 경우는 global 사용 필요

## raise

- Python에서 에러(예외)를 의도적으로 발생시키는 키워드
- 에러 관리에서 사용함.
