# 05 Chatbot - 단계 1 (Dummy)

## 현재 단계

- **단계 1**: FastAPI 기본 구조 + Dummy Generator 완료
- **단계 2**: 반복 호출 방식의 간이 생성기 완료 (규칙 기반)
- **단계 3**: PyTorch 기반 실제 모델 적용 (진행 예정)

## 프로젝트 구조

```bash
05_Chat_Bot/
├── main.py              # FastAPI 앱
├── generator.py         # 생성 로직 (단계 2 기준)
├── schemas.py           # Pydantic Request/Response
├── requirements.txt
├── README.md
└── test_generator.py    # generator 테스트용 (선택)
```

## 실행 방법

```bash
python -m venv venv
source .venv/bin/activate # 1. 가상환경 활성화
pip install -r requirements.txt # 2. 의존성 설치
uvicorn main:app --reload --port 8000 # 3. 서버 실행
python test_generator.py # 4. 테스트 방법
```
