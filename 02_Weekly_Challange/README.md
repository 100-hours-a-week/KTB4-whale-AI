# 2 Week Weekly Challange

## Community Web Page Server Developing

- [커뮤니티 사이트](https://edu-community.adapterz.kr) 백엔드 서버 구현
- FastAPI로 구현

## 회고록

- [2주차 회고록 바로가기](./RETROSPECTIVE.md)

## 설치 방법

```bash
# 1. Repository 클론 및 이동
git clone https://github.com/100-hours-a-week/KTB4-whale-AI.git
cd 01_Weekly_Challange
```

```python
# 2. 설치
python3 -m venv .venv # 가상환경 생성
source .venv/bin/activate # 가상환경 활성화
pip install "fastapi[standard]" # 라이브러리 설치
```

```python
# 3. 실행
fastapi dev
```

### 사용 예시

- "[스웨거 페이지](http://127.0.0.1:8000/docs)" 진입
- 개별 API 사용

## 기술 스택

- FastAPI
