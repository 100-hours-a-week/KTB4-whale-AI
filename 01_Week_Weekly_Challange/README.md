# 1주차 프로젝트

## Auto File Organizer ('afo')

원하는 폴더를 자동으로 정리해주는 Python CLI 프로그램

## 회고록

- [회고록 바로가기](./RETROSPECTIVE.md)

## 기능

- `afo stats`: 확장자별 파일 개수, 비율, 이동될 폴더 분석
- `afo rename`: `{num:02d}_{title}_{date}` 패턴으로 파일명 정리 (**VirusTotal async** 스캔 포함, 숨김 파일 자동 스킵)
- `afo move`: 확장자에 따라 Images, Documents, Archives, Videos, Audio, Others 폴더로 자동 이동 (폴더 자동 생성)
- `afo organize`: 파일명 변경 + 폴더 이동을 한 번에 (**VirusTotal async** 스캔 포함)

## 설치 방법

```bash
# 1. Repository 클론 및 이동
git clone https://github.com/100-hours-a-week/KTB4-whale-AI.git
cd auto-file-organizer
```

```python
# 2. 설치
python3 -m venv .venv # 가상환경 생성
source .venv/bin/activate # 가상환경 활성화
pip install -e . # 라이브러리 설치
```

### 사용 예시

```python
# 1. 폴더 분석
afo stats # 현재 폴더 분석
afo stats ~/Downloads # Downloads 폴더 분석
```

```python
# 2. 파일명 변경
afo rename ~/Downloads
afo rename ~/Downloads --dry-run # 미리 보기
afo rename ~/Downloads --scan # 바이러스 검사
```

```python
# 3. 파일 이동
afo move ~/Downloads
afo move ~/Downloads --dry-run # 미리 보기
```

```python
# 4. 파일 정리 (파일명 변경 + 파일 이동)
afo organize ~/Downloads
afo organize ~/Downloads --dry-run # 미리 보기
afo organize ~/Downloads --scan # 바이러스 검사
```

### 주요 옵션

- `--dry-run` / `-n`: 실제 변경 없이 미리보기
- `--backup`: 원본을 ``폴더에 복사
- `--yes` / `-y`: 확인 창 없이 바로 실행
- `--scan`: Virus 검사(스캔) (.env 파일에 Virus Total API Key 선 추가 필요)

### 지원 확장자

- [rules.py 파일](/afo/core/rules.py) Images, Documents, Archives, Videos, Audio, Others

## 기술 스택

- Python 3.11+
- Typer (CLI)
- Rich (출력 스타일)
- pathlib (파일 처리)
