# 1주차 프로젝트

## Auto File Organizer ('afo')

원하는 폴더를 자동으로 정리해주는 Python CLI 프로그램

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

### Typer 라이브러리

1. Typer 선택 이유
   - VS Argparse, Click, Fire, Docopt
     - Argparse: 코드량이 2~3배 많아짐, 깔끔한 출력 결과 어려움
     - Click: Typer보다 코드량 조금 많아짐
     - Fire: 제어가 약함
     - Docopt: 직관적이나 유지보수 조금 어려움

2. Typer.Argument, Typer.Option 비교
   - Argument(위치 인자)
     - 사용 방식: flag 없이 명령어 뒤에 바로 값을 적음
     - 필수 여부: 기본적으로 필수 (default 값 주면 선택 가능)
     - CLI 예시: `afo stats [folder path]`
     - 코드 작성법: `folder: str = typer.Argument(...)`
     - 장점: 타이핑이 짧고 간결함, 자연스러운 명령어 형태
     - 단점: 여러 개가 있으면 순서가 중요함

   - Option(옵션)
     - 사용 방식: `--이름` 또는 `-단축` 형태로 flag를 붙여서 사용
     - 필수 여부: 기본적으로 선택 (default 값이 있음)
     - CLI 예시: `afo stats --folder [folder path]`
     - 코드 작성법: `folder: str = typer.Option(...)`
     - 장점: 옵션이 많을 때 직관적, 순서 상관없이 사용 가능
     - 단점: 타이핑이 조금 길어짐

   - Typer.Argument를 사용한 이유
     - stats 명령어는 자주 사용하는 명령어이기 때문에 최대한 짧고 직관적으로 만들기 위함
     - `afo stats [folder path]`처럼 한 번에 폴더를 지정하는 게 사용자 경험상 더 편함
     - `rename`과 `move` 명령어도 같은 방식으로 자주 사용하는 명령어이므로 Argument를 사용

### Rich 라이브러리

1. Rich 선택 이유
   - VS Textual
     - Rich는 테이블, 진행바, 색상, 이모지, 마크다운까지 지원
     - Textual는 인터렉티브 TUI 구현에 필요 (현재로서는 필요 없음)

   - Colorama, Tabulate, tqdm
     - Colorama: 색상 지원 전문
     - Tabulate: 테이블 전문
     - tqdm: 진행바 전문
