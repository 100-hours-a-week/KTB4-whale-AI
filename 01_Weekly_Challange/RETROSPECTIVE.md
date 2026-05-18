# 회고록

## 1. 라이브러리 선택 및 비교

### Typer 라이브러리

1. **Typer 선택 이유**
   - VS Argparse, Click, Fire, Docopt
     - **Argparse**: 코드량이 2~3배 많아짐, 깔끔한 출력 결과 어려움
     - **Click**: Typer보다 코드량 조금 많아짐
     - **Fire**: 제어가 약함
     - **Docopt**: 직관적이나 유지보수 조금 어려움

2. **Typer.Argument, Typer.Option 비교**
   - **Argument(위치 인자)**
     - 사용 방식: flag 없이 명령어 뒤에 바로 값을 적음
     - 필수 여부: 기본적으로 필수 (default 값 주면 선택 가능)
     - CLI 예시: `afo stats [folder path]`
     - 코드 작성법: `folder: str = typer.Argument(...)`
     - 장점: 타이핑이 짧고 간결함, 자연스러운 명령어 형태
     - 단점: 여러 개가 있으면 순서가 중요함

   - _Option(옵션)_
     - 사용 방식: `--이름` 또는 `-단축` 형태로 flag를 붙여서 사용
     - 필수 여부: 기본적으로 선택 (default 값이 있음)
     - CLI 예시: `afo stats --folder [folder path]`
     - 코드 작성법: `folder: str = typer.Option(...)`
     - 장점: 옵션이 많을 때 직관적, 순서 상관없이 사용 가능
     - 단점: 타이핑이 조금 길어짐

   - **Typer.Argument를 사용한 이유**
     - stats 명령어는 자주 사용하는 명령어이기 때문에 최대한 짧고 직관적으로 만들기 위함
     - `afo stats [folder path]`처럼 한 번에 폴더를 지정하는 게 사용자 경험상 더 편함
     - `rename`과 `move` 명령어도 같은 방식으로 자주 사용하는 명령어이므로 Argument를 사용

### Rich 라이브러리

1. **Rich 선택 이유**
   - VS Textual
     - Rich는 테이블, 진행바, 색상, 이모지, 마크다운까지 지원
     - Textual는 인터렉티브 TUI 구현에 필요 (현재로서는 필요 없음)

   - Colorama, Tabulate, tqdm
     - Colorama: 색상 지원 전문
     - Tabulate: 테이블 전문
     - tqdm: 진행바 전문

## 2. 주요 기술적 의사결정 및 구현

- **모듈러 구조** (`afo/core/` 폴더)
  - analyzer, renamer, mover, organizer, scanner로 분리 (재사용성, 유지보수성 고려)

- **VirusTotal Async 스캔**
  - Public API는 분당 제한 4 requests / min
  - 동기 방식으로는 1분에 4개 파일 밖에 검사 못 함
  - async + rate limiter를 쓰면 제한 시간 내 최대한 병렬 처리 가능

- organize 명령어
  - rename + move를 한 번에 수행하는 통합 명령어

- 안전장치 고려
  - `--dry-run`, `--backup`, `--yes`, 숨김 파일 자동 제외, 이름 충돌 방지 등 안전장치 구현

## 3. 배운 점

- 비동기 프로그래밍: asyncio + aiohttp를 통해 I/O bound 작업(네트워크 API 호출)에서 async의 실질적인 효과 경험
- CLI 모듈 설계: 명령어별로 모듈화 하여 설계, Typer의 Argument/Option 차이점
- Rate Limiting: 외부 API를 사용할 때 고려해야 하는 개념 확인

## 4. 개선하고 싶은 점

- organize 명령어에 진행 Bar (Rich Progress) 실시간 표시 기능 추가
- pytest를 활용한 단위 테스트 코드 추가
- GUI 버전 확대
