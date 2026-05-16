## Typer.Argument, Typer.Option 비교

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
