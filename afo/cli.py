import typer
from rich.console import Console

app = typer.Typer(
    name="afo",
    help="Auto File Organizer - 파일 자동 정리 CLI 도구",
    add_completion=True,
)

console = Console()

# 프로그램 시작할 때 제일 먼저 실행할 공통 함수
@app.callback()
def callback():
    """Auto File Organizer 시작"""
    
    pass

# TODO: 나중에 명령어를 여기에 추가 예정
if __name__ == "__main__":
    app()