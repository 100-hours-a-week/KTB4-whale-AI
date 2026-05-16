import typer
from rich.console import Console
from afo.core.analyzer import analyze_folder
from afo.core.renamer import rename_files
from afo.core.mover import move_files

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

@app.command()
def stats(
    folder: str = typer.Argument(".", help="분석할 폴더 경로 (기본: 현재 폴더)")
):
    """폴더의 확장자별 파일 개수, 비율, 이동될 폴더 분석"""
    try:
        analyze_folder(folder)
    except Exception as e:
        console.print(f"[red]Error Occurred: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def rename(
    folder: str = typer.Argument(".", help="이름을 변경할 폴더 경로 (기본: 현재 폴더)"),
    pattern: str = typer.Option(
        "{num:02d}_{title}_{date}",
        "--pattern",
        "-p",
        help="파일명 패턴 (기본: {num:02d}_{title}_{date})"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="실제 변경 없이 미리보기"),
    backup: bool = typer.Option(False, "--backup", help="원본 파일 백업"),
    yes: bool = typer.Option(False, "--yes", "-y", help="확인 없이 바로 실행")
):
    """파일명을 지정된 패턴으로 변경"""
    try:
        rename_files(folder, pattern, dry_run, backup, yes)
    except Exception as e:
        console.print(f"[red]오류 발생: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def move(
    folder: str = typer.Argument(".", help="파일을 이동할 폴더 경로 (기본: 현재 폴더)"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="실제 이동 없이 미리보기"),
    backup: bool = typer.Option(False, "--backup", help="원본 파일 백업"),
    yes: bool = typer.Option(False, "--yes", help="확인 없이 바로 실행"),
):
    """확장자 RULES에 따라 파일을 자동으로 폴더 이동"""
    try:
        move_files(folder, dry_run, backup, yes)
    except Exception as e:
        console.print(f"[red]오류 발생: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()