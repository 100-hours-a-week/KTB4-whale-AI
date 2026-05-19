import typer
from pathlib import Path
from collections import Counter
from rich.table import Table
from rich.console import Console
from .rules import get_destination_folder

console = Console()

def analyze_folder(folder_path: str) -> None:
    """주어진 폴더의 확장자를 분석한 후, Rich 테이블로 출력"""
    target = Path(folder_path).expanduser().resolve()

    if not target.is_dir():
        console.print(f"[red]오류: {target} 폴더가 존재하지 않습니다.[/red]")
        raise typer.Exit(1)
    
    # 파일만 수집 (폴더 제외, 숨김 파일 제외)
    files = [f for f in target.iterdir() if f.is_file()]

    if not files:
        console.print(f"[yellow]{target} 폴더에 파일이 없습니다.[/yellow]")
        return
    
    # 확장자별 카운트
    ext_counter = Counter()
    for file in files:
        ext = file.suffix.lower() if file.suffix else "(no ext)"
        ext_counter[ext] += 1

    total_files = len(files)

    # Rich 테이블 생성
    table = Table(title=f"📊 {target} 폴더 분석 결과 (총 {total_files}개 파일)")
    table.add_column("확장자", style="cyan")
    table.add_column("개수", justify="right", style="green")
    table.add_column("비율", justify="right")
    table.add_column("이동될 폴더", style="magenta")

    for ext, count in sorted(ext_counter.items()):
        percentage = (count / total_files) * 100
        dest_folder = get_destination_folder(ext)
        table.add_row(
            ext,
            str(count),
            f"{percentage:.1f}%",
            dest_folder
        )

    console.print(table)
    console.print(f"\n[bold green]총 {len(ext_counter)}개 확장자 발견됨.[/bold green]")