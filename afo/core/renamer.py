import typer
import shutil
import re
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from .utils import safe_filename, should_proceed, create_backup

console = Console()

def is_already_formatted(filename: str) -> bool:
    """이미 {num:02d}_{title}_{date} 패턴인지 확인"""
    # 01_title_20250514.jpg 형태인지 체크
    pattern = r'^\d{2}_.+_\d{8}\.[^.]+$'
    return bool(re.match(pattern, filename))

def format_filename(file: Path, pattern: str, index: int) -> str:
    """파일명을 패턴에 맞게 포맷팅"""
    date_str = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y%m%d")
    title = safe_filename(file.stem)

    new_name = pattern.format(
        num=index,
        title=title,
        date=date_str
    )

    return new_name + file.suffix.lower() # 확장자는 소문자로

def rename_files(
        folder_path: str,
        pattern: str = "{num:02d}_{title}_{date}",
        dry_run: bool = False,
        backup: bool = False,
        yes: bool = False,
) -> None:
    """파일명 변경 실행"""
    target = Path(folder_path).expanduser().resolve()

    if not target.is_dir():
        console.print(f"[red]오류: {target} 폴더가 존재하지 않습니다.[/red]")
        raise typer.Exit(1)
    
    files = sorted(
        (f for f in target.iterdir() if f.is_file() and not f.name.startswith('.')), # 숨김, 설정 파일은 제외
        key=lambda f: f.name.lower() # 대소문자 구분없이 정렬
    )
    
    if not files:
        console.print(f"[yellow]{target} 폴더에 파일이 없습니다.[/yellow]")
        return
    
    if backup:
        create_backup(target)

    if not should_proceed(dry_run, yes, "파일명 변경"):
        console.print("[yellow]작업이 취소되었습니다.[/yellow]")
        return
    
    table = Table(title=f"🔄 파일명 변경 결과 ({len(files)}개 파일)")
    table.add_column("원래 이름", style="red")
    table.add_column("새 이름", style="green")

    changed_count = 0
    skipped_count = 0
    for i, file in enumerate(files, 1):
        # 이미 포멧팅된 파일은 스킵
        if is_already_formatted(file.name):
            table.add_row(str(file.name), "[yellow]이미 포멧됨 (스킵)[/yellow]")
            skipped_count += 1
            continue

        new_name = format_filename(file, pattern, i)
        new_path = target / new_name

        # 이름이 같으면 스킵
        if new_path == file:
            table.add_row(str(file.name), "[dim]변경 불필요[/dim]")
            continue

        # 충돌 방지
        counter = 1
        original_new_path = new_path
        while new_path.exists():
            stem = original_new_path.stem
            new_path = target / f"{stem}({counter}){original_new_path.suffix}"
            counter += 1

        if dry_run:
            table.add_row(str(file.name), str(new_path.name))
        else:
            if backup:
                shutil.copy2(file, target / ".afo_backup" / file.name)
            file.rename(new_path)
            table.add_row(str(file.name), str(new_path.name))
            changed_count += 1

    console.print(table)
    if not dry_run:
        console.print(f"[bold green]✅ {changed_count}개 파일명 변경 완료! [/bold green]"
                      f"[yellow]({skipped_count}개 스킵됨)[/yellow]")
    else:
        console.print("[bold cyan]dry-run 모드: 실제 변경 없음[/bold cyan]")