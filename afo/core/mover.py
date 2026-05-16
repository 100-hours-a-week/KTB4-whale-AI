import typer
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from .utils import should_proceed, create_backup
from .rules import get_destination_folder

console = Console()

def move_files(
    folder_path: str,
    dry_run: bool = False,
    backup: bool = False,
    yes: bool = False,
) -> None:
    """확장자 RULES에 따라 파일을 자동으로 폴더 이동"""
    target = Path(folder_path).expanduser().resolve()

    if not target.is_dir():
        console.print(f"[red]오류: {target} 폴더가 존재하지 않습니다. [/red]")
        raise typer.Exit(1)
    
    # 숨김 파일 제외 + 이름순 정렬 (rename과 동일)
    files = sorted(
        (f for f in target.iterdir() if f.is_file() and not f.name.startswith('.')),
        key=lambda f: f.name.lower()
    )

    hidden_count = len([f for f in target.iterdir() if f.is_file() and f.name.startswith('.')])

    if not files:
        console.print(f"[yellow]{target} 폴더에 처리할 파일이 없습니다.[/yellow]")
        if hidden_count > 0:
            console.print(f"[dim](숨김 파일 {hidden_count}개는 제외됨)[/dim]")
        return
    
    if backup:
        create_backup(target)

    if not should_proceed(dry_run, yes, "파일 이동"):
        console.print("[yellow]작업이 취소되었습니다.[/yellow]")
        return
    
    table = Table(title=f"📦 파일 이동 결과 ({len(files)}개 파일)")
    table.add_column("원래 이름", style="red")
    table.add_column("이동될 위치", style="green")

    moved_count = 0

    for file in files:
        ext = file.suffix
        dest_folder_name = get_destination_folder(ext)
        dest_folder = target / dest_folder_name

        # 이동할 폴더가 없으면 자동 생성
        dest_folder.mkdir(exist_ok=True)
        dest_path = dest_folder / file.name

        # 같은 위치면 스킵
        if dest_path == file:
            table.add_row(str(file.name), "[dim]이동 불필요 (이미 해당 폴더)[/dim]")
            continue

        # 충돌 방지 (같은 이름 파일 이미 있으면 (1), (2) 붙임)
        counter = 1
        original_dest = dest_path
        while dest_path.exists():
            stem = original_dest.stem
            dest_path = dest_folder / f"{stem}({counter}){original_dest.suffix}"
            counter += 1
        
        if dry_run:
            table.add_row(str(file.name), str(dest_path.relative_to(target)))
        else:
            if backup:
                shutil.copy2(file, target / ".afo_backup" / file.name)
            shutil.move(str(file), str(dest_path))
            table.add_row(str(file.name), str(dest_path.relative_to(target)))
            moved_count += 1
        
    console.print(table)
    
    if hidden_count > 0:
        console.print(f"[dim]숨김 파일 {hidden_count}개는 move에서 제외되었습니다.[/dim]")
    
    if not dry_run:
        console.print(f"[bold green]✅ {moved_count}개 파일 이동 완료![/bold green]")
    else:
        console.print("[bold cyan]dry-run 모드: 실제 이동 없음[/bold cyan]")