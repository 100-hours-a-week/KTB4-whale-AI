import shutil
import asyncio
import typer
from pathlib import Path
from rich.console import Console
from .utils import should_proceed, create_backup
from .renamer import format_filename, is_already_formatted
from .rules import get_destination_folder
from .scanner import scan_files

console = Console()

async def organize_files(
    folder_path: str,
    pattern: str = "{num:03d}_{title}_{date}",
    dry_run: bool = False,
    backup: bool = False,
    yes: bool = False,
    scan: bool = True,
) -> None:
    target = Path(folder_path).expanduser().resolve()
    if not target.is_dir():
        console.print(f"[red]오류: {target} 폴더가 존재하지 않습니다.[/red]")
        raise typer.Exit(1)
    
    files = sorted(
        (f for f in target.iterdir() if f.is_file() and not f.name.startswith('.')),
        key=lambda f: f.name.lower()
    )

    if not files:
        console.print(f"[yellow]{target} 폴더에 처리할 파일이 없습니다.[/yellow]")
        return

    if backup:
        create_backup(target)

    if not should_proceed(dry_run, yes, "파일명 변경 + 폴더 이동"):
        console.print("[yellow]작업이 취소되었습니다.[/yellow]")
        return
    
    # VirusTotal async 스캔
    if scan:
        await scan_files(files)   # 스캔 결과는 콘솔에 바로 출력됨

    # ==================== 1. 비동기 rename ====================
    console.print("[bold cyan]1단계: 파일명 변경 중...[/bold cyan]")
    rename_tasks = []
    rename_results = []  # (원래파일, 새이름) 저장

    for i, file in enumerate(files, 1):
        if is_already_formatted(file.name):
            continue
        new_name = format_filename(file, pattern, i)
        new_path = target / new_name
        # 충돌 방지
        counter = 1
        original = new_path
        while new_path.exists():
            stem = original.stem
            new_path = target / f"{stem}({counter}){original.suffix}"
            counter += 1
        rename_tasks.append(asyncio.to_thread(_rename_one, file, new_path, dry_run, backup))
        rename_results.append((file, new_path))

    if rename_tasks:
        await asyncio.gather(*rename_tasks)

    # ==================== 2. 비동기 move ====================
    console.print("[bold cyan]2단계: 폴더 이동 중...[/bold cyan]")
    move_tasks = []

    for orig_file, renamed_file in rename_results:
        # rename이 dry-run이었으면 orig_file 그대로 사용
        current_file = renamed_file if not dry_run else orig_file
        if not current_file.exists():
            continue

        ext = current_file.suffix
        dest_folder_name = get_destination_folder(ext)
        dest_folder = target / dest_folder_name
        dest_folder.mkdir(exist_ok=True)
        dest_path = dest_folder / current_file.name

        # 충돌 방지
        counter = 1
        original_dest = dest_path
        while dest_path.exists():
            stem = original_dest.stem
            dest_path = dest_folder / f"{stem}({counter}){original_dest.suffix}"
            counter += 1

        move_tasks.append(asyncio.to_thread(_move_one, current_file, dest_path, dry_run, backup))

    if move_tasks:
        await asyncio.gather(*move_tasks)

    console.print("[bold green]✅ organize 작업 완료! (비동기 처리)[/bold green]")
    if dry_run:
        console.print("[bold cyan]dry-run 모드: 실제 변경 없음[/bold cyan]")

def _rename_one(file: Path, new_path: Path, dry_run: bool, backup: bool):
    """동기 rename (to_thread에서 호출)"""
    if dry_run:
        return
    if backup:
        shutil.copy2(file, file.parent / ".afo_backup" / file.name)
    file.rename(new_path)

def _move_one(file: Path, dest_path: Path, dry_run: bool, backup: bool):
    """동기 move (to_thread에서 호출)"""
    if dry_run:
        return
    if backup:
        shutil.copy2(file, file.parent / ".afo_backup" / file.name)
    shutil.move(str(file), str(dest_path))