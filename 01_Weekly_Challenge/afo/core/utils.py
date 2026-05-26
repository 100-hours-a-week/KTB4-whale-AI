# 안전장치 함수 - 공통 Utils
import shutil
import re
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.prompt import Confirm

console = Console()

def safe_filename(name: str) -> str:
    """파일명 안전하게 정리 (특수문자 제거)"""
    name = re.sub(r'[<>:"/\\|?*]', '_', name) # Windows 금지 문자
    name = re.sub(r'\s+', '_', name.strip()) # 공백은 '_'
    return name

def create_backup(folder: Path) -> None:
    """백업 폴더 생성"""
    backup_dir = folder / ".afo_backup"
    backup_dir.mkdir(exist_ok=True)
    console.print(f"[yellow]백업 폴더 생성: {backup_dir}[/yellow]")

def should_proceed(dry_run: bool, yes: bool, action: str) -> bool:
    """dry-run 또는 확인 창 처리"""
    if dry_run:
        console.print(f"[bold cyan]{action} (dry-run 모드 - 실제 변경 없음)[/bold cyan]")
        return True
    if yes:
        return True
    return Confirm.ask(f"[bold]정말 {action}하시겠습니까?[/bold]", default=False)
