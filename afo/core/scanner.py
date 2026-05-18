import asyncio
import aiohttp
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

load_dotenv()
console = Console()

API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
if not API_KEY:
    console.print("[red]오류: VIRUSTOTAL_API_KEY가 .env에 설정되지 않았습니다.[/red]")
    raise SystemExit(1)

VT_BASE_URL = "https://www.virustotal.com/api/v3"

# Rate limiter: 분당 4회 제한
semaphore = asyncio.Semaphore(4)

async def scan_file_virustotal(file_path: str) -> dict:
    """단일 파일을 VirusTotal로 async 스캔 (rate limit + YARA 포함)"""
    file_path = Path(file_path)

    async with semaphore:
        async with aiohttp.ClientSession() as session:
            # 1. 파일 업로드
            with open(file_path, "rb") as f:
                data = {"file": f}
                headers = {"x-apikey": API_KEY}
                async with session.post(f"{VT_BASE_URL}/files", data=data, headers=headers) as response:
                    if response.status != 200:
                        return {"error": f"Upload failed: {response.status}"}
                    result = await response.json()
                    analysis_id = result["data"]["id"]
            
            # 2. 분석 결과 대기 (최대 30초 polling)
            for _ in range(10):  # 3초 간격 × 10회 = 최대 30초
                await asyncio.sleep(3)
                async with session.get(f"{VT_BASE_URL}/analyses/{analysis_id}", headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        stats = data["data"]["attributes"]["stats"]

                        # YARA 규칙 매칭 확인
                        yara_matches = data["data"]["attributes"].get("crowdsourced_yara_results", [])
                        yara_detected = len(yara_matches) > 0

                        return {
                            "sha256": data["data"]["attributes"]["sha256"],
                            "malicious": stats.get("malicious", 0),
                            "suspicious": stats.get("suspicious", 0),
                            "harmless": stats.get("harmless", 0),
                            "yara_detected": yara_detected,
                            "yara_rules": [rule["rule_name"] for rule in yara_matches[:3]] if yara_matches else []
                        }
            
            return {"error": "Analysis timeout"}
        
async def scan_files(files: list[Path]) -> dict:
    """여러 파일을 async + rate limit 적용하여 병렬 스캔"""
    console.print(f"[bold cyan]VirusTotal 스캔 시작 ({len(files)}개 파일)...[/bold cyan]")

    tasks = [scan_file_virustotal(str(f)) for f in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scan_result = {}
    for file, result in zip(files, results):
        if isinstance(result, dict):
            scan_result[str(file)] = result
        else:
            scan_result[str(file)] = {"error": str(result)}

    console.print("[bold green]VirusTotal 스캔 완료![/bold green]")
    return scan_result