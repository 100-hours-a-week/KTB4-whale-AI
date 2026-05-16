from pathlib import Path

# 확장자 - 폴더 매핑 (소문자로만 저장)
DEFAULT_RULES = {
    # Images
    ".jpg": "Images", ".jpeg": "Images", ".png": "Images", ".gif": "Images",
    ".webp": "Images", ".bmp": "Images", ".tiff": "Images", ".heic": "Images",
    ".avif": "Images", ".svg": "Images", ".ico": "Images",

    # Documents
    ".pdf": "Documents", ".doc": "Documents", ".docx": "Documents",
    ".xls": "Documents", ".xlsx": "Documents", ".ppt": "Documents",
    ".pptx": "Documents", ".txt": "Documents", ".md": "Documents",
    ".csv": "Documents", ".rtf": "Documents",
    
    # Archives
    ".zip": "Archives", ".rar": "Archives", ".7z": "Archives",
    ".tar": "Archives", ".gz": "Archives", ".bz2": "Archives",
    ".xz": "Archives", ".iso": "Archives", ".dmg": "Archives",
    
    # Videos
    ".mp4": "Videos", ".mov": "Videos", ".avi": "Videos",
    ".mkv": "Videos", ".webm": "Videos", ".m4v": "Videos",
    ".flv": "Videos", ".wmv": "Videos",
    
    # Audio
    ".mp3": "Audio", ".wav": "Audio", ".m4a": "Audio",
    ".flac": "Audio", ".aac": "Audio", ".ogg": "Audio",
    ".opus": "Audio", ".wma": "Audio",
}

def get_destination_folder(ext: str) -> str:
    """확장자를 받아서 이동할 폴더 이름을 반환 (없으면 Others)"""
    ext = ext.lower() # 소문자로 통일
    return DEFAULT_RULES.get(ext, "Others")