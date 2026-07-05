from app.server_stat.schemas import ServerStatusResponse
import psutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = (BASE_DIR / "../../storage").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_ram() -> float:
    return psutil.virtual_memory().percent


def get_folder_size() -> int:
    return sum(f.stat().st_size for f in Path(UPLOAD_DIR).rglob("*") if f.is_file())


def format_size(bytes_val):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def get_server_status():
    return ServerStatusResponse(ram=get_ram(), storage=format_size(get_folder_size()))
