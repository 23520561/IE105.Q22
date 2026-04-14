from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = (BASE_DIR / "../../storage").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def save_chunk(file_name, chunk):
    file_path = UPLOAD_DIR / file_name
    with open(file_path, "ab") as f:
        f.write(chunk)
