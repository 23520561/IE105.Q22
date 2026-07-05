from app.dependencies.session_manager import add_dataset
from app.dependencies.session_manager import get_session
from app.dataset_transfer.schemas import UploadedDataset
from datetime import datetime
import csv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = (BASE_DIR / "../../storage").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_uploaded_dataset(wp_id: str):
    datasets = []
    session = get_session(wp_id)
    if session:
        workspace = session.datasets
        for info in workspace:
            name = f"{info.stored_name}.csv"
            file = UPLOAD_DIR / name
            stat = file.stat()

            datasets.append(
                UploadedDataset(
                    id=str(info.id),
                    name=info.name,
                    dateModified=datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    size=stat.st_size,
                )
            )

    return datasets


async def validate_file(file_name: str):
    MAX_COLUMNS = 100
    MAX_ROWS = 1_000_000

    file_path = UPLOAD_DIR / file_name
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV is empty")

        if len(header) > MAX_COLUMNS:
            raise ValueError("Too many columns")

        if len(set(header)) != len(header):
            raise ValueError("Duplicate column names")

        row_count = 0

        for row in reader:
            row_count += 1

            if len(row) != len(header):
                raise ValueError(f"Row {row_count + 1} has the wrong number of columns")

            if row_count > MAX_ROWS:
                raise ValueError("Too many rows")


async def delete_file(file_name: str):
    file_path = UPLOAD_DIR / file_name
    os.remove(file_path)


async def save_chunk(file_name: str, chunk):
    file_path = UPLOAD_DIR / file_name
    with open(file_path, "ab") as f:
        f.write(chunk)


def save_file(file_name, session_id, stored_name):
    add_dataset(session_id=session_id, name=file_name, stored_name=stored_name)


MAX_AGE = 3600
