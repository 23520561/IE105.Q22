from fastapi import FastAPI
import asyncio
from pathlib import Path
import os
from contextlib import asynccontextmanager
from time import time

BASE_DIR = Path(__file__).resolve().parent
MAX_AGE = 3600
UPLOAD_DIRS = [
    (BASE_DIR / "../../storage").resolve(),
    (BASE_DIR / "../../cached").resolve(),
    (BASE_DIR / "../../projects").resolve(),
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(cleanup_loop())

    yield

    task.cancel()


def cleanup_folder():
    now = time()
    for UPLOAD_DIR in UPLOAD_DIRS:
        for file in os.listdir(UPLOAD_DIR):
            path = os.path.join(UPLOAD_DIR, file)

            if os.path.isfile(path):
                if now - os.path.getmtime(path) > MAX_AGE:
                    os.remove(path)


async def cleanup_loop():
    while True:
        cleanup_folder()
        await asyncio.sleep(60)
