import json
import uuid
from collections import defaultdict

from fastapi import APIRouter, Cookie, Query, WebSocket

from app.dataset_transfer.service import (
    delete_file,
    get_uploaded_dataset,
    save_chunk,
    save_file,
    validate_file,
)

router = APIRouter(
    prefix="/dataset",
    tags=["dataset, transfer"],
    responses={404: {"description": "Not found"}},
)


MAX_CONNECTIONS_PER_IP = 3

connections_per_ip = defaultdict(int)


@router.get("/uploaded")
def get_uploaded(x_session_id: str = Cookie(None)):
    return get_uploaded_dataset(x_session_id)


@router.delete("/uploaded")
async def delete_uploaded(file_id=Query(...)):
    return await delete_file(file_id)


# TODO: check every function, add error, clean upload dataset, add session id for pipeline
# add header for session id, add rate limit for ws


@router.websocket("/upload")
async def upload_dataset(ws: WebSocket):
    await ws.accept()
    ip = ws.client.host if ws.client else "unknown"
    if connections_per_ip[ip] >= MAX_CONNECTIONS_PER_IP:
        await ws.close(code=1008)
        return
    connections_per_ip[ip] += 1

    MAX_SIZE = 50 * 1024 * 1024
    message = await ws.receive_text()
    data = json.loads(message)

    session_id = ws.cookies.get("x_session_id")
    file_name = data["fileName"]
    stored_name = uuid.uuid4()

    if file_name:
        await ws.send_json({"type": "ready"})
    total_received = 0
    while True:
        message = await ws.receive()
        if "bytes" not in message:
            break
        chunk = message["bytes"]
        if not chunk:
            break
        total_received += len(chunk)
        if total_received > MAX_SIZE:
            await ws.send_json(
                {"type": "error", "message": "File exceeds the 50 MB limit."}
            )
            await ws.close(code=1009)
            return
        await save_chunk(f"{str(stored_name)}.csv", chunk)
        await ws.send_json({"type": "progress", "uploaded_bytes": total_received})
    save_file(session_id=session_id, file_name=file_name, stored_name=stored_name)
    try:
        await validate_file(f"{stored_name}.csv")
        await ws.send_json({"type": "success"})
    except ValueError as e:
        await ws.send_json(
            {
                "type": "error",
                "message": str(e),
            }
        )
        await delete_file(str(stored_name))  # cleanup
    finally:
        connections_per_ip[ip] -= 1
        await ws.close()
