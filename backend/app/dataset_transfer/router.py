from app.dataset_transfer.service import save_chunk
from fastapi import APIRouter, WebSocket

router = APIRouter(
    prefix="/dataset",
    tags=["dataset, transfer"],
    responses={404: {"description": "Not found"}},
)


@router.websocket("/upload")
async def upload_dataset(ws: WebSocket):
    await ws.accept()
    file_name = await ws.receive_text()
    if file_name:
        await ws.send_json({"type": "ready"})
    total_received = 0
    while True:
        try:
            message = await ws.receive()
            if "bytes" in message:
                chunk = message["bytes"]
                if not chunk:
                    break
                await save_chunk(file_name, chunk)
                total_received += len(chunk)
                await ws.send_json(
                    {"type": "progress", "uploaded_bytes": total_received}
                )
            else:
                break
        except Exception:
            break
    await ws.close()
