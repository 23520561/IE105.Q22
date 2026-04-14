from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dataset_eda import router as eda
from app.server_stat import router as ServerStat
from app.dataset_transfer import router as Storage
from app.dataset_column import router as Column
from app.dataset_chart import router as Chart

app = FastAPI()
app.include_router(eda.router)
app.include_router(ServerStat.router)
app.include_router(Storage.router)
app.include_router(Column.router)
app.include_router(Chart.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}
