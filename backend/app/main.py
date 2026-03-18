from fastapi import FastAPI
from .routers.datasets import explore

app = FastAPI()
app.include_router(explore.router)


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}
