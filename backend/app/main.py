from fastapi import FastAPI

from app.dataset_eda import router as eda

app = FastAPI()
app.include_router(eda.router)


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}
