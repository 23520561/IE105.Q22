from app.cleanup.service import lifespan
from fastapi import Cookie
from app.dependencies.session_manager import create_session
from starlette.responses import Response

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.dataset_chart import router as Chart
from app.dataset_column import router as Column
from app.dataset_eda import router as eda
from app.dataset_transfer import router as Storage
from app.decision_tree import router as DecisionTree
from app.feature_encoding import router as Encoding
from app.feature_engineering import router as FeatureEngineer
from app.feature_imbalance import router as Imbalanced
from app.feature_selection import router as FeatureSelection
from app.feature_transformation import router as Transformation
from app.pipeline import router as Pipeline
from app.server_stat import router as ServerStat
from app.dataset_project import router as Project

app = FastAPI(lifespan=lifespan)
app.include_router(eda.router)
app.include_router(ServerStat.router)
app.include_router(Storage.router)
app.include_router(Column.router)
app.include_router(Chart.router)
app.include_router(FeatureSelection.router)
app.include_router(Encoding.router)
app.include_router(Transformation.router)
app.include_router(Imbalanced.router)
app.include_router(FeatureEngineer.router)
app.include_router(Pipeline.router)
app.include_router(DecisionTree.router)
app.include_router(Project.router)


async def rate_limit_handler(
    request: Request,
    exc: Exception,
) -> Response:
    assert isinstance(exc, RateLimitExceeded)
    return _rate_limit_exceeded_handler(request, exc)


limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    rate_limit_handler,
)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def security_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


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


@app.get("/session")
def get_session(response: Response, x_session_id=Cookie(None)):
    if x_session_id:
        return
    id = create_session()
    response.set_cookie(
        key="x_session_id", value=str(id), httponly=True, secure=True, samesite="none"
    )
    return id
