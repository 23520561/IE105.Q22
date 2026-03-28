import pandas as pd
import service as transformation
from fastapi import APIRouter

from .schemas import TransformRequest

router = APIRouter(prefix="/transformation", tags=["Transformation"])


@router.post("/")
def transform_data(req: TransformRequest):
    df = pd.DataFrame(req.data)
    method = req.method
    if method == "log":
        df = transformation.log_transform(df, req.columns)
    elif method == "sqrt":
        df = transformation.sqrt_transform(df, req.columns)
    elif method == "minmax":
        df = transformation.minmax_scale(df, req.columns)
    elif method == "standard":
        df = transformation.standard_scale(df, req.columns)
    elif method == "robust":
        df = transformation.robust_scale(df, req.columns)
    elif method == "power":
        df = transformation.power_transform(df, req.columns)
    elif method == "normalize":
        df = transformation.normalize(df, req.columns)
    else:
        raise ValueError("Unsupported transformation method")
    return {"data": df.to_dict("records")}
