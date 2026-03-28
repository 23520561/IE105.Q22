import pandas as pd
import service as encoding
from fastapi import APIRouter

from .schemas import EncodingRequest

router = APIRouter(prefix="/encoding", tags=["Encoding"])


@router.post("/")
def encode_data(req: EncodingRequest):
    df = pd.DataFrame(req.data)
    method = req.method
    if method == "one_hot":
        df = encoding.one_hot(df, req.columns)
    elif method == "label":
        df = encoding.label_encode(df, req.column)
    elif method == "target":
        df = encoding.target_encode(df, req.column, req.target)
    elif method == "count":
        df = encoding.count_encode(df, req.column)
    elif method == "freq":
        df = encoding.freq_encode(df, req.column)
    elif method == "binary":
        df = encoding.binary_encode(df, req.column)
    elif method == "ordinal":
        df = encoding.ordinal_encode(df, req.column, req.mapping)
    else:
        raise ValueError("Unsupported encoding method")
    return {"data": df.to_dict("records")}
