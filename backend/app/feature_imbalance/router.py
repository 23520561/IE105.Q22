import pandas as pd
from fastapi import APIRouter
from service import handle_imbalanced

from .schemas import ImbalancedRequest

router = APIRouter(prefix="/imbalanced", tags=["Imbalanced Data"])


@router.post("/")
def handle_imbalanced_data(req: ImbalancedRequest):
    df = pd.DataFrame(req.data)
    result_df = handle_imbalanced(df, req.target, req.method)
    return {"data": result_df.to_dict("records"), "shape": result_df.shape}
