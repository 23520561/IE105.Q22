from fastapi import Depends
import pandas as pd
from fastapi import HTTPException
from ...dependencies import get_dataset


def check_column_exist(
    column_name: str, df: pd.DataFrame = Depends(get_dataset)
) -> str:
    if column_name not in df.columns:
        raise HTTPException(status_code=404, detail="Column not found")
    return column_name


def check_column_numberic(
    column_name: str = Depends(check_column_exist),
    df: pd.DataFrame = Depends(get_dataset),
) -> str:
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise HTTPException(status_code=400, detail="Column is not numeric")
    return column_name
