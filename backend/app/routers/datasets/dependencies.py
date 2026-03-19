from fastapi import Depends
import pandas as pd
from fastapi import HTTPException, Request
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


def build_query(request: Request, df: pd.DataFrame = Depends(get_dataset)) -> str:
    filters = {
        k: v for k, v in request.query_params.items() if k not in {"limit", "offset"}
    }
    expressions = []

    for key, value in filters.items():
        if "__" in key:
            raise HTTPException(400, f"Invalid filter param: {key}")

        # Determine operator and column
        if key.startswith("min_"):
            col = key[4:]
            op = ">="
        elif key.startswith("max_"):
            col = key[4:]
            op = "<="
        elif key.startswith("not_"):
            col = key[4:]
            op = "!="
        else:
            col = key
            op = "=="

        if col not in df.columns:
            raise HTTPException(400, f"Column not found: {col}")

        # wrap string between ""
        if pd.api.types.is_string_dtype(df[col]):
            val_str = f'"{value}"'
        # turn number to string
        else:
            val_str = str(value)

        expressions.append(f"{col} {op} {val_str}")

    # build query string
    query_str = " and ".join(expressions)
    return query_str
