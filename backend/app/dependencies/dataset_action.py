from typing import List
from fastapi import Depends
from fastapi import Query, HTTPException
from typing import cast
from sklearn.utils import Bunch
import pandas as pd


def get_dataset(dataset_id: str = Query(...)):
    if dataset_id == "iris":
        from sklearn.datasets import load_iris

        data = cast(Bunch, load_iris())
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    elif dataset_id == "wine":
        from sklearn.datasets import load_wine

        data: Bunch = cast(Bunch, load_wine())
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    else:
        raise HTTPException(status_code=404, detail="Dataset not found")


def check_column_exist(
    column_name: str = Query(...), df: pd.DataFrame = Depends(get_dataset)
) -> str:
    if column_name not in df.columns:
        raise HTTPException(status_code=404, detail="Column not found")
    return column_name


def check_columns_exist(
    subset: List[str] = Query(default=None), df: pd.DataFrame = Depends(get_dataset)
) -> List[str] | None:
    if not subset:
        return None
    return [check_column_exist(c, df) for c in subset]


def check_column_numberic(
    column_name: str = Depends(check_column_exist),
    df: pd.DataFrame = Depends(get_dataset),
) -> str:
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise HTTPException(status_code=400, detail="Column is not numeric")
    return column_name
