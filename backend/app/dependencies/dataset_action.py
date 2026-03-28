from fastapi import Query, HTTPException
from typing import cast
from sklearn.utils import Bunch
import pandas as pd


def get_dataset(dataset_name: str = Query(...)):
    if dataset_name == "iris":
        from sklearn.datasets import load_iris

        data = cast(Bunch, load_iris())
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    elif dataset_name == "wine":
        from sklearn.datasets import load_wine

        data: Bunch = cast(Bunch, load_wine())
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df

    else:
        raise HTTPException(status_code=404, detail="Dataset not found")
