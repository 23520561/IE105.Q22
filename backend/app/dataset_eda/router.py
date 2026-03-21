from app.dataset_eda.schemas import HistogramResponse
from app.dataset_eda.schemas import BoxPlotResponse
from app.dataset_eda.schemas import ColumnInfoResponse
from typing import Annotated
from app.dataset_eda.schemas import PagingParams
from typing import Any, Dict, Hashable, List

import pandas as pd
from fastapi import APIRouter, Depends, Query

from app.dataset_eda.service import EdaService

from ..dependencies import get_dataset
from .dependencies import build_query, check_column_numberic

router = APIRouter(
    prefix="/dataset",
    tags=["dataset"],
    responses={404: {"description": "Not found"}},
)


@router.get("/filters")
def get_filtered_rows(
    paging: Annotated[PagingParams, Query()],
    query: str = Depends(build_query),
    df: pd.DataFrame = Depends(get_dataset),
) -> List[Dict[Hashable, Any]]:
    return EdaService.get_filtered_rows(
        query,
        paging.limit,
        paging.offset,
        df,
    )


@router.get("/columns")
async def get_columns(
    df: pd.DataFrame = Depends(get_dataset),
) -> ColumnInfoResponse:
    return EdaService.get_columns(df)


# Get histogram statistics for a single column
@router.get("/columns/{column_name}/histogram")
async def get_column_histogram(
    column_name: str = Depends(
        check_column_numberic
    ),  # The column name (from the URL path)
    bins: int = Query(10, ge=1, le=100),  # Number of bins, must be between 1 and 100
    df: pd.DataFrame = Depends(
        get_dataset
    ),  # The DataFrame passed via the Depends function
) -> HistogramResponse:
    return EdaService.get_column_histogram(
        column_name,
        bins,
        df,
    )


@router.get("/columns/{column_name}/boxplot")
async def get_boxplot_statistics(
    column_name: str = Depends(
        check_column_numberic
    ),  # Column name of the numeric column to check
    df: pd.DataFrame = Depends(
        get_dataset
    ),  # DataFrame passed via dependency injection
) -> BoxPlotResponse:
    return EdaService.get_boxplot_statistics(column_name, df)
