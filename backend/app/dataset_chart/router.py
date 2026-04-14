from fastapi import HTTPException
from app.dataset_chart.schemas import PCAResponse
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, Query

from app.dataset_chart import service as ChartService
from app.dataset_chart.schemas import (
    BoxPlotResponse,
    HeatmapResponse,
    HistogramResponse,
)
from app.dependencies.dataset_action import (
    check_column_numberic,
    check_columns_exist,
    get_dataset,
)

router = APIRouter(
    prefix="/dataset/charts",
    tags=["dataset"],
    responses={404: {"description": "Not found"}},
)


# Get histogram statistics for a single column
@router.get("/histogram")
async def get_column_histogram(
    column_name: str = Depends(
        check_column_numberic
    ),  # The column name (from the URL path)
    bins: int = Query(10, ge=1, le=100),  # Number of bins, must be between 1 and 100
    df: pd.DataFrame = Depends(
        get_dataset
    ),  # The DataFrame passed via the Depends function
) -> HistogramResponse:
    return ChartService.get_column_histogram(
        column_name,
        bins,
        df,
    )


@router.get("/boxplot")
async def get_boxplot_statistics(
    column_name: str = Depends(
        check_column_numberic
    ),  # Column name of the numeric column to check
    df: pd.DataFrame = Depends(
        get_dataset
    ),  # DataFrame passed via dependency injection
) -> BoxPlotResponse:
    return ChartService.get_boxplot_statistics(column_name, df)


@router.get("/heatmap")
def get_heatmap(
    df: pd.DataFrame = Depends(get_dataset),
    subset: List[str] = Depends(check_columns_exist),
) -> HeatmapResponse:
    return ChartService.get_heatmap(df=df, subset=subset)


@router.get("/pca")
def get_PCA(df: pd.DataFrame = Depends(get_dataset)) -> PCAResponse:
    try:
        return ChartService.get_pca_chart(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
