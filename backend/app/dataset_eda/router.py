from typing import Annotated, List, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.dataset_eda.dependencies import check_columns_exist
from app.dataset_eda.schemas import (
    BoxPlotResponse,
    ColumnInfoResponse,
    HeatmapResponse,
    HistogramResponse,
    KDEResponse,
    PagingParams,
    PCAResponse,
    RowsResponse,
)
from app.dataset_eda.service import EdaService
from app.dependencies.dataset_action import get_dataset

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
) -> RowsResponse:
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


@router.get("/duplicates")
def get_duplicates(
    paging: PagingParams = Depends(),
    df: pd.DataFrame = Depends(get_dataset),
    subset: List[str] | None = Depends(check_columns_exist),
    keep: Literal["first", "last", "false"] = "false",
) -> RowsResponse:
    return EdaService.get_duplicated_rows(
        limit=paging.limit,
        offset=paging.offset,
        df=df,
        subset=subset,
        keep=False if keep == "false" else keep,
    )


@router.get("/missing")
def get_missings(
    paging: PagingParams = Depends(),
    df: pd.DataFrame = Depends(get_dataset),
    subset: List[str] | None = Depends(check_columns_exist),
) -> RowsResponse:
    return EdaService.get_missing_rows(
        limit=paging.limit,
        offset=paging.offset,
        df=df,
        subset=subset,
    )


@router.get("/pca")
def get_PCA(df: pd.DataFrame = Depends(get_dataset)) -> PCAResponse:
    try:
        return EdaService.get_pca_chart(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/scatterplot")
def get_scatterPlot(
    paging: PagingParams = Depends(),
    df: pd.DataFrame = Depends(get_dataset),
    subset: List[str] | None = Depends(check_columns_exist),
):
    if not (subset):
        raise HTTPException(status_code=400, detail="There is no input columns")
    return EdaService.get_scatterplot(df, subset, paging.limit, paging.offset)


@router.get("/kdeplot")
def get_kdeplot(
    df: pd.DataFrame = Depends(get_dataset),
    subset: str = Depends(check_column_numberic),
) -> KDEResponse:
    return EdaService.get_KDEplot(df, subset)


@router.get("/heatmap")
def get_heatmap(
    df: pd.DataFrame = Depends(get_dataset),
    subset: List[str] = Depends(check_columns_exist),
) -> HeatmapResponse:
    return EdaService.get_heatmap(df=df, subset=subset)
