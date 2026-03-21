from typing import Hashable
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class RowsResponse(BaseModel):
    rows: List[Dict[Hashable, Any]]
    count: int


class PagingParams(BaseModel):
    limit: int = Field(10, ge=1, le=25, description="Number of rows to return")
    offset: int = Field(0, ge=0, description="Number of rows to skip")


class ColumnInfoResponse(BaseModel):
    columns: Dict[Hashable, Dict[Hashable, Any]]
    head: RowsResponse
    shape: Tuple[int, int]


class HistogramResponse(BaseModel):
    column: str
    bins: int
    histogram: List[Dict[str, int | float]]


class BoxPlotResponse(BaseModel):
    column: str
    min: float | int
    q1: float | int
    median: float | int
    q3: float | int
    max: float | int
    outliers: List[float | int]
    lower_bound: float | int
    upper_bound: float | int
