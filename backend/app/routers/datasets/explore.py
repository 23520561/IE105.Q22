from typing import Hashable
from .dependencies import check_column_numberic, build_query
from fastapi import APIRouter, Depends, Query
import pandas as pd
import numpy as np
from ...dependencies import get_dataset
from typing import List, Dict, Union, Any, Tuple

router = APIRouter(
    prefix="/dataset",
    tags=["dataset"],
    responses={404: {"description": "Not found"}},
)


@router.get("/filters")
def query_rows(
    query: str = Depends(build_query),
    limit: int = Query(10, ge=1, le=25),
    offset: int = Query(0, ge=0),
    df: pd.DataFrame = Depends(get_dataset),
) -> List[Dict[Hashable, Any]]:
    filtered_df = df.query(query) if query else df

    # apply limit and offset
    filtered_df = filtered_df.iloc[offset : offset + limit]
    return filtered_df.to_dict(orient="records")


@router.get("/columns")
async def get_columns(
    df: pd.DataFrame = Depends(get_dataset),
) -> Dict[
    str,
    Union[
        Dict[Hashable, Dict[Hashable, int | float]],
        List[Dict[Hashable, Any]],
        Tuple[int, int],
    ],
]:
    """
    Get the list of column names, first 5 rows (head), and shape of the dataset.

    Args:
        df (pd.DataFrame): The DataFrame passed via the Depends function.

    Returns:
        dict: A dictionary containing:
            - "columns": List of column names and their info.
            - "head": The first 5 rows of the DataFrame.
            - "shape": Tuple representing the shape of the DataFrame (rows, columns).
    """
    # Get the columns as a list
    columnsInfo: Dict[Hashable, Dict[Hashable, int | float]] = df.describe().to_dict(
        orient="index"
    )
    # Get the first 5 rows (head) of the DataFrame
    head: List[Dict[Hashable, Any]] = df.head().to_dict(
        orient="records"
    )  # Convert to a list of dictionaries

    # Get the shape of the DataFrame (rows, columns)
    shape: Tuple[int, int] = df.shape  # This is a tuple (rows, columns)

    return {
        "columns": columnsInfo,
        "head": head,
        "shape": shape,
    }


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
) -> Dict[str, Union[str, int, List[Dict[str, Union[float, int]]]]]:
    """
    Get the histogram statistics for a given column in the dataset.

    Args:
        column_name (str): The name of the column to compute the histogram for.
        bins (int): The number of bins to use for the histogram.
        df (pd.DataFrame): The pandas DataFrame containing the data.

    Returns:
        dict: A dictionary containing the column name, number of bins, and the histogram.
              The histogram is a list of dicts with 'bin_start', 'bin_end', and 'count' for each bin.
    """
    # Compute histogram using numpy
    col_data = np.asarray(df[column_name])
    counts, bin_edges = np.histogram(col_data, bins=bins)

    # Prepare the histogram data as a list of dicts
    histogram = [
        {
            "bin_start": float(bin_edges[i]),
            "bin_end": float(bin_edges[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    return {
        "column": column_name,  # Column name (string)
        "bins": bins,  # Number of bins (integer)
        "histogram": histogram,  # Histogram data as a list of dictionaries
    }


@router.get("/columns/{column_name}/boxplot")
async def get_boxplot_statistics(
    column_name: str = Depends(
        check_column_numberic
    ),  # Column name of the numeric column to check
    df: pd.DataFrame = Depends(
        get_dataset
    ),  # DataFrame passed via dependency injection
) -> dict[str, float | int | List[float | int] | str]:
    """
    Retrieve boxplot statistics for a specific column in a DataFrame and identify outliers.

    Args:
        column_name (str): The name of the column in the DataFrame to calculate statistics for.
        df (pd.DataFrame): The pandas DataFrame containing the data.

    Returns:
        dict: A dictionary containing boxplot statistics (min, q1, median, q3, max, outliers, etc.).
    """

    # Get the data for the column
    col_data: pd.Series = df[column_name]

    # Calculate the box plot statistics and convert to native types
    min_val: float | int = col_data.min()
    q1: float | int = col_data.quantile(0.25)
    median: float | int = col_data.median()
    q3: float | int = col_data.quantile(0.75)
    max_val: float | int = col_data.max()

    # Calculate IQR (Interquartile Range)
    iqr: float = q3 - q1

    # Calculate lower and upper bounds for outliers
    lower_bound: float | int = q1 - 1.5 * iqr
    upper_bound: float | int = q3 + 1.5 * iqr

    # Identify outliers (values outside the whiskers)
    outliers: pd.Series = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

    # Ensure that outliers is always a list (even if there are no outliers)
    outliers_list: List[float | int] = (
        outliers.tolist()
    )  # Convert the Series of outliers to a list

    # Return only outliers, along with the stats to draw the box plot
    return {
        "column": column_name,
        "min": min_val,
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max_val,
        "outliers": outliers_list,  # Send only outliers as list
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }
