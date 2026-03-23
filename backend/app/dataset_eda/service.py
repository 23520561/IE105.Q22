from app.dataset_eda.schemas import PCAResponse
from app.dataset_eda.schemas import RowsResponse
from typing import Literal
from typing import Any, Dict, Hashable, List, Tuple

from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

from app.dataset_eda.schemas import (
    BoxPlotResponse,
    ColumnInfoResponse,
    HistogramResponse,
)


class EdaService:
    def get_filtered_rows(
        query: str,
        limit: int,
        offset: int,
        df: pd.DataFrame,
    ) -> RowsResponse:
        filtered_df = df.query(query) if query else df

        # apply limit and offset
        filtered_df = filtered_df.iloc[offset : offset + limit]
        rows: List[Dict[Hashable, Any]] = filtered_df.to_dict(orient="records")
        return RowsResponse(rows=rows, count=len(rows))

    def get_columns(
        df: pd.DataFrame,
    ) -> ColumnInfoResponse:
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
        columnsInfo: Dict[Hashable, Dict[Hashable, Any]] = df.describe().to_dict(
            "index"
        )
        # Get the first 5 rows (head) of the DataFrame
        rows = df.head().to_dict(orient="records")  # Convert to a list of dictionaries
        head = RowsResponse(rows=rows, count=len(rows))

        # Get the shape of the DataFrame (rows, columns)
        shape: Tuple[int, int] = df.shape  # This is a tuple (rows, columns)

        return ColumnInfoResponse(
            columns=columnsInfo,
            head=head,
            shape=shape,
        )

    def get_column_histogram(
        column_name: str,
        bins: int,
        df: pd.DataFrame,
    ) -> HistogramResponse:
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
        col_data = df[column_name].dropna().to_numpy()
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

        return HistogramResponse(
            column=column_name,
            bins=bins,
            histogram=histogram,
        )

    def get_boxplot_statistics(
        column_name: str,
        df: pd.DataFrame,
    ) -> BoxPlotResponse:
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
        outliers: pd.Series = col_data[
            (col_data < lower_bound) | (col_data > upper_bound)
        ]

        # Ensure that outliers is always a list (even if there are no outliers)
        outliers_list: List[float | int] = (
            outliers.tolist()
        )  # Convert the Series of outliers to a list

        # Return only outliers, along with the stats to draw the box plot
        return BoxPlotResponse(
            column=column_name,
            min=min_val,
            q1=q1,
            median=median,
            q3=q3,
            max=max_val,
            outliers=outliers_list,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def get_duplicated_rows(
        limit: int,
        offset: int,
        df: pd.DataFrame,
        keep: Literal[False, "first", "last"],
        subset: None | List[str] = None,
    ) -> RowsResponse:
        """
        Find duplicated rows in a DataFrame.

        Args:
            df: Input DataFrame
            subset: Columns to consider (None = all columns)
            keep: 'first', 'last', or False
                - 'first': mark duplicates except first occurrence
                - 'last': mark duplicates except last occurrence
                - False: mark ALL duplicates

        Returns:
            DataFrame containing duplicated rows
        """
        duplicates_mask = df.duplicated(subset=subset, keep=keep)
        duplicates_mask = duplicates_mask.iloc[offset : offset + limit]
        rows = df[duplicates_mask].to_dict("records")
        return RowsResponse(rows=rows, count=len(rows))

    def get_missing_rows(
        limit: int,
        offset: int,
        df: pd.DataFrame,
        subset: None | List[str] = None,
    ) -> RowsResponse:
        """
        Find rows with missing values in a DataFrame.

        Args:
            df: Input DataFrame
            subset: Columns to check (None = all columns)

        Returns:
            Rows containing missing values
        """
        if subset:
            missing_mask = df[subset].isna().any(axis=1)
        else:
            missing_mask = df.isna().any(axis=1)
        missing_df = df[missing_mask]
        missing_df = missing_df.iloc[offset : offset + limit]

        rows = missing_df.to_dict("records")

        return RowsResponse(rows=rows, count=len(missing_df))

    def get_pca_chart(df: pd.DataFrame) -> PCAResponse:
        """
        Compute PCA projection for a DataFrame and return 2D coordinates
        along with explained variance statistics.

        Args:
            df: Input DataFrame containing numeric columns only.
                Non-numeric columns are automatically ignored.

        Returns:
            PCAResponse containing:
                - points: List of PCA-transformed coordinates.
                    Each point represents one row from the original dataset:
                        - pc1: Value along the first principal component
                        - pc2: Value along the second principal component

                - explained_variance: List of variance ratios explained by each component.
                    Example: [0.65, 0.25] means:
                        - PC1 explains 65% of variance
                        - PC2 explains 25% of variance

                - total_variance: Sum of explained variance of selected components.
                    Indicates how much information is preserved in the 2D projection.
        """
        # Ensure only numeric data is used
        numeric_df = df.select_dtypes(include="number").dropna()
        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for PCA")

        # Fit PCA with 2 components
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(numeric_df)

        # Convert PCA result into structured points
        points = [{"pc1": row[0], "pc2": row[1]} for row in transformed]

        # Explained variance metrics
        explained_variance: List = pca.explained_variance_ratio_.tolist()

        return PCAResponse(
            points=points,
            explained_variance=explained_variance,
            total_variance=sum(explained_variance),
        )
