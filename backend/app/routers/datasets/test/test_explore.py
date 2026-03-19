from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import get_dataset
from app.routers.datasets.dependencies import check_column_numberic

# Adjust this import based on where your FastAPI app is located
import pandas as pd


# Mock the `get_dataset` dependency for testing purposes
def mock_get_dataset():
    return pd.DataFrame({"feature1": [1, 2, 3, 4, 100], "feature2": [5, 4, 3, 2, 1]})


# Mock `check_column_numberic` to directly return the column name
def mock_check_column_numberic(column_name: str):
    # Assuming the column exists and is numeric, simply return the column name
    return column_name


# Override dependencies for testing
app.dependency_overrides[get_dataset] = mock_get_dataset
app.dependency_overrides[check_column_numberic] = mock_check_column_numberic

# Create TestClient instance for synchronous testing
client = TestClient(app)


# Test the `/dataset/columns` endpoint
def test_get_columns():
    response = client.get("/dataset/columns")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    expected_columns = {
        "count": {"feature1": 5.0, "feature2": 5.0},
        "mean": {"feature1": 22.0, "feature2": 3.0},
        "std": {"feature1": 43.617656975128774, "feature2": 1.5811388300841898},
        "min": {"feature1": 1.0, "feature2": 1.0},
        "25%": {"feature1": 2.0, "feature2": 2.0},
        "50%": {"feature1": 3.0, "feature2": 3.0},
        "75%": {"feature1": 4.0, "feature2": 4.0},
        "max": {"feature1": 100.0, "feature2": 5.0},
    }

    actual_columns = response.json().get("columns")
    assert actual_columns == expected_columns, (
        f"Expected columns {expected_columns}, got {actual_columns}"
    )

    # Check the shape of the DataFrame (should be 5 rows, 2 columns)
    expected_shape = [5, 2]  # 5 rows and 2 columns
    actual_shape = response.json().get("shape")
    assert actual_shape == expected_shape, (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )

    # Check the first 5 rows (head)
    expected_head = [
        {"feature1": 1, "feature2": 5},
        {"feature1": 2, "feature2": 4},
        {"feature1": 3, "feature2": 3},
        {"feature1": 4, "feature2": 2},
        {"feature1": 100, "feature2": 1},
    ]
    actual_head = response.json().get("head")
    assert actual_head == expected_head, (
        f"Expected head {expected_head}, got {actual_head}"
    )


# Test the `/dataset/columns/{column_name}/histogram` endpoint
def test_get_column_histogram():
    response = client.get("/dataset/columns/feature1/histogram?bins=2")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()

    # Column name check
    expected_column = "feature1"
    assert data["column"] == expected_column, (
        f"Expected column '{expected_column}', got '{data['column']}'"
    )

    # Bin count check
    expected_bins = 2
    assert data["bins"] == expected_bins, (
        f"Expected bins {expected_bins}, got {data['bins']}"
    )

    # Histogram length
    assert len(data["histogram"]) == expected_bins, (
        f"Expected {expected_bins} bins, got {len(data['histogram'])}"
    )

    # Check counts sum
    total_count = sum(bin["count"] for bin in data["histogram"])
    expected_count = 5  # We have 5 rows in mock dataset
    assert total_count == expected_count, (
        f"Total histogram count mismatch: expected {expected_count}, got {total_count}"
    )

    # Check bin structure
    for i, bin in enumerate(data["histogram"]):
        assert "bin_start" in bin and "bin_end" in bin and "count" in bin, (
            f"Bin {i} missing required keys"
        )


# Test for the boxplot statistics endpoint
def test_boxplot_statistics():
    expected_column = "feature1"
    # Test for feature1
    response = client.get("/dataset/columns/feature1/boxplot")

    # Assert the status code is 200 (success)
    assert response.status_code == 200

    # Assert the returned data contains the expected statistics
    data = response.json()

    # Check if the correct column name is returned
    assert data["column"] == expected_column, (
        f"Expected column '{expected_column}', got '{data['column']}'"
    )

    # Check if boxplot statistics are included in the response
    assert "min" in data, "Missing 'min' value in response"
    assert "q1" in data, "Missing 'q1' value in response"
    assert "median" in data, "Missing 'median' value in response"
    assert "q3" in data, "Missing 'q3' value in response"
    assert "max" in data, "Missing 'max' value in response"

    # Check if outliers are included in the response
    assert "outliers" in data, "Missing 'outliers' key in response"
    assert isinstance(data["outliers"], list), (
        f"Expected 'outliers' to be a list, but got {type(data['outliers'])}"
    )
    assert len(data["outliers"]) == 1, (
        f"Expected 1 outlier, but got {len(data['outliers'])} outliers"
    )

    # Test the expected outlier value (100 is the outlier in our mock data)
    assert data["outliers"] == [100], (
        f"Expected outlier values to be [100], but got {data['outliers']}"
    )


def mock_get_dataset():
    return pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40],
            "salary": [3000, 4000, 5000, 6000, 7000],
            "city": ["HCM", "HN", "HCM", "HN", "HCM"],
        }
    )


# Assume df has columns: "age", "salary", "city"
def test_filters_basic():
    app.dependency_overrides[get_dataset] = mock_get_dataset
    response = client.get("/dataset/filters?min_age=31")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    # All ages should be >= 31
    assert all(row["age"] >= 31 for row in data)


def test_filters_limit_offset():
    response = client.get("/dataset/filters?min_age=25&limit=2&offset=1")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2  # limit applied
    # Should be correct slice from dataset
    assert data[0]["age"] == 30
    assert data[1]["age"] == 35


def test_filters_multiple_conditions():
    response = client.get("/dataset/filters?min_age=25&city=HCM")
    assert response.status_code == 200

    data = response.json()
    # All rows should satisfy both conditions
    assert all(row["age"] >= 25 and row["city"] == "HCM" for row in data)


def test_filters_empty_result():
    response = client.get("/dataset/filters?max_age=10")
    assert response.status_code == 200

    data = response.json()
    assert data == []


def test_filters_output_format():
    response = client.get("/dataset/filters?min_age=25")
    data = response.json()

    # Each row is a dict and contains all expected columns
    assert all(isinstance(row, dict) for row in data)
    assert all(key in row for key in ["age", "salary", "city"] for row in data)
