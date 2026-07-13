from datetime import datetime
from app.dataset_project.schema import ProjectResponse
from app.dependencies.session_manager import get_session
from fastapi import HTTPException
from sklearn.utils import Bunch
from typing import cast
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = (BASE_DIR / "../../projects").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

projects_dict = {}


def get_projects(session_id: str):
    projects = []
    session = get_session(session_id)
    if session:
        for info in session.projects:
            name = info.stored_name
            file = UPLOAD_DIR / f"{name}.pkl"
            stat = file.stat()

            projects.append(
                ProjectResponse(
                    id=info.id,
                    name=info.name,
                    date=datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    size=stat.st_size,
                )
            )

    return projects


def get_df(dataset_id: str):
    UPLOAD_DIR = (BASE_DIR / "../../storage").resolve()
    csv_path = UPLOAD_DIR / f"{dataset_id}.csv"
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
    elif dataset_id == "breast":
        from sklearn.datasets import load_breast_cancer

        data: Bunch = cast(Bunch, load_breast_cancer())
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    else:
        raise HTTPException(status_code=404, detail="Dataset not found")


def create_project(project_filename, dataset_filename):
    print(dataset_filename)
    file_path = UPLOAD_DIR / f"{project_filename}.pkl"
    df: pd.DataFrame = get_df(dataset_filename)
    df.to_pickle(file_path)
