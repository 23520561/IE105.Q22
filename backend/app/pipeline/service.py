from app.feature_transformation.schemas import TransformRequest
import json
from app.feature_encoding.schemas import EncodingRequest
from pathlib import Path
import pandas as pd
import app.feature_encoding.service as encoding
import app.feature_transformation.service as transformation

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = (BASE_DIR / "../../cached").resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_pipeline_path(dataset_id: str) -> Path:
    return UPLOAD_DIR / f"{dataset_id}_pipeline.json"


def load_pipeline(dataset_id: str) -> list[EncodingRequest | TransformRequest]:
    path = get_pipeline_path(dataset_id)
    if not path.exists():
        return []
    result: list[EncodingRequest | TransformRequest] = []
    for step in json.loads(path.read_text()):
        if "column" in step:
            result.append(EncodingRequest(**step))
        elif "columns" in step:
            result.append(TransformRequest(**step))

    return result


def save_pipeline(dataset_id: str, steps: list[EncodingRequest | TransformRequest]):
    path = get_pipeline_path(dataset_id)
    path.write_text(json.dumps([s.model_dump() for s in steps]))


def apply_pipeline(
    df: pd.DataFrame, steps: list[EncodingRequest | TransformRequest]
) -> pd.DataFrame:
    for step in steps:
        if isinstance(step, TransformRequest):
            if step.method == "minmax":
                df = transformation.minmax_scale(df, step.columns)

            elif step.method == "standard":
                df = transformation.standard_scale(df, step.columns)

            elif step.method == "robust":
                df = transformation.robust_scale(df, step.columns)

            elif step.method == "power":
                df = transformation.power_transform(df, step.columns)

            elif step.method == "normalize":
                df = transformation.normalize(df, step.columns)

        elif isinstance(step, EncodingRequest):
            if step.method == "one_hot":
                df = encoding.one_hot(df, step.column)

            elif step.method == "label":
                df = encoding.label_encode(df, step.column)

            elif step.method == "target":
                df = encoding.target_encode(df, step.column, step.target)

            elif step.method == "count":
                df = encoding.count_encode(df, step.column)

            elif step.method == "freq":
                df = encoding.freq_encode(df, step.column)

            elif step.method == "binary":
                df = encoding.binary_encode(df, step.column)

            elif step.method == "ordinal":
                df = encoding.ordinal_encode(df, step.column, step.mapping)
            elif step.method == "log":
                df = transformation.log_transform(df, step.columns)
            elif step.method == "sqrt":
                df = transformation.sqrt_transform(df, step.columns)

        else:
            raise ValueError(f"Unsupported method: {step.method}")

    return df
