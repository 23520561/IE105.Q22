from pydantic import BaseModel


class TransformRequest(BaseModel):
    data: list[dict]
    method: str  # "log", "sqrt", "minmax", "standard", "robust", "power", "normalize"
    columns: list[str]
