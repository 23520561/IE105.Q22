from pydantic import BaseModel


class EncodingRequest(BaseModel):
    data: list[dict]
    method: str  # "one_hot", "label", "target", "count", "freq", "binary", "ordinal"
    columns: list[str] = []
    column: str = ""
    target: str = ""
    mapping: dict = {}
