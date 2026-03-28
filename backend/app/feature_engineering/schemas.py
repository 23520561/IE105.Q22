from pydantic import BaseModel


class FeatureEngRequest(BaseModel):
    data: list[dict]
    operation: str  # e.g., "extract_datetime", "text_length", etc., "expression"
    column: str = ""
    new_col: str = ""
    params: dict = {}  # for additional parameters
