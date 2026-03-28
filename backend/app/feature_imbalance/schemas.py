from pydantic import BaseModel


class ImbalancedRequest(BaseModel):
    data: list[dict]
    target: str
    method: str = "smote"
