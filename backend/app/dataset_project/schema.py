from pydantic import BaseModel


class ProjectRequest(BaseModel):
    name: str
    dataset_id: str


class ProjectResponse(BaseModel):
    id: str
    name: str
    date: str
