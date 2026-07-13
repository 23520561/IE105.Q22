from pydantic import BaseModel


class ProjectRequest(BaseModel):
    project_filename: str
    dataset_filename: str


class ProjectResponse(BaseModel):
    id: str
    name: str
    date: str
