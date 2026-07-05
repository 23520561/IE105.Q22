from pydantic import BaseModel


class UploadedDataset(BaseModel):
    id: str
    name: str
    dateModified: str
    size: int
