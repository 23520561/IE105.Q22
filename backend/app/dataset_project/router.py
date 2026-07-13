from fastapi import APIRouter, Cookie

import app.dataset_project.service as Service
from app.dataset_project.schema import ProjectRequest

router = APIRouter(
    prefix="/project",
    tags=["dataset, project"],
    responses={404: {"description": "Not found"}},
)


@router.get("")
def get_projects(x_session_id: str = Cookie(...)):
    return Service.get_projects(x_session_id)


@router.post("")
def create_projects(req: ProjectRequest):
    Service.create_project(
        dataset_filename=req.dataset_filename, project_filename=req.project_filename
    )
    return "ok"
